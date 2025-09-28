import re
import json
import base64
import os
from pathlib import Path
from fastapi import HTTPException
from typing import List, Literal
from enum import Enum

# Assuming these are configured in a 'config' module
from config import (
    anthropic_client, groq_client, openrouter_client, openai_client,
    GROQ_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY # Added OPENAI_API_KEY
)

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI # Needed for LangChain OpenAI/OpenRouter
from langchain_groq import ChatGroq # Needed for LangChain Groq
from anthropic import Anthropic
# Assuming 'openai' is imported and configured for the direct client
# from openai import OpenAI # Not strictly necessary if 'openai_client' is pre-configured

# --- Enums for Model Selection -----------------------------------------------
class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    OPENAI = "openai"

# --- Helpers -----------------------------------------------------------------
def _build_content_blocks(prompt: str, images: List[Path]) -> list:
    """
    Build Claude/Anthropic-compatible blocks (text/image base64).
    OpenAI's latest API can also use this format (list of dicts).
    """
    blocks = [{"type": "text", "text": prompt}]
    for img_path in images:
        media_type = "image/png" if str(img_path).lower().endswith(".png") else "image/jpeg"
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        blocks.append({
            "type": "image_url", # Changed 'image' to 'image_url' for compatibility with OpenAI's structure in list of dicts
            "image_url": {
                "url": f"data:{media_type};base64,{img_b64}",
            }
        })
    # For a client that strictly expects Anthropic's 'image' block type:
    # blocks = [{"type": "text", "text": prompt}]
    # for img_path in images:
    #     media_type = "image/png" if str(img_path).lower().endswith(".png") else "image/jpeg"
    #     with open(img_path, "rb") as f:
    #         img_b64 = base64.b64encode(f.read()).decode("utf-8")
    #     blocks.append({
    #         "type": "image",
    #         "source": {"type": "base64", "media_type": media_type, "data": img_b64}
    #     })
    return blocks

# ---------------- Main function ----------------
def get_llm_response(
    prompt: str, 
    system_prompt: str, 
    model_name: LLMProvider, # Required Enum parameter
    images: List[Path] | None = None
) -> str:
    """
    Calls a specific LLM provider based on the model_name.
    
    :param prompt: The user prompt.
    :param system_prompt: The system instruction.
    :param model_name: The LLMProvider to use (required).
    :param images: Optional list of image paths for multimodal calls.
    :return: The LLM's text response.
    """
    images = images or []
    
    # Content blocks for Anthropic/OpenAI multimodal calls
    content_blocks = _build_content_blocks(prompt, images) 
    
    # Groq-compatible user content (text-only, with image notifier)
    groq_user_content = prompt
    if images:
        groq_user_content = prompt + "\n\n[Note: Images were attached to the original request.]"

    # --- ANTHROPIC (Claude) ---
    if model_name == LLMProvider.ANTHROPIC:
        if 'anthropic_client' in globals() and anthropic_client is not None:
            try:
                print("--- Calling Anthropic API (Specified) ---")
                # Re-initialize the client here to ensure API key is used, or rely on the global/config
                anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
                client = Anthropic(api_key=anthropic_api_key)
                anth_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
                
                # Note: Anthropic uses the specific 'image' block format, 
                # which may require a slightly different helper function 
                # if full cross-compatibility with OpenAI is needed in _build_content_blocks.
                # Assuming the original _build_content_blocks is correct for Anthropic:
                anthropic_blocks = _build_content_blocks(prompt, images)
                
                response = client.messages.create(
                    model=anth_model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": system_prompt}],
                    system=system_prompt # Anthropic uses a separate system parameter
                )
                return response.content[0].text
            except Exception as e:
                print(f"Anthropic API failed: {e}")
                raise HTTPException(status_code=500, detail=f"Anthropic API failed: {e}")
        else:
            raise ValueError("Anthropic provider selected but not configured.")

    # --- OPENAI ---
    elif model_name == LLMProvider.OPENAI:
        if 'openai_client' in globals() and openai_client is not None:
            try:
                print("--- Calling OpenAI API (Specified) ---")
                model = os.getenv("OPENAI_MODEL", "gpt-4o-2024-05-13")
                
                resp = openai_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content_blocks}, # OpenAI uses the list of dicts format
                    ],
                    timeout=60,
                )
                return resp.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API failed: {e}")
                raise HTTPException(status_code=500, detail=f"OpenAI API failed: {e}")
        else:
            raise ValueError("OpenAI provider selected but not configured.")

    # --- OPENROUTER (OpenAI-Compatible) ---
    elif model_name == LLMProvider.OPENROUTER:
        if 'openrouter_client' in globals() and openrouter_client is not None:
            try:
                print("--- Calling OpenRouter API (Specified) via openrouter_client ---")
                model = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct:free")

                # openrouter_client should be an OpenAI-like client
                try:
                    resp = openrouter_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}, # Note: OpenRouter's support for image blocks varies by model, using text only here for safety unless a specific multimodal model is known.
                        ],
                        timeout=60,
                    )
                except TypeError:
                    # Fallback for clients not accepting 'timeout'
                    resp = openrouter_client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt},
                        ],
                    )

                # Adapt to OpenAI-like response shape
                try:
                    return resp.choices[0].message.content
                except Exception:
                    try:
                        return resp["choices"][0]["message"]["content"]
                    except Exception as e:
                        return str(resp)
            except Exception as e:
                print(f"OpenRouter API failed: {e}")
                raise HTTPException(status_code=500, detail=f"OpenRouter API failed: {e}")
        else:
            raise ValueError("OpenRouter provider selected but not configured.")

    # --- GROQ ---
    elif model_name == LLMProvider.GROQ:
        if 'groq_client' in globals() and groq_client is not None:
            try:
                print("--- Calling Groq API (Specified) ---")
                
                chat_completion = groq_client.chat.completions.create(
                    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": groq_user_content} # Using text-only content
                    ],
                )
                # Adapt to Groq SDK response shape
                return chat_completion.choices[0].message.content
            except Exception as e:
                print(f"Groq API failed: {e}")
                raise HTTPException(status_code=500, detail=f"Groq API failed: {e}")
        else:
            raise ValueError("Groq provider selected but not configured.")

    else:
        # Should be unreachable if Enum is used correctly, but good for completeness
        raise ValueError(f"Unknown model provider specified: {model_name}")


# ---------------- LangChain helper ----------------
def get_langchain_llm(model_name: LLMProvider): # Required Enum parameter
    """
    LangChain Chat LLM selection: returns the specified LLM client.
    """
    
    # --- ANTHROPIC (Claude) ---
    if model_name == LLMProvider.ANTHROPIC:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
        if anthropic_api_key:
            print("Initializing LangChain ChatAnthropic...")
            return ChatAnthropic(
                temperature=0.5,
                model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
                api_key=anthropic_api_key,
                max_tokens=4096
            )
        else:
            raise ValueError("Anthropic API key not found for ChatAnthropic.")

    # --- OPENAI ---
    elif model_name == LLMProvider.OPENAI:
        openai_api_key = os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY
        if openai_api_key:
            print("Initializing LangChain ChatOpenAI...")
            # LangChain's ChatOpenAI works with OpenAI's API
            return ChatOpenAI(
                temperature=0.5,
                model=os.getenv("OPENAI_MODEL", "gpt-4o-2024-05-13"),
                api_key=openai_api_key,
                max_tokens=4096,
            )
        else:
            raise ValueError("OpenAI API key not found for ChatOpenAI.")

    # --- OPENROUTER (LangChain's ChatOpenAI can be used) ---
    elif model_name == LLMProvider.OPENROUTER:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
        if openrouter_api_key:
            print("Initializing LangChain ChatOpenAI for OpenRouter...")
            # Use ChatOpenAI and override base_url
            return ChatOpenAI(
                temperature=0.5,
                model=os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct:free"),
                base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                api_key=openrouter_api_key,
                max_tokens=9000
            )
        else:
            raise ValueError("OpenRouter API key not found for LangChain OpenRouter.")

    # --- GROQ ---
    elif model_name == LLMProvider.GROQ:
        groq_api_key = os.getenv("GROQ_API_KEY") or GROQ_API_KEY
        if groq_api_key:
            print("Initializing ChatGroq...")
            return ChatGroq(
                temperature=0.5,
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                api_key=groq_api_key,
                max_tokens=9000
            )
        else:
            raise ValueError("Groq API key not found for ChatGroq.")
    
    else:
        raise ValueError(f"Unknown LLMProvider specified: {model_name}")


# --- Utility Functions (Kept as is) ------------------------------------------

def extract_json_from_response(text: str) -> dict:
    """
    Extracts JSON content robustly from an LLM response by finding the
    outermost curly braces or square brackets.
    """
    # Handle both object and array style JSON
    start_brace = text.find('{')
    end_brace = text.rfind('}')
    start_bracket = text.find('[')
    end_bracket = text.rfind(']')

    start = -1
    end = -1

    if start_brace != -1 and end_brace != -1:
        if start_bracket != -1 and start_bracket < start_brace:
            start = start_bracket
            end = end_bracket
        else:
            start = start_brace
            end = end_brace
    elif start_bracket != -1 and end_bracket != -1:
        start = start_bracket
        end = end_bracket
    else:
        raise ValueError("No valid JSON object or array found in the LLM response.")
    
    json_text = text[start:end+1]
    
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from LLM response: {e}")
        print(f"Attempted to parse text: {json_text}")
        raise ValueError("LLM did not return valid JSON.")

def extract_code_from_response(text: str) -> str:
    """
    Extracts a code block (Python, etc.) from a markdown-formatted string.
    """
    pattern = r"```(python)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(2).strip()
    raise ValueError("No code block found in the response.")
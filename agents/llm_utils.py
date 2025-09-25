import re
import json
import base64
from pathlib import Path
from fastapi import HTTPException
from config import anthropic_client, groq_client, GROQ_API_KEY, ANTHROPIC_API_KEY
import os
def get_llm_response(prompt: str, system_prompt: str, images: list[Path] = None) -> str:
    """
    Gets a response from an LLM.
    - First tries Groq (GPT-OSS).
    - Falls back to Anthropic Claude.
    - Supports optional vision (images) in both cases.
    """
    images = images or []

    # Build content payload (text + optional images)
    content = [{"type": "text", "text": prompt}]
    for img_path in images:
        with open(img_path, "rb") as f:
            img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode("utf-8")
        media_type = "image/png" if str(img_path).endswith(".png") else "image/jpeg"
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": img_base64
            }
        })

    # --- Priority 1: Groq ---
    if groq_client:
        try:
            print("--- Calling Groq API (Primary) ---")
            chat_completion = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq API failed: {e}. Falling back to Anthropic.")

    # --- Priority 2: Anthropic ---
    if anthropic_client:
        try:
            print("--- Calling Anthropic API (Fallback) ---")
            message = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4096,
                system=system_prompt,
                messages=[{"role": "user", "content": content}]
            )
            return message.content[0].text
        except Exception as e:
            print(f"Anthropic API failed: {e}")
            raise HTTPException(status_code=500, detail="Both LLM providers failed.")

    raise HTTPException(status_code=500, detail="No LLM providers are configured.")

def get_langchain_llm():
    """
    Returns a LangChain-compatible LLM instance by checking for API keys
    in environment variables, preferring Groq over Anthropic.
    """
    # Attempt to get the Groq API key from environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        print("Found GROQ_API_KEY. Initializing ChatGroq...")
        from langchain_groq import ChatGroq
        return ChatGroq(
            temperature=0.5,
            model="openai/gpt-oss-20b",
            api_key=groq_api_key
        )

    # If Groq key is not found, attempt to get the Anthropic API key
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        print("Found ANTHROPIC_API_KEY. Initializing ChatAnthropic...")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            temperature=0.5,
            model_name="claude-3-sonnet-20240229",
            api_key=anthropic_api_key
        )
    
    # If neither key is found, raise an error
    raise ValueError("No LLM API keys (GROQ_API_KEY or ANTHROPIC_API_KEY) found in environment variables.")

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
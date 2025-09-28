import re
import json
import base64
from pathlib import Path
from fastapi import HTTPException
from typing import List
from config import anthropic_client, groq_client, GROQ_API_KEY, ANTHROPIC_API_KEY
import os


# --- Helpers -----------------------------------------------------------------
def _build_content_blocks(prompt: str, images: List[Path]) -> list:
    """Build Claude/Anthropic-compatible blocks: first text, then image blocks (base64)."""
    blocks = [{"type": "text", "text": prompt}]
    for img_path in images:
        media_type = "image/png" if str(img_path).lower().endswith(".png") else "image/jpeg"
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": img_b64}
        })
    return blocks


def _extract_anthropic_text(message) -> str:
    """Try common Anthropic SDK response shapes and return the text string."""
    try:
        # Common pattern: message.content is a list and first item has .text or ["text"]
        if hasattr(message, "content") and isinstance(message.content, (list, tuple)):
            first = message.content[0]
            # object with attribute .text
            if hasattr(first, "text"):
                return first.text
            # dict-like
            if isinstance(first, dict) and "text" in first:
                return first["text"]
        # Sometimes SDK returns .text at top level
        if hasattr(message, "text"):
            return message.text
    except Exception:
        pass
    return str(message)


# ---------------- Main function ----------------
def get_llm_response(prompt: str, system_prompt: str, images: List[Path] | None = None) -> str:
    """
    Priority:
      1) Anthropic Sonnet (primary)  <-- configured by ANTHROPIC_MODEL
      2) Groq (fallback)
    """
    images = images or []
    content_blocks = _build_content_blocks(prompt, images)



    if 'groq_client' in globals() and groq_client is not None:
        try:
            print("--- Calling Groq API (Fallback) ---")
            # Groq in your codebase is text-first; if images exist we append a notifier
            user_content = prompt
            if images:
                user_content = prompt + "\n\n[Images were attached (handled by Anthropic in primary).]"

            chat_completion = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
            )
            # adapt to Groq SDK response shape
            try:
                return chat_completion.choices[0].message.content
            except Exception:
                return str(chat_completion)
        except Exception as e:
            print("Groq API failed:", e)

    # -------- Anthropic (PRIMARY) --------
    if 'anthropic_client' in globals() and anthropic_client is not None:
        try:
            print("--- Calling Anthropic API (Primary - Sonnet) ---")
            anth_model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4@20250514")

            # IMPORTANT: pass `system` as a top-level parameter (not as a message role)
            # Messages array should NOT contain {"role":"system", ...}
            resp = anthropic_client.messages.create(
                model=anth_model,
                max_tokens=4096,
                system=system_prompt,                       # top-level system
                messages=[{"role": "user", "content": content_blocks}],  # only user messages here
            )

            return _extract_anthropic_text(resp)

        except Exception as e:
            err_text = str(e)
            print("Anthropic primary call failed:", err_text)

            # If user accidentally used a system role in messages, print a clear hint
            if "Unexpected role \"system\"" in err_text or "Unexpected role 'system'" in err_text:
                print("Error hint: remove any messages with role='system' and pass your system prompt via the top-level `system=` parameter.")
            # If it's a model-not-found or other API error, try to show models (best-effort)
            try:
                status_code = getattr(e, "status_code", None) or getattr(e, "status", None)
            except Exception:
                status_code = None

            if status_code == 404 or "not_found" in err_text.lower():
                print("Anthropic model not found (404). Attempting to list available models:")
                try:
                    if hasattr(anthropic_client, "models") and hasattr(anthropic_client.models, "list"):
                        for page in anthropic_client.models.list(limit=100):
                            for m in getattr(page, "data", []) or []:
                                print("  -", getattr(m, "id", m))
                    else:
                        print("Cannot list models via SDK. Run: curl -H 'x-api-key: $ANTHROPIC_API_KEY' https://api.anthropic.com/v1/models")
                except Exception as list_e:
                    print("Failed to list Anthropic models:", list_e)
            # Fallthrough to Groq fallback
            print("Falling back to Groq (if available).")

    # -------- Groq (FALLBACK) --------
    

    # If we reach here, both providers failed or weren't configured
    raise HTTPException(status_code=500, detail="Both LLM providers failed or are not configured.")


# ---------------- LangChain helper ----------------
def get_langchain_llm():
    """
    LangChain Chat LLM selection: Anthropic Sonnet preferred, Groq fallback.
    Set ANTHROPIC_MODEL env var if you want to override the default sonnet id.
    """


    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        print("Found GROQ_API_KEY. Initializing ChatGroq (fallback)...")
        from langchain_groq import ChatGroq
        return ChatGroq(
            temperature=0.5,
            model="openai/gpt-oss-20b",
            api_key=groq_api_key
        )
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_api_key:
        print("Found ANTHROPIC_API_KEY. Initializing ChatAnthropic (Sonnet preferred)...")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            temperature=0.5,
            model_name=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4@20250514"),
            api_key=anthropic_api_key
        )

    

    raise ValueError("No LLM API keys (ANTHROPIC_API_KEY or GROQ_API_KEY) found in environment variables.")

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
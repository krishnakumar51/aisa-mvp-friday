import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file (you need a .env file with your keys)
load_dotenv()

# --- LLM API Keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Added OpenAI API Key

# --- LLM Model Defaults (Optional, can be set via ENV) ---
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct:free")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# --- Artifacts Directory (Phase 1 Change) ---
ARTIFACTS_DIR = Path("D:/project-mvp/ENHANCED/") / "AISA_TASKS"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)  # ensure the dir exists

# --- LLM Client Initialization ---
anthropic_client = None
groq_client = None
openrouter_client = None
openai_client = None # Added OpenAI Client

# Anthropic client
try:
    if ANTHROPIC_API_KEY:
        from anthropic import Anthropic
        # Using ANTHROPIC_MODEL is not typical for client init, but for convenience:
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        print("Anthropic client initialized successfully.")
except ImportError:
    print("Warning: 'anthropic' library not found. To use Anthropic, run 'pip install anthropic'.")
except Exception as e:
    print(f"Error initializing Anthropic client: {e}")

# Groq client
try:
    if GROQ_API_KEY:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized successfully.")
except ImportError:
    print("Warning: 'groq' library not found. To use Groq, run 'pip install groq'.")
except Exception as e:
    print(f"Error initializing Groq client: {e}")

# OpenAI client
try:
    if OPENAI_API_KEY:
        from openai import OpenAI
        # OpenAI client automatically picks up the API key from the environment/api_key arg
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("OpenAI client initialized successfully.")
except ImportError:
    print("Warning: 'openai' library not found. To use OpenAI, run 'pip install openai'.")
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")

# OpenRouter "client" (simple requests.Session with Authorization header)
try:
    if OPENROUTER_API_KEY:
        # Note: OpenRouter is often used with the official 'openai' library by setting
        # base_url, but your current setup uses a requests.Session. I'll stick to that
        # for compatibility with the rest of your original code structure.
        import requests
        session = requests.Session()
        session.headers.update({
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        })
        # Attach model default and base url for convenience
        session.openrouter_base_url = "https://openrouter.ai/api/v1"
        session.openrouter_default_model = OPENROUTER_MODEL
        openrouter_client = session
        print(f"OpenRouter client initialized (model={OPENROUTER_MODEL}).")
except ImportError:
    print("Warning: 'requests' library not found. To use OpenRouter, run 'pip install requests'.")
except Exception as e:
    print(f"Error initializing OpenRouter client: {e}")
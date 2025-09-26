import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- LLM API Keys ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Artifacts Directory (Phase 1 Change) ---
# This is the single most important change to prevent server reloads.
# We define a directory OUTSIDE the project folder to store all generated files.
# By default, it creates a folder in your user's home directory.
# You can change this path to whatever you like (e.g., "D:/AISA_TASKS").
ARTIFACTS_DIR = Path("D:/project-mvp/ENHANCED/") / "AISA_TASKS"

# --- LLM Client Initialization ---
# Initialize clients once here and import them wherever needed.
anthropic_client = None
groq_client = None

try:
    if ANTHROPIC_API_KEY:
        from anthropic import Anthropic
        anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
        print("Anthropic client initialized successfully.")
except ImportError:
    print("Warning: 'anthropic' library not found. To use Anthropic, run 'pip install anthropic'.")
except Exception as e:
    print(f"Error initializing Anthropic client: {e}")

try:
    if GROQ_API_KEY:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        print("Groq client initialized successfully.")
except ImportError:
    print("Warning: 'groq' library not found. To use Groq, run 'pip install groq'.")
except Exception as e:
    print(f"Error initializing Groq client: {e}")
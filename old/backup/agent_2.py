# agents/agent_2.py

from pathlib import Path
import json
from fastapi import HTTPException
from typing import TypedDict, List, Optional, Dict

from config import ARTIFACTS_DIR
from agents.llm_utils import get_llm_response, get_langchain_llm

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Pydantic models and TypedDict
class BlueprintStep(BaseModel):
    step_id: int
    screen_name: str
    description: str
    action: str
    target_element_description: str
    value_to_enter: Optional[str] = None
    associated_image: Optional[str] = None

class BlueprintSummary(BaseModel):
    overall_goal: str
    target_application: str
    platform: str

class AutomationBlueprint(BaseModel):
    summary: BlueprintSummary
    steps: List[BlueprintStep]

class CodeOutput(BaseModel):
    python_code: str = Field(description="The full, self-contained Python script as a string.")
    requirements: str = Field(description="The content of requirements.txt as a string, with one package per line.")

class Agent2Output(TypedDict):
    script: str
    requirements: str

# Full setup templates
MOBILE_SETUP_TEMPLATE = """
from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.actions import interaction
import time
from faker import Faker

def setup_driver(app_package, app_activity):
    print("Setting up Appium driver...")
    options = UiAutomator2Options()
    options.platform_name = 'Android'
    options.automation_name = 'UiAutomator2'
    options.app_package = app_package
    options.app_activity = app_activity
    options.no_reset = False
    options.full_reset = True
    options.new_command_timeout = 300
    options.auto_grant_permissions = True
    
    try:
        driver = webdriver.Remote("http://127.0.0.1:4723", options=options)
        print("✓ Driver is ready.")
        return driver
    except Exception as e:
        print(f"✗ Driver setup failed: {e}")
        return None
"""

WEB_SETUP_TEMPLATE = """
from playwright.sync_api import sync_playwright, Playwright, expect
import time
from faker import Faker

def run(playwright: Playwright) -> None:
    print("Setting up browser...")
    browser = playwright.chromium.launch(headless=False)
    context = browser.new_context(
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    page = context.new_page()
    page.set_default_timeout(60000)
    print("✓ Browser is ready.")
"""

def run_agent2(seq_no: str, blueprint_dict: dict) -> Agent2Output:
    print(f"[{seq_no}] Running Agent 2: Code Generation with Structured Output")
    
    try:
        blueprint = AutomationBlueprint(**blueprint_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid blueprint structure: {e}")

    out_dir = ARTIFACTS_DIR / seq_no / "agent2"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    platform = blueprint.summary.platform
    framework = "Appium with Selenium" if platform == "mobile" else "Playwright"
    setup_code = MOBILE_SETUP_TEMPLATE if platform == "mobile" else WEB_SETUP_TEMPLATE

    image_descriptions: Dict[str, str] = {}
    for step in blueprint.steps:
        if step.associated_image:
            img_path = ARTIFACTS_DIR / seq_no / "agent1" / step.associated_image
            if img_path.exists():
                description = get_llm_response("Describe this screenshot in extreme detail for automation scripting.", "You are an expert UI analyzer.", images=[img_path])
                image_descriptions[img_path.name] = description

    llm = get_langchain_llm()

    # --- KEY CHANGE: Corrected `tool_choice` format for Groq API ---
    structured_llm = llm.with_structured_output(CodeOutput).bind(
        tool_choice={"type": "function", "function": {"name": "CodeOutput"}}
    )

    system_prompt = f"""
    You are an expert-level automation script generator for {framework} on {platform}.
    Your goal is to write a complete, runnable Python script and a corresponding requirements.txt file based on the provided blueprint and image descriptions.
    The script MUST start with the mandatory setup code provided.
    Add detailed comments explaining each automation step, referencing the step number from the blueprint.
    Use the 'faker' library for any required test data generation.
    Ensure the script is self-contained and executable.
    You MUST output a JSON object that perfectly matches the provided schema.
    """

    human_prompt_template = """
    Please generate the Python script and requirements.txt content based on the following context.

    **Automation Blueprint:**
    ---
    {blueprint_json}
    ---
    **Image Descriptions (for visual context):**
    ---
    {image_descriptions_json}
    ---
    **MANDATORY Setup Code (Your script MUST begin with this):**
    ---
    ```python
    {setup_code}
    ```
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt_template),
    ])
    
    chain = prompt | structured_llm
    
    try:
        print(f"[{seq_no}] Invoking structured output chain for code generation...")
        
        input_variables = {
            "blueprint_json": blueprint.model_dump_json(indent=2),
            "image_descriptions_json": json.dumps(image_descriptions, indent=2),
            "setup_code": setup_code
        }

        code_obj: CodeOutput = chain.invoke(input_variables)
        script_code = code_obj.python_code
        requirements_text = code_obj.requirements

        if not script_code or not requirements_text:
            raise ValueError("LLM response was valid but missing 'python_code' or 'requirements' content.")

        script_path = out_dir / "automation_script.py"
        reqs_path = out_dir / "requirements.txt"
        
        script_path.write_text(script_code, encoding="utf-8")
        reqs_path.write_text(requirements_text, encoding="utf-8")
        
        print(f"[{seq_no}] Agent 2 finished successfully. Script and requirements generated.")
        return {"script": str(script_path), "requirements": str(reqs_path)}
        
    except Exception as e:
        print(f"[{seq_no}] Agent 2 failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 2 (Code Gen) failed: {e}")
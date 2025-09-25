# agents/agent_1.py

from pathlib import Path
import json
from fastapi import HTTPException
from typing import List, Optional

from config import ARTIFACTS_DIR
from agents.llm_utils import get_llm_response, get_langchain_llm

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Pydantic models for strict validation
class BlueprintSummary(BaseModel):
    overall_goal: str = Field(description="A brief one-sentence goal for the automation.")
    target_application: str = Field(description="The application identifier, e.g., 'com.google.android.gm' or a website URL.")
    platform: str = Field(description="The target platform, MUST be 'mobile' or 'web'.")

class BlueprintStep(BaseModel):
    step_id: int = Field(description="A unique integer identifier for the step, starting from 1.")
    screen_name: str = Field(description="The name of the screen or page where the action takes place.")
    description: str = Field(description="A detailed description of the action being performed.")
    action: str = Field(description="The specific action to take, e.g., 'click', 'type_text', 'scroll'.")
    target_element_description: str = Field(description="A description of the UI element to interact with.")
    value_to_enter: Optional[str] = Field(description="The text value to enter. Use null if not applicable.")
    associated_image: Optional[str] = Field(description="The filename of the associated image from the context. Use null if not applicable.")

class BlueprintOutput(BaseModel):
    summary: BlueprintSummary
    steps: List[BlueprintStep]

def run_agent1(seq_no: str, pdf_path: Path, instructions: str, platform: str) -> dict:
    print(f"[{seq_no}] Running Agent 1: Blueprint Generation with Structured Output")
    out_dir = ARTIFACTS_DIR / seq_no / "agent1"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_text_content, image_paths = "", []
    try:
        import fitz
        doc = fitz.open(pdf_path)
        for page_num, page in enumerate(doc):
            pdf_text_content += f"\n--- PDF Page {page_num + 1} Text ---\n{page.get_text()}"
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes, image_ext = base_image["image"], base_image["ext"]
                image_filename = f"page{page_num+1}_img{img_index}.{image_ext}"
                image_filepath = out_dir / image_filename
                image_filepath.write_bytes(image_bytes)
                image_paths.append(image_filepath)
    except Exception:
        pdf_text_content = "Could not read PDF. Relying on user instructions only."

    image_descriptions = {}
    for img_path in image_paths:
        description = get_llm_response("Describe this screenshot in detail for automation scripting.", "You are an expert UI analyzer.", images=[img_path])
        image_descriptions[img_path.name] = description

    llm = get_langchain_llm()
    
    # --- KEY CHANGE: Corrected `tool_choice` format for Groq API ---
    structured_llm = llm.with_structured_output(BlueprintOutput).bind(
        tool_choice={"type": "function", "function": {"name": "BlueprintOutput"}}
    )

    system_prompt = (
        "You are a master test automation planner. Your task is to create a comprehensive JSON blueprint for an automation script. "
        "The JSON object must have two top-level keys: 'summary' and 'steps'. "
        "The 'summary' object must contain: 'overall_goal', 'target_application', and 'platform'. "
        "The 'steps' key must be an array of objects, where each object represents a single action and includes: 'step_id' (as a simple integer), 'screen_name', "
        "'description', 'action' (e.g., 'click', 'type_text', 'press_and_hold'), 'target_element_description', "
        "'value_to_enter' (use null if not applicable), and 'associated_image' (filename, or null). "
        "You MUST output a JSON object that perfectly matches the provided schema. Do not add any extra commentary."
    )
    
    human_prompt_template = """
    Generate a complete JSON blueprint based on the following information.
    Use the provided images for visual context to accurately describe elements and actions.

    **Execution Platform:** {platform}
    **User Instructions:**
    ---
    {instructions}
    ---
    **Extracted PDF Text Content:**
    ---
    {pdf_text_content}
    ---
    **Available Image Files for Context (use these in 'associated_image'):**
    ---
    {image_filenames}
    ---
    **Image Descriptions:**
    ---
    {image_descriptions_json}
    ---
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt_template),
    ])

    chain = prompt | structured_llm

    try:
        print(f"[{seq_no}] Invoking structured output chain...")
        
        input_variables = {
            "platform": platform,
            "instructions": instructions,
            "pdf_text_content": pdf_text_content,
            "image_filenames": ", ".join([p.name for p in image_paths]),
            "image_descriptions_json": json.dumps(image_descriptions, indent=2)
        }
        
        blueprint_obj: BlueprintOutput = chain.invoke(input_variables)
        blueprint_dict = blueprint_obj.model_dump()
        
        blueprint_path = out_dir / "blueprint.json"
        blueprint_path.write_text(json.dumps(blueprint_dict, indent=2), encoding="utf-8")
        
        print(f"[{seq_no}] Agent 1 finished successfully. Blueprint created at {blueprint_path}")
        return blueprint_dict
    except Exception as e:
        print(f"[{seq_no}] Agent 1 failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent 1 (Blueprint) failed: {e}")
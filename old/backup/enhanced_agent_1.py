# agents/enhanced_agent_1.py

from pathlib import Path
import json
import re
from fastapi import HTTPException
from typing import List, Optional, Dict, Any, Tuple
import base64
from io import BytesIO
from PIL import Image
import hashlib

from config import ARTIFACTS_DIR
from agents.llm_utils import get_llm_response, get_langchain_llm

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Enhanced Pydantic models with validation
class BlueprintSummary(BaseModel):
    overall_goal: str = Field(description="A comprehensive goal statement for the automation task.")
    target_application: str = Field(description="The application identifier with version/context info.")
    platform: str = Field(description="The target platform, MUST be 'mobile' or 'web'.")
    estimated_duration: Optional[int] = Field(description="Estimated execution time in seconds.")
    complexity_indicators: List[str] = Field(description="List of complexity factors detected.")

class BlueprintStep(BaseModel):
    step_id: int = Field(description="A unique integer identifier for the step, starting from 1.")
    screen_name: str = Field(description="The name of the screen or page where the action takes place.")
    description: str = Field(description="A detailed description of the action being performed.")
    action: str = Field(description="The specific action to take, e.g., 'click', 'type_text', 'scroll'.")
    target_element_description: str = Field(description="Detailed description of the UI element with fallback selectors.")
    value_to_enter: Optional[str] = Field(description="The text value to enter. Use null if not applicable.")
    associated_image: Optional[str] = Field(description="The filename of the associated image from the context.")
    prerequisites: List[str] = Field(description="Conditions that must be met before this step.", default=[])
    expected_outcome: str = Field(description="What should happen after this step completes.")
    fallback_actions: List[str] = Field(description="Alternative actions if primary action fails.", default=[])
    timing_notes: Optional[str] = Field(description="Special timing considerations for this step.")

class EnhancedBlueprintOutput(BaseModel):
    summary: BlueprintSummary
    steps: List[BlueprintStep]
    metadata: Dict[str, Any] = Field(description="Additional metadata about the blueprint generation")

def analyze_pdf_content(pdf_text: str, images_count: int) -> Dict[str, Any]:
    """Analyze PDF content to extract automation insights"""
    analysis = {
        "content_type": "unknown",
        "ui_patterns": [],
        "complexity_factors": [],
        "estimated_steps": 0,
        "platform_hints": []
    }
    
    # Content type detection
    mobile_keywords = ["android", "ios", "mobile", "app", "appium", "device"]
    web_keywords = ["browser", "web", "website", "playwright", "selenium", "url"]
    
    text_lower = pdf_text.lower()
    
    if any(keyword in text_lower for keyword in mobile_keywords):
        analysis["platform_hints"].append("mobile")
    if any(keyword in text_lower for keyword in web_keywords):
        analysis["platform_hints"].append("web")
    
    # UI pattern detection
    ui_patterns = {
        "forms": ["form", "input", "field", "textbox", "password"],
        "navigation": ["menu", "tab", "navigation", "nav", "button"],
        "dialogs": ["dialog", "popup", "modal", "alert", "confirm"],
        "lists": ["list", "table", "grid", "row", "item"],
        "media": ["image", "video", "audio", "file", "upload"]
    }
    
    for pattern_name, keywords in ui_patterns.items():
        if any(keyword in text_lower for keyword in keywords):
            analysis["ui_patterns"].append(pattern_name)
    
    # Complexity detection
    complexity_keywords = {
        "high": ["captcha", "verification", "oauth", "login", "authentication", "payment"],
        "medium": ["dropdown", "calendar", "picker", "validation", "upload"],
        "dynamic": ["ajax", "loading", "wait", "async", "dynamic"]
    }
    
    for complexity_level, keywords in complexity_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            analysis["complexity_factors"].append(complexity_level)
    
    # Estimate steps based on content
    step_indicators = len(re.findall(r'\b(?:step|click|enter|select|navigate)\b', text_lower, re.IGNORECASE))
    analysis["estimated_steps"] = max(step_indicators, images_count, 3)
    
    return analysis

def enhance_image_analysis(image_path: Path, step_context: str = "") -> Dict[str, str]:
    """Enhanced image analysis with multiple perspectives"""
    
    enhanced_prompts = {
        "technical": f"""Analyze this screenshot for automation scripting with focus on:
1. EXACT element identifiers (IDs, classes, accessibility labels)
2. Element hierarchy and DOM structure clues
3. Interactive elements (buttons, inputs, dropdowns)
4. Loading states, animations, or dynamic content
5. Error states, validation messages, or alerts
6. Navigation elements and breadcrumbs
7. Form structure and field relationships
8. Mobile-specific elements (gestures, native controls)
Context: {step_context}
Provide technical details for robust element selection.""",

        "ux_patterns": f"""Analyze this UI screenshot for UX patterns and user flow:
1. User journey stage and context
2. Primary and secondary actions available
3. Visual hierarchy and information architecture
4. Responsive design elements
5. Accessibility considerations
6. Error prevention patterns
7. Progress indicators or step flows
8. User guidance elements (tooltips, hints)
Context: {step_context}
Focus on user experience and interaction patterns.""",

        "automation_strategy": f"""Analyze this screenshot for automation strategy:
1. Most reliable element selection strategies
2. Potential flaky elements or timing issues
3. Alternative interaction methods
4. Wait conditions and success criteria
5. Error handling scenarios
6. Cross-platform compatibility considerations
7. Performance impact factors
8. Maintenance and stability concerns
Context: {step_context}
Provide automation-specific insights and recommendations."""
    }
    
    results = {}
    for analysis_type, prompt in enhanced_prompts.items():
        try:
            result = get_llm_response(
                prompt,
                "You are an expert in UI/UX analysis and test automation with deep technical knowledge.",
                images=[image_path]
            )
            results[analysis_type] = result
        except Exception as e:
            results[analysis_type] = f"Analysis failed: {str(e)}"
    
    return results

def extract_pdf_with_ocr(pdf_path: Path) -> Tuple[str, List[Path], List[Dict]]:
    """Enhanced PDF extraction with OCR fallback and image analysis"""
    pdf_text_content = ""
    image_paths = []
    image_metadata = []
    
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        
        for page_num, page in enumerate(doc):
            # Extract text
            page_text = page.get_text()
            pdf_text_content += f"\n--- PDF Page {page_num + 1} Text ---\n{page_text}"
            
            # Extract images with metadata
            for img_index, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Generate unique filename with hash
                    image_hash = hashlib.md5(image_bytes).hexdigest()[:8]
                    image_filename = f"page{page_num+1}_img{img_index}_{image_hash}.{image_ext}"
                    
                    # Save image
                    out_dir = ARTIFACTS_DIR / "temp" / "agent1"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    image_filepath = out_dir / image_filename
                    image_filepath.write_bytes(image_bytes)
                    
                    # Collect metadata
                    img_metadata = {
                        "filename": image_filename,
                        "page": page_num + 1,
                        "index": img_index,
                        "size": len(image_bytes),
                        "format": image_ext,
                        "hash": image_hash
                    }
                    
                    image_paths.append(image_filepath)
                    image_metadata.append(img_metadata)
                    
                except Exception as img_error:
                    print(f"Failed to extract image {img_index} from page {page_num + 1}: {img_error}")
                    continue
        
        # OCR fallback for images if text is sparse
        if len(pdf_text_content.strip()) < 100 and image_paths:
            try:
                import pytesseract
                for img_path in image_paths[:3]:  # Process first 3 images
                    try:
                        ocr_text = pytesseract.image_to_string(Image.open(img_path))
                        if ocr_text.strip():
                            pdf_text_content += f"\n--- OCR from {img_path.name} ---\n{ocr_text}"
                    except Exception as ocr_error:
                        print(f"OCR failed for {img_path.name}: {ocr_error}")
            except ImportError:
                print("pytesseract not available for OCR fallback")
                
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        pdf_text_content = "Could not read PDF content. Relying on user instructions only."
    
    return pdf_text_content, image_paths, image_metadata

def create_enhanced_system_prompt(platform: str, pdf_analysis: Dict[str, Any]) -> str:
    """Create enhanced system prompt based on PDF analysis"""
    
    base_prompt = f"""You are an elite automation architect and test strategy expert specializing in {platform} automation.

EXPERTISE AREAS:
🔧 Technical Excellence: Deep knowledge of automation frameworks, design patterns, and best practices
📱 Platform Mastery: Expert in {platform}-specific automation challenges and solutions  
🎯 Strategy Design: Ability to create comprehensive, maintainable automation blueprints
🔍 Analysis Skills: Extract automation insights from visual and textual content
⚡ Performance Focus: Design for reliability, speed, and maintainability

CONTEXT ANALYSIS:
"""
    
    if pdf_analysis:
        base_prompt += f"""
Content Type: {pdf_analysis.get('content_type', 'Unknown')}
UI Patterns Detected: {', '.join(pdf_analysis.get('ui_patterns', []))}
Complexity Factors: {', '.join(pdf_analysis.get('complexity_factors', []))}
Estimated Steps: {pdf_analysis.get('estimated_steps', 'Unknown')}
Platform Hints: {', '.join(pdf_analysis.get('platform_hints', []))}
"""

    base_prompt += f"""

BLUEPRINT REQUIREMENTS:
1. Create a comprehensive automation blueprint with detailed step analysis
2. Include prerequisites, expected outcomes, and fallback strategies
3. Provide multiple element selection strategies for reliability
4. Add timing considerations and wait conditions
5. Identify potential failure points and recovery methods
6. Include complexity indicators and execution estimates
7. Design for maintainability and cross-environment compatibility
8. Follow modern automation best practices and design patterns

OUTPUT SPECIFICATION:
Generate a JSON object with 'summary', 'steps', and 'metadata' sections.
Each step must include: step_id, screen_name, description, action, target_element_description, 
value_to_enter, associated_image, prerequisites, expected_outcome, fallback_actions, timing_notes.

QUALITY STANDARDS:
- Production-ready blueprint suitable for enterprise automation
- Comprehensive error handling and recovery strategies
- Clear, actionable descriptions for code generation
- Optimal balance of reliability and performance
- Future-proof design patterns and practices
"""
    
    return base_prompt

def run_enhanced_agent1(seq_no: str, pdf_path: Path, instructions: str, platform: str) -> dict:
    """Enhanced Agent 1 with advanced blueprint generation capabilities"""
    print(f"[{seq_no}] 🚀 Running Enhanced Agent 1: Advanced Blueprint Generation")
    
    out_dir = ARTIFACTS_DIR / seq_no / "enhanced_agent1"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[{seq_no}] 📄 Extracting PDF content with OCR fallback...")
    pdf_text_content, image_paths, image_metadata = extract_pdf_with_ocr(pdf_path)
    
    # Move images to correct directory
    final_image_paths = []
    for img_path, metadata in zip(image_paths, image_metadata):
        final_path = out_dir / metadata["filename"]
        try:
            final_path.write_bytes(img_path.read_bytes())
            final_image_paths.append(final_path)
        except Exception as e:
            print(f"Failed to copy image {img_path.name}: {e}")
            final_image_paths.append(img_path)
    
    print(f"[{seq_no}] 🔍 Analyzing PDF content for automation insights...")
    pdf_analysis = analyze_pdf_content(pdf_text_content, len(final_image_paths))
    
    print(f"[{seq_no}] 🖼️ Performing enhanced image analysis...")
    enhanced_image_descriptions = {}
    for img_path in final_image_paths:
        step_context = f"Step analysis for {img_path.name} in {platform} automation"
        analysis_results = enhance_image_analysis(img_path, step_context)
        enhanced_image_descriptions[img_path.name] = analysis_results

    # Get enhanced LLM
    llm = get_langchain_llm()
    
    # Create structured output with proper tool_choice format
    structured_llm = llm.with_structured_output(EnhancedBlueprintOutput)

    # Create enhanced system prompt
    system_prompt = create_enhanced_system_prompt(platform, pdf_analysis)
    
    # Enhanced human prompt template
    human_prompt_template = """
🎯 BLUEPRINT GENERATION REQUEST

**Context Analysis:**
Platform: {platform}
Estimated Complexity: {complexity_factors}
UI Patterns: {ui_patterns}
Estimated Steps: {estimated_steps}

**User Instructions:**
{instructions}

**PDF Content Analysis:**
{pdf_text_content}

**Enhanced Image Analysis:**
{enhanced_image_descriptions_json}

**Image Metadata:**
{image_metadata_json}

**Generation Requirements:**
1. Create a comprehensive automation blueprint
2. Include detailed step analysis with prerequisites and outcomes
3. Provide multiple element selection strategies
4. Add timing considerations and error handling
5. Include complexity assessment and execution estimates
6. Design for reliability and maintainability
7. Follow modern automation architecture patterns

Generate a production-ready blueprint for enterprise automation.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt_template),
    ])

    chain = prompt | structured_llm

    try:
        print(f"[{seq_no}] 🤖 Invoking enhanced structured output chain...")
        
        input_variables = {
            "platform": platform,
            "complexity_factors": ", ".join(pdf_analysis.get("complexity_factors", [])),
            "ui_patterns": ", ".join(pdf_analysis.get("ui_patterns", [])),
            "estimated_steps": pdf_analysis.get("estimated_steps", "Unknown"),
            "instructions": instructions,
            "pdf_text_content": pdf_text_content,
            "enhanced_image_descriptions_json": json.dumps(enhanced_image_descriptions, indent=2),
            "image_metadata_json": json.dumps(image_metadata, indent=2)
        }
        
        print(f"[{seq_no}] 🧠 Generating enhanced blueprint...")
        blueprint_obj: EnhancedBlueprintOutput = chain.invoke(input_variables)
        blueprint_dict = blueprint_obj.model_dump()
        
        # Add generation metadata
        blueprint_dict["metadata"].update({
            "generation_timestamp": Path(__file__).stat().st_mtime,
            "pdf_analysis": pdf_analysis,
            "images_processed": len(final_image_paths),
            "enhanced_analysis": True,
            "agent_version": "enhanced_1.0"
        })
        
        # Save enhanced blueprint
        blueprint_path = out_dir / "blueprint.json"
        blueprint_path.write_text(json.dumps(blueprint_dict, indent=2), encoding="utf-8")
        
        # Save additional analysis files
        analysis_path = out_dir / "content_analysis.json"
        analysis_path.write_text(json.dumps(pdf_analysis, indent=2), encoding="utf-8")
        
        image_analysis_path = out_dir / "image_analysis.json"
        image_analysis_path.write_text(json.dumps(enhanced_image_descriptions, indent=2), encoding="utf-8")
        
        print(f"[{seq_no}] ✅ Enhanced Agent 1 completed successfully!")
        print(f"[{seq_no}] 📊 Generated {len(blueprint_dict['steps'])} automation steps")
        print(f"[{seq_no}] 🔍 Processed {len(final_image_paths)} images with enhanced analysis")
        print(f"[{seq_no}] 📁 Files created:")
        print(f"  - Blueprint: {blueprint_path}")
        print(f"  - Content Analysis: {analysis_path}")
        print(f"  - Image Analysis: {image_analysis_path}")
        
        return blueprint_dict
        
    except Exception as e:
        print(f"[{seq_no}] ❌ Enhanced Agent 1 failed: {e}")
        raise HTTPException(status_code=500, detail=f"Enhanced Agent 1 failed: {e}")

# Export main function with backward compatibility
def run_agent1(seq_no: str, pdf_path: Path, instructions: str, platform: str) -> dict:
    """Backward compatible wrapper for enhanced agent 1"""
    return run_enhanced_agent1(seq_no, pdf_path, instructions, platform)

# Additional utility functions
def validate_blueprint_structure(blueprint_dict: dict) -> List[str]:
    """Validate blueprint structure and return issues"""
    issues = []
    
    # Check required top-level keys
    required_keys = ["summary", "steps", "metadata"]
    for key in required_keys:
        if key not in blueprint_dict:
            issues.append(f"Missing required key: {key}")
    
    # Validate steps structure
    if "steps" in blueprint_dict and isinstance(blueprint_dict["steps"], list):
        for i, step in enumerate(blueprint_dict["steps"]):
            if not isinstance(step, dict):
                issues.append(f"Step {i} is not a dictionary")
                continue
                
            required_step_keys = ["step_id", "action", "target_element_description"]
            for key in required_step_keys:
                if key not in step:
                    issues.append(f"Step {i} missing required key: {key}")
    
    return issues

__all__ = ["run_enhanced_agent1", "run_agent1", "analyze_pdf_content", "enhance_image_analysis"]
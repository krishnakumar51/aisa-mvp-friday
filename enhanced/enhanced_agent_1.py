# agents/enhanced_agent_1.py

from pathlib import Path
import json
import re
import time
from fastapi import HTTPException
from typing import List, Optional, Dict, Any, Tuple, Union, Set
import base64
from io import BytesIO
from PIL import Image
import hashlib

from config import ARTIFACTS_DIR
from agents.llm_utils import get_llm_response, get_langchain_llm

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator

# Advanced caching system to reduce LLM calls
class LLMCache:
    """Smart caching system for LLM responses with TTL and content hashing"""
    def __init__(self, default_ttl: int = 7200):  # 2 hours
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.hit_count = 0
        self.miss_count = 0

    def _create_cache_key(self, prompt: str, images: List = None, system_message: str = "") -> str:
        """Create a unique cache key from prompt content and images"""
        content_hash = hashlib.sha256()
        content_hash.update(prompt.encode('utf-8'))
        content_hash.update(system_message.encode('utf-8'))
        
        if images:
            for img in images:
                if hasattr(img, 'read_bytes'):
                    content_hash.update(str(img.stat().st_mtime).encode())
                    content_hash.update(str(img.stat().st_size).encode())
                else:
                    content_hash.update(str(img).encode())
        
        return content_hash.hexdigest()

    def get(self, prompt: str, images: List = None, system_message: str = "") -> Optional[Any]:
        """Get cached response if available and not expired"""
        cache_key = self._create_cache_key(prompt, images, system_message)
        
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() < entry["expires_at"]:
                self.hit_count += 1
                return entry["value"]
            else:
                del self._cache[cache_key]
        
        self.miss_count += 1
        return None

    def set(self, prompt: str, value: Any, images: List = None, system_message: str = "", ttl: Optional[int] = None):
        """Cache a response with TTL"""
        cache_key = self._create_cache_key(prompt, images, system_message)
        expires_at = time.time() + (ttl or self.default_ttl)
        
        self._cache[cache_key] = {
            "value": value,
            "expires_at": expires_at,
            "created_at": time.time()
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) * 100 if total_requests > 0 else 0
        
        return {
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate_percent": hit_rate,
            "cache_size": len(self._cache)
        }

# Global cache instance
agent1_cache = LLMCache(default_ttl=7200)

# Enhanced Pydantic models with validation
class BlueprintSummary(BaseModel):
    overall_goal: str = Field(description="A comprehensive goal statement for the automation task.")
    target_application: str = Field(description="The application identifier with version/context info.")
    platform: str = Field(description="The target platform, MUST be 'mobile' or 'web'.")
    estimated_duration: Optional[Union[int, str]] = Field(description="Estimated execution time, can be seconds or a descriptive range.")
    complexity_indicators: List[str] = Field(description="List of complexity factors detected.")

class BlueprintStep(BaseModel):
    step_id: int = Field(description="A unique integer identifier for the step, starting from 1.")
    screen_name: str = Field(description="The name of the screen or page where the action takes place.")
    description: str = Field(description="A detailed description of the action being performed.")
    action: str = Field(description="The specific action to take, e.g., 'click', 'type_text', 'scroll'.")
    target_element_description: Optional[Union[str, dict]] = Field(default="", description="Detailed description of the UI element with fallback selectors.")
    value_to_enter: Optional[Union[str, dict]] = Field(description="The text value to enter or complex selection. Use null if not applicable.")
    associated_image: Optional[str] = Field(default=None, description="The filename of the associated image from the context.")
    prerequisites: List[str] = Field(description="Conditions that must be met before this step.", default=[])
    expected_outcome: Union[str, dict] = Field(description="What should happen after this step completes.")
    fallback_actions: List[str] = Field(description="Alternative actions if primary action fails.", default=[])
    timing_notes: Optional[Union[str, dict]] = Field(description="Special timing considerations for this step.")
    
    @model_validator(mode="before")
    def coerce_nulls_to_defaults(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run *before* normal validation. Convert None/missing fields to safe defaults so
        LLM-produced nulls don't cause Pydantic validation errors.
        """
        # Guard: values may sometimes not be a dict (leave it unchanged)
        if not isinstance(values, dict):
            return values

        # String fields -> "" if None/missing
        string_fields = ["screen_name", "description", "action", "target_element_description", "expected_outcome"]
        for fld in string_fields:
            if values.get(fld) is None:
                values[fld] = ""

        # List fields -> [] if None/missing
        list_fields = ["prerequisites", "fallback_actions"]
        for lf in list_fields:
            if values.get(lf) is None:
                values[lf] = []

        # Keep timing_notes/value_to_enter/associated_image as-is (allow None)
        return values
    
    @model_validator(mode="before")
    def coerce_types(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce/flatten common structured fields from LLM into simple primitives to avoid validation failures."""
        if not isinstance(values, dict):
            return values

        # Step id coerced to int when possible
        if "step_id" in values:
            try:
                values["step_id"] = int(values["step_id"])
            except Exception:
                pass

        # If target_element_description is object -> stringify intelligently
        ted = values.get("target_element_description")
        if isinstance(ted, dict):
            # Prefer explicit 'description' key if present
            if "description" in ted and isinstance(ted["description"], str):
                values["target_element_description"] = ted["description"]
            else:
                # Fallback: join keys / values
                try:
                    flat = []
                    for k, v in ted.items():
                        flat.append(f"{k}:{v}")
                    values["target_element_description"] = " || ".join(flat)
                except Exception:
                    values["target_element_description"] = str(ted)

        # If expected_outcome is dict -> stringify
        eo = values.get("expected_outcome")
        if isinstance(eo, dict):
            if "description" in eo and isinstance(eo["description"], str):
                values["expected_outcome"] = eo["description"]
            else:
                values["expected_outcome"] = json.dumps(eo, ensure_ascii=False)

        # Ensure lists are lists
        if values.get("prerequisites") is None:
            values["prerequisites"] = []
        if values.get("fallback_actions") is None:
            values["fallback_actions"] = []

        return values

class EnhancedBlueprintOutput(BaseModel):
    summary: BlueprintSummary
    steps: List[BlueprintStep]
    metadata: Dict[str, Any] = Field(description="Additional metadata about the blueprint generation")

# Agent scratchpad for reflection and memory
class Agent1Scratchpad:
    """Memory system for Agent 1 to reduce redundant processing and enable reflection"""
    def __init__(self):
        self.content_analysis_cache: Dict[str, Dict[str, Any]] = {}
        self.image_analysis_cache: Dict[str, Dict[str, str]] = {}
        self.reflection_log: List[Dict[str, Any]] = []
        self.processed_patterns: Set[str] = set()
    
    def add_content_analysis(self, content_hash: str, analysis: Dict[str, Any]):
        """Cache content analysis to avoid reprocessing similar PDFs"""
        self.content_analysis_cache[content_hash] = analysis
    
    def get_content_analysis(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached content analysis"""
        return self.content_analysis_cache.get(content_hash)
    
    def add_image_analysis(self, image_path: str, analysis: Dict[str, str]):
        """Cache image analysis to avoid reprocessing similar images"""
        self.image_analysis_cache[image_path] = analysis
    
    def get_image_analysis(self, image_path: str) -> Optional[Dict[str, str]]:
        """Retrieve cached image analysis"""
        return self.image_analysis_cache.get(image_path)
    
    def reflect_on_complexity(self, pdf_analysis: Dict[str, Any], images_count: int) -> Dict[str, Any]:
        """Reflect on the complexity without making additional LLM calls"""
        reflection = {
            "complexity_score": 1,
            "confidence_level": "high",
            "reasoning": [],
            "optimization_suggestions": []
        }
        
        # Rule-based complexity assessment (no LLM needed)
        complexity_factors = pdf_analysis.get("complexity_factors", [])
        ui_patterns = pdf_analysis.get("ui_patterns", [])
        
        # Calculate complexity score
        base_score = 2
        if "high" in complexity_factors:
            base_score += 3
        if "medium" in complexity_factors:
            base_score += 2
        if "dynamic" in complexity_factors:
            base_score += 2
        
        # Pattern-based complexity
        if len(ui_patterns) > 3:
            base_score += 1
        if images_count > 5:
            base_score += 1
        
        reflection["complexity_score"] = min(base_score, 10)
        
        # Add reasoning without LLM calls
        if reflection["complexity_score"] >= 7:
            reflection["reasoning"].append("High complexity detected due to verification/authentication steps")
            reflection["optimization_suggestions"].extend([
                "Implement comprehensive error handling",
                "Add retry mechanisms with exponential backoff",
                "Use multiple element selection strategies"
            ])
        
        self.reflection_log.append({
            "timestamp": time.time(),
            "action": "complexity_reflection",
            "data": reflection
        })
        
        return reflection

# Global scratchpad instance
agent1_scratchpad = Agent1Scratchpad()

def get_llm_response_with_cache(prompt: str, system_message: str = "", images: List = None, **kwargs) -> str:
    """Cached LLM response wrapper to reduce API calls"""
    # Check cache first
    cached_response = agent1_cache.get(prompt, images, system_message)
    if cached_response:
        print("🎯 Cache HIT - Skipping LLM call")
        return cached_response
    
    print("🔄 Cache MISS - Making LLM call")
    
    # Make LLM call
    response = get_llm_response(prompt, system_message, images=images, **kwargs)
    
    # Cache the response
    agent1_cache.set(prompt, response, images, system_message)
    
    return response

def analyze_pdf_content_smart(pdf_text: str, images_count: int) -> Dict[str, Any]:
    """Enhanced PDF analysis with reduced LLM dependency and intelligent caching"""
    
    # Create content hash for caching
    content_hash = hashlib.md5(pdf_text.encode()).hexdigest()
    
    # Check scratchpad cache first
    cached_analysis = agent1_scratchpad.get_content_analysis(content_hash)
    if cached_analysis:
        print("📋 Using cached content analysis")
        return cached_analysis
    
    analysis = {
        "content_type": "unknown",
        "ui_patterns": [],
        "complexity_factors": [],
        "estimated_steps": 0,
        "platform_hints": [],
        "confidence_score": 0.8
    }

    # Enhanced content type detection with more keywords
    mobile_keywords = ["android", "ios", "mobile", "app", "appium", "device", "smartphone", "tablet", "touch", "gesture"]
    web_keywords = ["browser", "web", "website", "playwright", "selenium", "url", "chrome", "firefox", "dom", "javascript"]
    
    text_lower = pdf_text.lower()
    
    mobile_matches = sum(1 for keyword in mobile_keywords if keyword in text_lower)
    web_matches = sum(1 for keyword in web_keywords if keyword in text_lower)
    
    if mobile_matches > web_matches:
        analysis["platform_hints"].append("mobile")
        analysis["confidence_score"] = min(0.9, 0.6 + (mobile_matches * 0.05))
    elif web_matches > mobile_matches:
        analysis["platform_hints"].append("web")
        analysis["confidence_score"] = min(0.9, 0.6 + (web_matches * 0.05))
    
    # Enhanced UI pattern detection
    ui_patterns = {
        "forms": ["form", "input", "field", "textbox", "password", "email", "submit", "register", "login"],
        "navigation": ["menu", "tab", "navigation", "nav", "button", "link", "breadcrumb", "sidebar"],
        "dialogs": ["dialog", "popup", "modal", "alert", "confirm", "notification", "toast"],
        "lists": ["list", "table", "grid", "row", "item", "card", "tile", "carousel"],
        "media": ["image", "video", "audio", "file", "upload", "download", "gallery", "photo"]
    }

    for pattern_name, keywords in ui_patterns.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        if matches >= 2:  # Require at least 2 keyword matches
            analysis["ui_patterns"].append(pattern_name)

    # Enhanced complexity detection with scoring
    complexity_keywords = {
        "high": ["captcha", "verification", "oauth", "login", "authentication", "payment", "2fa", "biometric"],
        "medium": ["dropdown", "calendar", "picker", "validation", "upload", "search", "filter", "sort"],
        "dynamic": ["ajax", "loading", "wait", "async", "dynamic", "lazy", "infinite scroll", "realtime"]
    }

    for complexity_level, keywords in complexity_keywords.items():
        matches = sum(1 for keyword in keywords if keyword in text_lower)
        if matches > 0:
            analysis["complexity_factors"].append(complexity_level)

    # Smarter step estimation based on content analysis
    step_indicators = len(re.findall(r'\b(?:step|click|enter|select|navigate|tap|swipe|scroll)\b', text_lower, re.IGNORECASE))
    action_words = len(re.findall(r'\b(?:fill|check|uncheck|choose|pick|drag|drop)\b', text_lower, re.IGNORECASE))
    
    base_steps = max(step_indicators + action_words, images_count, 3)
    
    # Adjust based on complexity
    if "high" in analysis["complexity_factors"]:
        base_steps = int(base_steps * 1.5)
    if "dynamic" in analysis["complexity_factors"]:
        base_steps = int(base_steps * 1.3)
    
    analysis["estimated_steps"] = base_steps
    
    # Cache the analysis in scratchpad
    agent1_scratchpad.add_content_analysis(content_hash, analysis)
    
    return analysis

def enhance_image_analysis_smart(image_path: Path, step_context: str = "") -> Dict[str, str]:
    """Smart image analysis with caching and batch processing to reduce LLM calls"""
    
    # Check scratchpad cache first
    cached_analysis = agent1_scratchpad.get_image_analysis(str(image_path))
    if cached_analysis:
        print(f"🖼️ Using cached image analysis for {image_path.name}")
        return cached_analysis

    # Batch multiple analysis types into a single LLM call instead of 3 separate calls
    comprehensive_prompt = f"""
    Analyze this UI screenshot comprehensively for automation scripting. Provide analysis for ALL three areas below in a structured format:

    **TECHNICAL ANALYSIS:**
    - EXACT element identifiers (IDs, classes, accessibility labels)
    - Element hierarchy and DOM structure clues
    - Interactive elements (buttons, inputs, dropdowns)
    - Loading states, animations, or dynamic content
    - Error states, validation messages, or alerts

    **UX PATTERNS ANALYSIS:**
    - User journey stage and context
    - Primary and secondary actions available
    - Visual hierarchy and information architecture
    - Accessibility considerations
    - Progress indicators or step flows

    **AUTOMATION STRATEGY:**
    - Most reliable element selection strategies
    - Potential flaky elements or timing issues
    - Alternative interaction methods
    - Wait conditions and success criteria
    - Error handling scenarios

    Context: {step_context}

    Format your response as JSON with keys: "technical", "ux_patterns", "automation_strategy"
    """

    try:
        # Single LLM call instead of three
        combined_result = get_llm_response_with_cache(
            comprehensive_prompt,
            "You are an expert in UI/UX analysis and test automation with deep technical knowledge.",
            images=[image_path]
        )
        
        # Try to parse as JSON, fallback to text sections
        try:
            parsed_result = json.loads(combined_result)
            results = {
                "technical": parsed_result.get("technical", "Analysis not available"),
                "ux_patterns": parsed_result.get("ux_patterns", "Analysis not available"),
                "automation_strategy": parsed_result.get("automation_strategy", "Analysis not available")
            }
        except json.JSONDecodeError:
            # Fallback: split text by sections
            results = {
                "technical": combined_result[:len(combined_result)//3],
                "ux_patterns": combined_result[len(combined_result)//3:2*len(combined_result)//3],
                "automation_strategy": combined_result[2*len(combined_result)//3:]
            }
        
        # Cache the analysis
        agent1_scratchpad.add_image_analysis(str(image_path), results)
        
        return results

    except Exception as e:
        error_result = {
            "technical": f"Analysis failed: {str(e)}",
            "ux_patterns": f"Analysis failed: {str(e)}",
            "automation_strategy": f"Analysis failed: {str(e)}"
        }
        
        # Still cache the error to avoid repeated failures
        agent1_scratchpad.add_image_analysis(str(image_path), error_result)
        
        return error_result

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

def create_enhanced_system_prompt_smart(platform: str, pdf_analysis: Dict[str, Any], reflection: Dict[str, Any]) -> str:
    """Create enhanced system prompt with reflection insights to guide LLM thinking"""
    
    complexity_score = reflection.get("complexity_score", 5)
    confidence = reflection.get("confidence_level", "medium")
    
    base_prompt = f"""You are an elite automation architect specializing in {platform} automation with deep technical expertise.

🧠 COGNITIVE FRAMEWORK:
- Think step-by-step before generating each blueprint component
- Consider multiple approaches and select the most robust one
- Apply lessons learned from similar automation patterns
- Validate each step against real-world implementation challenges

🔧 TECHNICAL EXCELLENCE:
- Modern automation frameworks and design patterns
- Production-grade error handling and retry mechanisms
- Performance optimization and reliability patterns
- Cross-platform compatibility considerations

📊 CONTEXT ANALYSIS:
Platform: {platform}
Complexity Score: {complexity_score}/10 (Confidence: {confidence})
UI Patterns: {', '.join(pdf_analysis.get('ui_patterns', []))}
Key Challenges: {', '.join(pdf_analysis.get('complexity_factors', []))}
Estimated Steps: {pdf_analysis.get('estimated_steps', 'Unknown')}

🎯 BLUEPRINT REQUIREMENTS:
1. Create a comprehensive automation blueprint with detailed step analysis
2. Include prerequisites, expected outcomes, and fallback strategies  
3. Provide multiple element selection strategies for reliability
4. Add timing considerations and wait conditions
5. Identify potential failure points and recovery methods
6. Include complexity indicators and execution estimates
7. Design for maintainability and cross-environment compatibility
8. Follow modern automation best practices and design patterns

⚡ OPTIMIZATION INSIGHTS:
{chr(10).join(f"- {suggestion}" for suggestion in reflection.get('optimization_suggestions', []))}

Generate a production-ready blueprint that demonstrates expert-level automation design."""

    if complexity_score >= 7:
        base_prompt += f"""

⚠️ HIGH COMPLEXITY DETECTED:
This automation involves complex interactions. Implement:
- Advanced retry logic with exponential backoff
- Multiple element finding strategies  
- Comprehensive error recovery mechanisms
- Detailed step-by-step logging
- Performance benchmarking
- Fallback interaction methods
"""

    return base_prompt

def run_enhanced_agent1(seq_no: str, pdf_path: Path, instructions: str, platform: str) -> dict:
    """Enhanced Agent 1 with advanced blueprint generation, caching, and reflection capabilities"""
    start_time = time.time()  
    print(f"[{seq_no}] 🚀 Running Enhanced Agent 1: Advanced Blueprint Generation with Smart Caching")

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

    print(f"[{seq_no}] 🔍 Analyzing PDF content with smart caching...")
    pdf_analysis = analyze_pdf_content_smart(pdf_text_content, len(final_image_paths))

    print(f"[{seq_no}] 🤔 Applying reflection patterns...")
    reflection = agent1_scratchpad.reflect_on_complexity(pdf_analysis, len(final_image_paths))
    
    print(f"[{seq_no}] 🖼️ Performing optimized image analysis...")
    enhanced_image_descriptions = {}
    
    # Process images in smaller batches to avoid overwhelming context
    for img_path in final_image_paths:
        step_context = f"Step analysis for {img_path.name} in {platform} automation"
        analysis_results = enhance_image_analysis_smart(img_path, step_context)
        enhanced_image_descriptions[img_path.name] = analysis_results

    # Get enhanced LLM
    llm = get_langchain_llm()
    structured_llm = llm.with_structured_output(EnhancedBlueprintOutput)

    # Create enhanced system prompt with reflection
    system_prompt = create_enhanced_system_prompt_smart(platform, pdf_analysis, reflection)

    # Enhanced human prompt template with better organization
    human_prompt_template = """
🎯 BLUEPRINT GENERATION REQUEST

**Reflection-Driven Analysis:**
Complexity Score: {complexity_score}/10
Confidence Level: {confidence_level} 
Key Insights: {key_insights}

**Context Analysis:**
Platform: {platform}
Target Application: Extracted from PDF content
UI Patterns Detected: {ui_patterns}
Estimated Steps: {estimated_steps}

**User Instructions:**
{instructions}

**PDF Content Analysis:**
{pdf_text_content}

**Enhanced Image Analysis:**
{enhanced_image_descriptions_json}

**Image Metadata:**
{image_metadata_json}

**GENERATION REQUIREMENTS:**
1. Apply reflection insights to create robust automation steps
2. Use the complexity analysis to determine appropriate error handling
3. Include detailed step analysis with prerequisites and outcomes
4. Provide multiple element selection strategies based on image analysis
5. Add timing considerations and error handling appropriate to complexity level
6. Include comprehensive fallback strategies for high-complexity scenarios
7. Design for reliability, maintainability, and expert-level implementation

Generate a production-ready blueprint that reflects deep automation expertise.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt_template),
    ])

    chain = prompt | structured_llm

    try:
        print(f"[{seq_no}] 🤖 Invoking enhanced structured output chain with reflection...")
        
        input_variables = {
            "complexity_score": reflection["complexity_score"],
            "confidence_level": reflection["confidence_level"],
            "key_insights": "; ".join(reflection.get("reasoning", [])),
            "platform": platform,
            "ui_patterns": ", ".join(pdf_analysis.get("ui_patterns", [])),
            "estimated_steps": pdf_analysis.get("estimated_steps", "Unknown"),
            "instructions": instructions,
            "pdf_text_content": pdf_text_content,
            "enhanced_image_descriptions_json": json.dumps(enhanced_image_descriptions, indent=2),
            "image_metadata_json": json.dumps(image_metadata, indent=2)
        }

        print(f"[{seq_no}] 🧠 Generating enhanced blueprint with complexity score {reflection['complexity_score']}...")
        
        blueprint_obj: EnhancedBlueprintOutput = chain.invoke(input_variables)
        blueprint_dict = blueprint_obj.model_dump()

        # Add generation metadata with reflection and cache stats
        blueprint_dict.setdefault("metadata", {})
        blueprint_dict["metadata"].update({
            "generation_timestamp": time.time(),
            "pdf_analysis": pdf_analysis,
            "reflection_analysis": reflection,
            "images_processed": len(final_image_paths),
            "enhanced_analysis": True,
            "agent_version": "enhanced_1.1_optimized",
            "cache_stats": agent1_cache.get_stats(),
            "scratchpad_entries": len(agent1_scratchpad.reflection_log)
        })

        blueprint_dict.setdefault("metadata", {})
        for step in blueprint_dict["steps"]:
            if step.get("target_element_description") is None:
                step["target_element_description"] = ""

        # Save enhanced blueprint
        blueprint_path = out_dir / "blueprint.json"
        blueprint_path.write_text(json.dumps(blueprint_dict, indent=2), encoding="utf-8")

        # Save additional analysis files
        analysis_path = out_dir / "content_analysis.json"
        analysis_path.write_text(json.dumps(pdf_analysis, indent=2), encoding="utf-8")

        reflection_path = out_dir / "reflection_analysis.json"  
        reflection_path.write_text(json.dumps(reflection, indent=2), encoding="utf-8")

        image_analysis_path = out_dir / "image_analysis.json"
        image_analysis_path.write_text(json.dumps(enhanced_image_descriptions, indent=2), encoding="utf-8")

        cache_stats_path = out_dir / "cache_performance.json"
        cache_stats_path.write_text(json.dumps(agent1_cache.get_stats(), indent=2), encoding="utf-8")

        print(f"[{seq_no}] ✅ Enhanced Agent 1 completed successfully!")
        print(f"[{seq_no}] 📊 Generated {len(blueprint_dict['steps'])} automation steps")
        print(f"[{seq_no}] 🔍 Processed {len(final_image_paths)} images with smart caching")
        print(f"[{seq_no}] 🎯 Cache performance: {agent1_cache.get_stats()['hit_rate_percent']:.1f}% hit rate")
        print(f"[{seq_no}] 📁 Files created:")
        print(f"   - Blueprint: {blueprint_path}")
        print(f"   - Content Analysis: {analysis_path}")
        print(f"   - Reflection Analysis: {reflection_path}")
        print(f"   - Image Analysis: {image_analysis_path}")
        print(f"   - Cache Performance: {cache_stats_path}")


        result = {
            "status": "success",
            "sequence_number": seq_no,
            "blueprint_path": str(blueprint_path),
            "blueprint_filename": "blueprint.json",
            "steps_generated": len(blueprint_dict.get("steps", [])),
            "complexity_score": blueprint_dict["metadata"].get("complexity_score", 0.5),
            "execution_time": time.time() - start_time,
            "cache_stats": agent1_cache.get_stats(),
            "insights_learned": len(agent1_scratchpad.reflection_log),
            "blueprint_data": blueprint_dict
        }

        return result

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

__all__ = ["run_enhanced_agent1", "run_agent1", "analyze_pdf_content_smart", "enhance_image_analysis_smart", "agent1_cache", "agent1_scratchpad"]
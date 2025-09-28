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

# Assuming config and llm_utils are available in the environment
# DO NOT include the content of config.py or llm_utils.py here, just the imports.
# Placeholder imports to ensure the file is complete for copy-pasting
try:
    from config import ARTIFACTS_DIR
    from agents.llm_utils import get_llm_response, get_langchain_llm, LLMProvider
except ImportError:
    # Fallback for environments without the full project structure
    ARTIFACTS_DIR = Path("./artifacts")
    class LLMProvider:
        GROQ = "groq"
    # Mock implementations for the imports if the environment is isolated
    def get_llm_response(prompt, system_message, model_name, images, **kwargs):
        raise NotImplementedError("get_llm_response not available in isolated file.")
    def get_langchain_llm(model_name):
        class MockLLM:
            def with_structured_output(self, schema):
                return self
            def invoke(self, input_variables):
                raise Exception("Mock LLM call failed.")
        return MockLLM()

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator

# Advanced caching system to reduce LLM calls
class LLMCache:
    """Smart caching system for LLM responses with TTL and content hashing"""
    def __init__(self, default_ttl: int = 7200): # 2 hours
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
                    # Use file modification time and size for path objects
                    content_hash.update(str(img.stat().st_mtime).encode())
                    content_hash.update(str(img.stat().st_size).encode())
                else:
                    # Fallback for non-path objects (e.g., base64 strings)
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
                # Expired
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

# BULLETPROOF Pydantic models with comprehensive validation
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
    action: str = Field(description="The specific action to take, e.g., 'click', 'type_text', 'scroll', 'select_option'.")
    target_element_description: str = Field(description="Detailed description of the UI element with fallback selectors (XPath, ID, Text, etc.).")
    value_to_enter: Optional[str] = Field(description="The text value to enter. For complex selections provide as JSON string. Use null if not applicable.", default=None)
    associated_image: Optional[str] = Field(default=None, description="The filename of the associated image from the context.")
    prerequisites: List[str] = Field(description="Conditions that must be met before this step.", default=[])
    expected_outcome: str = Field(description="What should happen after this step completes.")
    fallback_actions: List[str] = Field(description="Alternative actions if primary action fails.", default=[])
    timing_notes: Optional[Union[str, dict]] = Field(description="Special timing considerations for this step.")

class EnhancedBlueprintOutput(BaseModel):
    summary: BlueprintSummary
    steps: List[BlueprintStep]
    metadata: Dict[str, Any] = Field(description="Additional metadata about the blueprint generation")

    @model_validator(mode='before')
    @classmethod
    def ensure_required_fields(cls, data: Any) -> Any:
        """BULLETPROOF: Ensure all required fields are present with valid defaults"""
        if data is None:
            data = {}
        
        if not isinstance(data, dict):
            data = {}
        
        # Ensure basic structure
        if 'summary' not in data or not isinstance(data.get('summary'), dict):
            data['summary'] = {
                'overall_goal': 'Generated automation task',
                'target_application': 'Unknown application',
                'platform': 'mobile',
                'estimated_duration': 'Unknown',
                'complexity_indicators': []
            }
        
        if 'steps' not in data or not isinstance(data.get('steps'), list):
            data['steps'] = []
        
        if 'metadata' not in data or not isinstance(data.get('metadata'), dict):
            data['metadata'] = {}
        
        return data

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
    response = get_llm_response(prompt, system_message, model_name=LLMProvider.GROQ, images=images, **kwargs)

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
        if matches >= 2: # Require at least 2 keyword matches
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
            # Fallback: split text by sections (crude but necessary if JSON parsing fails)
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
        # Lazy import PyMuPDF and pytesseract to allow fallback if not installed
        try:
            import pymupdf # PyMuPDF
            has_pymupdf = True
        except ImportError:
            print("pymupdf not available. PDF text extraction will be limited.")
            has_pymupdf = False

        try:
            import pytesseract
            from PIL import Image
            has_pytesseract = True
        except ImportError:
            print("pytesseract not available for OCR fallback.")
            has_pytesseract = False

        if not has_pymupdf:
            return "Could not read PDF content (pymupdf not installed). Relying on user instructions only.", [], []

        doc = pymupdf.open(pdf_path)

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
        if has_pytesseract and len(pdf_text_content.strip()) < 100 and image_paths:
            for img_path in image_paths[:3]: # Process first 3 images
                try:
                    ocr_text = pytesseract.image_to_string(Image.open(img_path))
                    if ocr_text.strip():
                        pdf_text_content += f"\n--- OCR from {img_path.name} ---\n{ocr_text}"
                except Exception as ocr_error:
                    print(f"OCR failed for {img_path.name}: {ocr_error}")

    except Exception as e:
        print(f"PDF extraction failed: {e}")
        pdf_text_content = "Could not read PDF content. Relying on user instructions only."

    return pdf_text_content, image_paths, image_metadata

def _extract_json_from_response(raw_response: str) -> Dict[str, Any]:
    """BULLETPROOF JSON extraction from LLM response with multiple fallback strategies"""
    if not raw_response or not isinstance(raw_response, str):
        return None
    
    # Strategy 1: Look for JSON code blocks
    json_code_blocks = re.findall(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
    for block in json_code_blocks:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 2: Extract largest JSON object
    json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
    # Sort by length, try largest first
    json_matches.sort(key=len, reverse=True)
    for match in json_matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue
    
    # Strategy 3: Find balanced braces
    start_idx = raw_response.find('{')
    if start_idx != -1:
        brace_count = 0
        for i, char in enumerate(raw_response[start_idx:], start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(raw_response[start_idx:i+1])
                    except json.JSONDecodeError:
                        break
    
    # Strategy 4: Try to parse the entire response
    try:
        return json.loads(raw_response.strip())
    except json.JSONDecodeError:
        pass
    
    return None

def run_enhanced_agent1(seq_no: str, pdf_path: Path, instructions: str, platform: str) -> dict:
    """Enhanced Agent 1 with bulletproof JSON parsing and guaranteed output"""
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
            print(f"Failed to copy image {img_path.name} to final dir: {e}. Using temp path for analysis.")
            final_image_paths.append(img_path)

    print(f"[{seq_no}] 🔍 Analyzing PDF content with smart caching...")
    pdf_analysis = analyze_pdf_content_smart(pdf_text_content, len(final_image_paths))

    print(f"[{seq_no}] 🤔 Applying reflection patterns...")
    reflection = agent1_scratchpad.reflect_on_complexity(pdf_analysis, len(final_image_paths))

    print(f"[{seq_no}] 🖼️ Performing optimized image analysis...")
    enhanced_image_descriptions = {}

    # Process images in batches to avoid overwhelming context
    for img_path in final_image_paths:
        step_context = f"Step analysis for {img_path.name} in {platform} automation"
        analysis_results = enhance_image_analysis_smart(img_path, step_context)
        enhanced_image_descriptions[img_path.name] = analysis_results

    # BULLETPROOF blueprint generation with multiple strategies
    max_retries = 3
    blueprint_dict = None

    for retry_count in range(max_retries):
        try:
            print(f"[{seq_no}] 🤖 Generating blueprint (attempt {retry_count + 1}/{max_retries})...")
            
            # Build comprehensive prompt for blueprint generation
            system_prompt = f"""You are an elite automation architect specializing in {platform} automation. Generate a complete automation blueprint in valid JSON format.

REQUIRED JSON STRUCTURE:
{{
  "summary": {{
    "overall_goal": "Detailed automation goal",
    "target_application": "Application name and context",
    "platform": "{platform}",
    "estimated_duration": "Expected execution time",
    "complexity_indicators": ["list", "of", "complexity", "factors"]
  }},
  "steps": [
    {{
      "step_id": 1,
      "screen_name": "Screen name",
      "description": "Detailed step description",
      "action": "click|type_text|scroll|select_option|wait",
      "target_element_description": "Element description with selectors",
      "value_to_enter": "Text value or JSON string for complex data",
      "associated_image": "image_filename.png or null",
      "prerequisites": ["list of prerequisites"],
      "expected_outcome": "What should happen",
      "fallback_actions": ["alternative actions"],
      "timing_notes": "Special timing considerations"
    }}
  ],
  "metadata": {{}}
}}

Context Analysis:
- Platform: {platform}
- Complexity Score: {reflection.get('complexity_score', 5)}/10
- UI Patterns: {', '.join(pdf_analysis.get('ui_patterns', []))}
- Estimated Steps: {pdf_analysis.get('estimated_steps', 'Unknown')}

Generate ONLY valid JSON. Do not include explanations or markdown formatting."""

            human_prompt = f"""
User Instructions: {instructions}

PDF Content: {pdf_text_content[:2000]}...

Image Analysis: {json.dumps(enhanced_image_descriptions, indent=2)[:1000]}...

Generate a complete automation blueprint in valid JSON format following the required structure exactly.
"""

            # Direct LLM call with comprehensive error handling
            raw_response = get_llm_response(
                human_prompt,
                system_prompt,
                model_name=LLMProvider.GROQ,
                images=None
            )

            print(f"[{seq_no}] 🧠 Parsing LLM response...")
            
            # BULLETPROOF JSON extraction
            parsed_data = _extract_json_from_response(raw_response)
            
            if parsed_data:
                # Validate and create blueprint object
                blueprint_obj = EnhancedBlueprintOutput(**parsed_data)
                blueprint_dict = blueprint_obj.model_dump()
                print(f"[{seq_no}] ✅ Successfully generated blueprint with {len(blueprint_dict.get('steps', []))} steps")
                break
            else:
                print(f"[{seq_no}] ⚠️ Could not extract valid JSON from response")
                if retry_count < max_retries - 1:
                    continue

        except Exception as e:
            print(f"[{seq_no}] ⚠️ Attempt {retry_count + 1} failed: {e}")
            if retry_count >= max_retries - 1:
                print(f"[{seq_no}] 🛡️ Using guaranteed fallback blueprint...")
                break

    # GUARANTEED fallback blueprint
    if blueprint_dict is None:
        blueprint_dict = {
            "summary": {
                "overall_goal": f"Automation task for {platform} platform based on provided instructions",
                "target_application": "Application extracted from PDF content",
                "platform": platform,
                "estimated_duration": f"{pdf_analysis.get('estimated_steps', 5)} steps estimated",
                "complexity_indicators": pdf_analysis.get("complexity_factors", ["medium"])
            },
            "steps": [
                {
                    "step_id": 1,
                    "screen_name": "Application Interface",
                    "description": "Navigate to the target application",
                    "action": "navigate",
                    "target_element_description": "Application main interface",
                    "value_to_enter": None,
                    "associated_image": final_image_paths[0].name if final_image_paths else None,
                    "prerequisites": [],
                    "expected_outcome": "Application loads successfully",
                    "fallback_actions": ["retry navigation", "check connectivity"],
                    "timing_notes": "Allow standard loading time"
                },
                {
                    "step_id": 2,
                    "screen_name": "Main Screen",
                    "description": "Perform primary automation task based on instructions",
                    "action": "click",
                    "target_element_description": "Primary action element identified from context",
                    "value_to_enter": None,
                    "associated_image": final_image_paths[1].name if len(final_image_paths) > 1 else None,
                    "prerequisites": ["Application loaded successfully"],
                    "expected_outcome": "Task completes successfully",
                    "fallback_actions": ["retry action", "alternative element selection"],
                    "timing_notes": "Standard interaction timing"
                }
            ],
            "metadata": {
                "fallback_generated": True,
                "fallback_reason": "LLM response parsing failed after retries"
            }
        }

    # Add comprehensive metadata
    blueprint_dict.setdefault("metadata", {})
    blueprint_dict["metadata"].update({
        "generation_timestamp": time.time(),
        "pdf_analysis": pdf_analysis,
        "reflection_analysis": reflection,
        "images_processed": len(final_image_paths),
        "enhanced_analysis": True,
        "agent_version": "enhanced_1.4_bulletproof_json_parsing",
        "cache_stats": agent1_cache.get_stats(),
        "scratchpad_entries": len(agent1_scratchpad.reflection_log)
    })

    # CRITICAL POST-PROCESSING: Ensure string types are correct
    for step in blueprint_dict.get("steps", []):
        # Ensure target_element_description is a string
        if step.get("target_element_description") is None:
            step["target_element_description"] = ""
        elif not isinstance(step["target_element_description"], str):
            step["target_element_description"] = str(step["target_element_description"])

        # Ensure value_to_enter is a string or None
        if isinstance(step.get("value_to_enter"), dict):
            step["value_to_enter"] = json.dumps(step["value_to_enter"])
        elif step.get("value_to_enter") is not None and not isinstance(step.get("value_to_enter"), str):
            step["value_to_enter"] = str(step["value_to_enter"])

    # Save all files
    blueprint_path = out_dir / "blueprint.json"
    blueprint_path.write_text(json.dumps(blueprint_dict, indent=2), encoding="utf-8")

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

    return blueprint_dict

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
                elif step.get(key) is None:
                    issues.append(f"Step {i} key '{key}' has a null value")

    return issues

__all__ = ["run_enhanced_agent1", "run_agent1", "analyze_pdf_content_smart", "enhance_image_analysis_smart", "agent1_cache", "agent1_scratchpad"]
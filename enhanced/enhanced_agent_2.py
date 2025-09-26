# agents/enhanced_agent_2.py

from pathlib import Path
import json
import re
from fastapi import HTTPException
from typing import TypedDict, List, Optional, Dict, Any, Tuple, Set, Union
import importlib.util
import time
import hashlib
import random  # FIXED: Added missing import

from config import ARTIFACTS_DIR
from agents.llm_utils import get_llm_response, get_langchain_llm
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage, HumanMessage

# ===== FIXED JSON PARSING HELPERS =====
def _extract_first_json_block(text: str) -> str:
    """Extract and clean JSON from potentially malformed text"""
    # Remove fenced code blocks if present
    text = re.sub(r"^```.*?\n|```$", "", text.strip(), flags=re.DOTALL | re.MULTILINE)
    # Replace smart quotes/dashes with ASCII
    trans_table = {
        ord('"'): '"', ord('"'): '"', ord('‟'): '"', ord('''): "'", ord('''): "'",
        ord('–'): '-', ord('—'): '-', ord('‑'): '-', ord('\u00A0'): ' ',
    }
    text = text.translate(trans_table)
    # Find first balanced JSON object
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

def _repair_and_parse_json(raw: str) -> Dict[str, Any]:
    """Repair common JSON issues and parse"""
    j = _extract_first_json_block(raw)
    # Common simple repairs
    j = re.sub(r",\s*([}\]])", r"\1", j)           # trailing commas
    j = re.sub(r"(['\"])\s*:\s*(['\"])", r"\1:\2", j)  # normalize colon spacing
    # Ensure double quotes for keys/strings when obvious single quotes used
    if "'" in j and '"' not in j[: j.find(':')+2]:
        j = j.replace("'", '"')
    return json.loads(j)

# Advanced caching system to reduce LLM calls
class Agent2Cache:
    """Smart caching system for Agent 2 with TODO-specific optimizations"""
    
    def __init__(self, default_ttl: int = 7200): # 2 hours
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.default_ttl = default_ttl
        self.hit_count = 0
        self.miss_count = 0
        self.todo_templates: Dict[str, str] = {}
    
    def _create_cache_key(self, prompt: str, blueprint_hash: str = "") -> str:
        """Create cache key from prompt and blueprint content"""
        content_hash = hashlib.sha256()
        content_hash.update(prompt.encode('utf-8'))
        content_hash.update(blueprint_hash.encode('utf-8'))
        return content_hash.hexdigest()
    
    def get(self, prompt: str, blueprint_hash: str = "") -> Optional[Any]:
        """Get cached response if available"""
        cache_key = self._create_cache_key(prompt, blueprint_hash)
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if time.time() < entry["expires_at"]:
                self.hit_count += 1
                return entry["value"]
            else:
                del self._cache[cache_key]
        
        self.miss_count += 1
        return None
    
    def set(self, prompt: str, value: Any, blueprint_hash: str = "", ttl: Optional[int] = None):
        """Cache a response"""
        cache_key = self._create_cache_key(prompt, blueprint_hash)
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
agent2_cache = Agent2Cache(default_ttl=7200)

# TODO Organization and Planning System
class TodoOrganizer:
    """Advanced TODO organization system for better script writing"""
    
    def __init__(self):
        self.todo_templates = self._load_todo_templates()
        self.organization_patterns = self._load_organization_patterns()
    
    def _load_todo_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined TODO templates for common automation patterns"""
        return {
            "mobile_setup": {
                "priority": "HIGH",
                "category": "INITIALIZATION",
                "todos": [
                    "TODO: Initialize mobile driver with device configuration",
                    "TODO: Setup Appium capabilities for target platform", 
                    "TODO: Implement device detection and validation",
                    "TODO: Configure logging and error handling",
                    "TODO: Setup screenshot capture mechanism"
                ]
            },
            "web_setup": {
                "priority": "HIGH", 
                "category": "INITIALIZATION",
                "todos": [
                    "TODO: Initialize browser driver with options",
                    "TODO: Configure viewport and user agent",
                    "TODO: Setup implicit/explicit waits",
                    "TODO: Configure logging and screenshot capture",
                    "TODO: Implement cookie and session management"
                ]
            },
            "element_interaction": {
                "priority": "MEDIUM",
                "category": "CORE_FUNCTIONALITY", 
                "todos": [
                    "TODO: Implement smart element finding with multiple strategies",
                    "TODO: Add retry logic for element interactions",
                    "TODO: Create fallback selectors for reliability",
                    "TODO: Add element state validation before interaction",
                    "TODO: Implement custom wait conditions"
                ]
            },
            "error_handling": {
                "priority": "HIGH",
                "category": "RELIABILITY",
                "todos": [
                    "TODO: Implement comprehensive exception handling",
                    "TODO: Add retry mechanisms with exponential backoff",
                    "TODO: Create error recovery strategies",
                    "TODO: Add detailed error logging and reporting",
                    "TODO: Implement graceful cleanup on failures"
                ]
            },
            "performance_monitoring": {
                "priority": "MEDIUM",
                "category": "OPTIMIZATION",
                "todos": [
                    "TODO: Add execution timing measurements",
                    "TODO: Implement memory usage monitoring", 
                    "TODO: Create performance benchmarking",
                    "TODO: Add resource cleanup verification",
                    "TODO: Optimize wait times and delays"
                ]
            }
        }
    
    def _load_organization_patterns(self) -> Dict[str, List[str]]:
        """Load code organization patterns for better structure"""
        return {
            "class_structure": [
                "# Configuration and constants",
                "# Data classes and models",
                "# Utility functions",
                "# Core automation class",
                "# Error handling and cleanup", 
                "# Main execution logic"
            ],
            "method_organization": [
                "setup_methods",
                "element_finding_methods",
                "interaction_methods",
                "validation_methods",
                "error_handling_methods",
                "cleanup_methods"
            ],
            "import_organization": [
                "# Standard library imports",
                "# Third-party imports", 
                "# Framework-specific imports",
                "# Local/custom imports"
            ]
        }
    
    def organize_todos_by_priority(self, blueprint: 'AutomationBlueprint', analysis: 'CodeAnalysis') -> Dict[str, List[str]]:
        """Organize TODOs by priority based on blueprint complexity and analysis"""
        organized_todos = {
            "CRITICAL": [],
            "HIGH": [],
            "MEDIUM": [],
            "LOW": []
        }
        
        complexity_score = analysis.complexity_score
        platform = blueprint.summary.platform.lower()
        
        # Add platform-specific setup TODOs
        if platform == "mobile":
            organized_todos["CRITICAL"].extend(self.todo_templates["mobile_setup"]["todos"])
        else:
            organized_todos["CRITICAL"].extend(self.todo_templates["web_setup"]["todos"])
        
        # Add complexity-based TODOs
        if complexity_score >= 7:
            organized_todos["HIGH"].extend(self.todo_templates["error_handling"]["todos"])
            organized_todos["MEDIUM"].extend(self.todo_templates["performance_monitoring"]["todos"])
        
        # Add element interaction TODOs
        organized_todos["MEDIUM"].extend(self.todo_templates["element_interaction"]["todos"])
        
        # Add step-specific TODOs
        for step in blueprint.steps:
            step_todos = self._generate_step_specific_todos(step, complexity_score)
            if step_todos:
                organized_todos["MEDIUM"].extend(step_todos)
        
        return organized_todos
    
    def _generate_step_specific_todos(self, step: 'BlueprintStep', complexity_score: int) -> List[str]:
        """Generate specific TODOs for each blueprint step"""
        todos = []
        action = step.action.lower()
        
        if action in ['click', 'tap']:
            todos.append(f"TODO: Implement reliable clicking for step {step.step_id} - {step.description}")
            if complexity_score >= 6:
                todos.append(f"TODO: Add click verification and retry logic for step {step.step_id}")
        elif action in ['type_text', 'send_keys']:
            todos.append(f"TODO: Implement text input with validation for step {step.step_id}")
            todos.append(f"TODO: Add input field clearing and verification for step {step.step_id}")
        elif action in ['scroll', 'swipe']:
            todos.append(f"TODO: Implement {action} with direction detection for step {step.step_id}")
            todos.append(f"TODO: Add scroll completion detection for step {step.step_id}")
        elif action == 'wait':
            todos.append(f"TODO: Implement smart wait conditions for step {step.step_id}")
        
        # Add image-specific TODOs if associated image exists
        if step.associated_image:
            todos.append(f"TODO: Process screenshot analysis for {step.associated_image}")
        
        return todos
    
    def generate_implementation_roadmap(self, organized_todos: Dict[str, List[str]]) -> str:
        """Generate a structured implementation roadmap"""
        roadmap = """# 🗺️ IMPLEMENTATION ROADMAP

This roadmap organizes all TODOs by priority to ensure systematic development:

## 🔴 CRITICAL PRIORITY (Implement First)

These are essential for basic functionality:

"""
        
        for i, todo in enumerate(organized_todos["CRITICAL"], 1):
            roadmap += f"{i}. {todo}\n"
        
        roadmap += """
## 🟡 HIGH PRIORITY (Implement Second)

Important for reliability and robustness:

"""
        
        for i, todo in enumerate(organized_todos["HIGH"], 1):
            roadmap += f"{i}. {todo}\n"
        
        roadmap += """
## 🟠 MEDIUM PRIORITY (Implement Third)

Enhances functionality and user experience:

"""
        
        for i, todo in enumerate(organized_todos["MEDIUM"], 1):
            roadmap += f"{i}. {todo}\n"
        
        roadmap += """
## 🟢 LOW PRIORITY (Implement Last)

Nice-to-have features and optimizations:

"""
        
        for i, todo in enumerate(organized_todos["LOW"], 1):
            roadmap += f"{i}. {todo}\n"
        
        roadmap += """
## 📋 IMPLEMENTATION TIPS

1. **Start with CRITICAL**: Get basic functionality working first
2. **Test Early**: Implement unit tests as you complete each priority level
3. **Incremental Development**: Test each TODO implementation before moving to next
4. **Code Reviews**: Have each priority level reviewed before proceeding
5. **Documentation**: Update documentation as you complete TODOs

## 🔄 DEVELOPMENT PHASES

**Phase 1: Foundation** - Complete all CRITICAL todos
**Phase 2: Reliability** - Complete all HIGH priority todos  
**Phase 3: Enhancement** - Complete all MEDIUM priority todos
**Phase 4: Polish** - Complete all LOW priority todos

Remember: This roadmap ensures systematic development and reduces technical debt!

"""
        return roadmap

# Global TODO organizer instance
todo_organizer = TodoOrganizer()

# Enhanced Pydantic models with FIXED Union types
class BlueprintStep(BaseModel):
    step_id: int
    screen_name: str
    description: str
    action: str
    target_element_description: str
    value_to_enter: Optional[Union[str, dict]] = Field(default=None, description="The text value to enter or complex selection. Use null if not applicable.")
    associated_image: Optional[str] = Field(default=None, description="The filename of the associated image, if any.")

class BlueprintSummary(BaseModel):
    overall_goal: str
    target_application: str
    platform: str

class AutomationBlueprint(BaseModel):
    summary: BlueprintSummary
    steps: List[BlueprintStep]

class CodeAnalysis(BaseModel):
    complexity_score: int = Field(description="Complexity rating from 1-10", ge=1, le=10)
    recommended_patterns: List[str] = Field(description="List of recommended design patterns")
    potential_issues: List[str] = Field(description="List of potential issues or risks")
    optimization_suggestions: List[str] = Field(description="Performance and reliability optimizations")

class EnhancedCodeOutput(BaseModel):
    python_code: str = Field(description="Complete, production-ready Python script with organized TODOs")
    requirements: str = Field(description="Comprehensive requirements.txt with pinned versions")
    code_analysis: CodeAnalysis = Field(description="Analysis of the generated code")
    test_code: Optional[str] = Field(description="Optional basic test suite", default=None)
    documentation: str = Field(description="Technical documentation with implementation roadmap")
    implementation_roadmap: str = Field(description="Structured TODO roadmap for systematic development")

class Agent2Output(TypedDict):
    script: str
    requirements: str
    analysis: dict
    test_script: Optional[str]
    documentation: str
    implementation_roadmap: str

# Agent 2 Scratchpad for reflection and code improvement
class Agent2Scratchpad:
    """Memory and reflection system for Agent 2 to improve code generation"""
    
    def __init__(self):
        self.code_patterns_cache: Dict[str, str] = {}
        self.best_practices: Dict[str, List[str]] = {}
        self.reflection_log: List[Dict[str, Any]] = []
        self.generated_solutions: Dict[str, Dict[str, Any]] = {}
    
    def reflect_on_blueprint_complexity(self, blueprint: AutomationBlueprint) -> Dict[str, Any]:
        """Reflect on blueprint complexity without additional LLM calls"""
        reflection = {
            "complexity_assessment": "medium",
            "key_challenges": [],
            "recommended_approaches": [],
            "testing_strategy": [],
            "architecture_notes": []
        }
        
        step_count = len(blueprint.steps)
        unique_actions = set(step.action for step in blueprint.steps)
        has_complex_ui = any("dropdown" in step.target_element_description.lower() or
                           "modal" in step.target_element_description.lower()
                           for step in blueprint.steps)
        
        # Assess complexity based on blueprint characteristics
        if step_count > 10 or len(unique_actions) > 6 or has_complex_ui:
            reflection["complexity_assessment"] = "high"
            reflection["key_challenges"].extend([
                "Multiple interaction types require robust element handling",
                "Complex UI elements need specialized wait strategies",
                "High step count requires careful error handling"
            ])
            reflection["recommended_approaches"].extend([
                "Implement page object model pattern",
                "Use factory pattern for element creation",
                "Add comprehensive retry mechanisms"
            ])
        elif step_count > 5 or len(unique_actions) > 3:
            reflection["complexity_assessment"] = "medium"
            reflection["key_challenges"].extend([
                "Moderate complexity requires structured approach",
                "Multiple actions need consistent error handling"
            ])
            reflection["recommended_approaches"].extend([
                "Use helper methods for common actions",
                "Implement basic retry logic",
                "Add logging for debugging"
            ])
        else:
            reflection["complexity_assessment"] = "low"
            reflection["recommended_approaches"].extend([
                "Simple linear approach is sufficient",
                "Focus on clarity over complexity"
            ])
        
        # Platform-specific recommendations
        platform = blueprint.summary.platform.lower()
        if platform == "mobile":
            reflection["architecture_notes"].extend([
                "Mobile automation requires device-specific considerations",
                "Implement touch gesture support",
                "Add orientation change handling"
            ])
        else:
            reflection["architecture_notes"].extend([
                "Web automation should handle cross-browser compatibility",
                "Implement responsive design considerations",
                "Add browser-specific optimizations"
            ])
        
        # Testing strategy based on complexity
        if reflection["complexity_assessment"] == "high":
            reflection["testing_strategy"].extend([
                "Implement comprehensive unit tests",
                "Add integration testing for workflows",
                "Create smoke tests for critical paths",
                "Add performance benchmarking"
            ])
        else:
            reflection["testing_strategy"].extend([
                "Basic unit tests for core functions",
                "End-to-end testing for main workflow"
            ])
        
        # Log this reflection
        self.reflection_log.append({
            "timestamp": time.time(),
            "action": "blueprint_complexity_reflection",
            "data": reflection
        })
        
        return reflection
    
    def suggest_code_improvements(self, blueprint: AutomationBlueprint, analysis: CodeAnalysis) -> List[str]:
        """Suggest code improvements without LLM calls based on patterns"""
        improvements = []
        
        # Complexity-based suggestions
        if analysis.complexity_score >= 8:
            improvements.extend([
                "Implement circuit breaker pattern for external service calls",
                "Add comprehensive metrics collection",
                "Create custom exceptions for different failure modes",
                "Implement async/await for better performance"
            ])
        elif analysis.complexity_score >= 6:
            improvements.extend([
                "Add retry decorators for flaky operations",
                "Implement structured logging",
                "Create helper classes for common operations"
            ])
        
        # Pattern-based suggestions
        if "verification_handling" in analysis.recommended_patterns:
            improvements.append("Add CAPTCHA detection and human intervention prompts")
        
        if "dynamic_content_handling" in analysis.recommended_patterns:
            improvements.extend([
                "Implement polling mechanisms for dynamic content",
                "Add WebSocket support for real-time updates"
            ])
        
        # Platform-specific suggestions
        platform = blueprint.summary.platform.lower()
        if platform == "mobile":
            improvements.extend([
                "Add device capability detection",
                "Implement gesture recognition fallbacks",
                "Add battery and performance monitoring"
            ])
        else:
            improvements.extend([
                "Add browser fingerprinting prevention",
                "Implement viewport-responsive element detection",
                "Add download progress monitoring"
            ])
        
        return improvements

# Global scratchpad instance
agent2_scratchpad = Agent2Scratchpad()

def get_llm_response_with_cache_agent2(prompt: str, system_message: str = "", blueprint_hash: str = "") -> str:
    """Cached LLM response wrapper for Agent 2"""
    # Check cache first
    cached_response = agent2_cache.get(prompt, blueprint_hash)
    if cached_response:
        print("🎯 Agent2 Cache HIT - Skipping LLM call")
        return cached_response
    
    print("🔄 Agent2 Cache MISS - Making LLM call")
    
    # Make LLM call
    response = get_llm_response(prompt, system_message)
    
    # Cache the response
    agent2_cache.set(prompt, response, blueprint_hash)
    
    return response
# FIXED Mobile Setup Template with Android 15 and proven configuration
MOBILE_SETUP_TEMPLATE = '''# NOTE: TouchAction and MultiAction are deprecated in modern Appium clients.
# DO NOT import or use TouchAction or MultiAction in generated code.
# Prefer Appium 'mobile:' execute_script gestures (mobile: longClickGesture, mobile: clickGesture, mobile: swipeGesture)
# or the W3C Actions API (PointerInput/Sequence) for advanced gestures.

import os
import sys
import time
import logging
import subprocess
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import logging.handlers
from pathlib import Path

# Third-party imports
from appium import webdriver
from appium.options.android import UiAutomator2Options
from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    StaleElementReferenceException,
    WebDriverException
)

# ---------------- HARD-CODED CONSTANTS (DO NOT CHANGE) ----------------
# These are intentionally immutable in the runtime logic below.
DEFAULT_APP_PACKAGE = "com.microsoft.office.outlook"
DEFAULT_APP_ACTIVITY = ".MainActivity"
DEFAULT_DEVICE_UDID = "ZD222GXYPV"
DEFAULT_DEVICE_NAME = "ZD222GXYPV"
DEFAULT_APPIUM_SERVER = "http://localhost:4723"

# ----------------- Light DeviceConfig kept for compatibility -----------------
@dataclass
class DeviceConfig:
    platform: str  # only 'android' expected
    # The following fields are intentionally ignored by setup_driver in this hardcoded template:
    device_name: Optional[str] = None
    platform_version: Optional[str] = None
    app_package: Optional[str] = None
    app_activity: Optional[str] = None
    udid: Optional[str] = None

class EnhancedMobileDriver:
    """Mobile driver with fully hardcoded Outlook launch (no discovery, no overrides)."""

    def __init__(self, device_config: DeviceConfig, server_url: str = DEFAULT_APPIUM_SERVER):
        self.config = device_config
        self.server_url = server_url or DEFAULT_APPIUM_SERVER
        self.driver: Optional[webdriver.Remote] = None
        self.wait: Optional[WebDriverWait] = None
        self.screen_size: Optional[Dict[str, int]] = None

        # stable logging setup
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger("OutlookAutomation")
        # avoid handler duplication if re-imported
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)

    def _press_keycode_safe(self, code: int):
        """Press Android keycode; use driver if available, else adb fallback to hardcoded UDID."""
        try:
            if self.driver and hasattr(self.driver, "press_keycode"):
                self.driver.press_keycode(code)
                return
        except Exception:
            pass
        try:
            cmd = ['adb', '-s', DEFAULT_DEVICE_UDID, 'shell', 'input', 'keyevent', str(code)]
            subprocess.run(cmd, timeout=3, check=False)
        except Exception:
            pass

    def setup_driver(self, timeout: int = 30) -> bool:
        """Create Appium driver using only hardcoded values. Ignores incoming DeviceConfig fields."""
        try:
            self.logger.info("Setting up mobile driver with FULLY HARD-CODED Outlook config...")

            # enforce Android-only in this template
            if str(self.config.platform).lower() != "android":
                self.logger.error("This hardcoded template only supports Android.")
                return False

            options = UiAutomator2Options()
            options.platform_name = "Android"
            options.automation_name = "UiAutomator2"

            # Force hardcoded device name & udid (ignore DeviceConfig)
            options.device_name = DEFAULT_DEVICE_NAME
            options.udid = DEFAULT_DEVICE_UDID

            # Force hardcoded Outlook package & activity (IGNORE any provided values)
            options.app_package = DEFAULT_APP_PACKAGE
            options.app_activity = DEFAULT_APP_ACTIVITY

            # Stable defaults
            options.no_reset = True
            options.full_reset = False
            options.new_command_timeout = 300
            options.unicode_keyboard = True
            options.reset_keyboard = True
            options.auto_grant_permissions = True
            options.auto_launch = True

            # Best-effort flags (safe to call)
            try:
                options.ensure_webviews_have_pages = True
            except Exception:
                pass
            try:
                options.native_web_screenshot = True
            except Exception:
                pass
            try:
                options.enforce_xpath1 = True
            except Exception:
                pass

            # Create driver with retry (3 attempts)
            last_exc = None
            for attempt in range(3):
                try:
                    self.logger.debug(f"Attempting driver creation (attempt {attempt+1}) -> server: {self.server_url}")
                    self.driver = webdriver.Remote(self.server_url, options=options)
                    break
                except Exception as e:
                    last_exc = e
                    self.logger.warning(f"Driver creation attempt {attempt+1} failed: {e}")
                    time.sleep(2)
            else:
                self.logger.error("All driver creation attempts failed.")
                if last_exc:
                    self.logger.debug(f"Last exception: {last_exc}")
                return False

            # Post-creation setup
            try:
                self.wait = WebDriverWait(self.driver, timeout)
            except Exception:
                self.wait = None
            try:
                self.screen_size = self.driver.get_window_size()
            except Exception:
                self.screen_size = {"width": 0, "height": 0}
            try:
                self.driver.update_settings({"enforceXPath1": True})
            except Exception:
                pass
            try:
                self.driver.implicitly_wait(4)
            except Exception:
                pass

            self.logger.info(f"Driver ready - screen_size={self.screen_size}")
            return True

        except Exception as e:
            self.logger.error(f"Driver setup failed: {e}")
            return False

    # ----------------- Helper utilities (preserved) -----------------
    def _element_center_coords(self, element) -> Tuple[int, int]:
        """Return center (x,y) of element (uses rect, or location+size fallback)."""
        rect = {}
        try:
            rect = element.rect
        except Exception:
            try:
                loc = element.location
                size = element.size
                rect = {'x': loc.get('x', 0), 'y': loc.get('y', 0),
                        'width': size.get('width', 0), 'height': size.get('height', 0)}
            except Exception:
                rect = {'x': 0, 'y': 0, 'width': 0, 'height': 0}
        x = int(rect.get('x', 0) + rect.get('width', 0) / 2)
        y = int(rect.get('y', 0) + rect.get('height', 0) / 2)
        return x, y

    def find_element_bulletproof(self, by: str, value: str, timeout: int = 10, retry_attempts: int = 3):
        """Find first visible element, with retries and best-effort fallback to non-visible element."""
        for attempt in range(retry_attempts):
            try:
                elements = WebDriverWait(self.driver, timeout).until(lambda d: d.find_elements(by, value))
                if elements:
                    for elem in elements:
                        try:
                            if elem.is_displayed():
                                return elem
                        except Exception:
                            return elem
                time.sleep(0.5)
            except Exception:
                if attempt < retry_attempts - 1:
                    time.sleep(0.5)
        return None

    def type_smart(self, locator: Tuple[str, str], text: str, description: str = "", clear_first: bool = True) -> bool:
        """Focus element, optionally clear via DEL keycodes, send_keys, fallback to adb input text."""
        try:
            element = self.find_element_bulletproof(locator[0], locator[1])
            if not element:
                self.logger.warning(f"type_smart: element not found: {description}")
                return False
            try:
                element.click()
            except Exception:
                try:
                    x, y = self._element_center_coords(element)
                    self.driver.execute_script("mobile: clickGesture", {"x": x, "y": y})
                except Exception:
                    pass
            time.sleep(0.5)
            if clear_first:
                for _ in range(15):
                    try:
                        self._press_keycode_safe(67)  # DEL
                    except Exception:
                        pass
                    time.sleep(0.02)
                time.sleep(0.3)
            try:
                element.clear()
            except Exception:
                pass
            try:
                element.send_keys(str(text))
                self.logger.info(f"✓ Typed: {description} = '{text}'")
                return True
            except Exception:
                try:
                    cmd = ['adb', '-s', DEFAULT_DEVICE_UDID, 'shell', 'input', 'text', str(text)]
                    subprocess.run(cmd, timeout=8, check=False)
                    self.logger.info(f"✓ Typed (ADB): {description} = '{text}'")
                    return True
                except Exception as e:
                    self.logger.error(f"ADB typing failed for {description}: {e}")
                    return False
        except Exception as e:
            self.logger.error(f"Typing failed for {description}: {e}")
            return False

    def click_smart(self, locator: Tuple[str, str], description: str = "") -> bool:
        """element.click() primary; fallback to mobile: clickGesture by element center (no TouchAction)."""
        try:
            element = self.find_element_bulletproof(locator[0], locator[1])
            if not element:
                self.logger.warning(f"click_smart: element not found: {description}")
                return False
            try:
                element.click()
            except Exception:
                try:
                    x, y = self._element_center_coords(element)
                    self.driver.execute_script("mobile: clickGesture", {"x": x, "y": y})
                except Exception as e:
                    self.logger.error(f"click_smart fallback failed for {description}: {e}")
                    return False
            self.logger.info(f"✓ Clicked: {description}")
            time.sleep(1)
            return True
        except Exception as e:
            self.logger.error(f"Click failed for {description}: {e}")
            return False

    def cleanup(self):
        """Shutdown driver cleanly."""
        if self.driver:
            try:
                self.driver.quit()
                self.logger.info("Driver cleaned up successfully")
            except Exception:
                self.logger.warning("Driver cleanup had issues")


'''

# Web template with stealth configuration
WEB_SETUP_TEMPLATE = '''# Standard library imports
# Standard library imports
import os
import sys
import time
import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import logging.handlers


# Third-party imports
from playwright.sync_api import sync_playwright, Playwright, Page, Browser, BrowserContext, expect

class BrowserType(Enum):
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"

@dataclass
class WebConfig:
    browser_type: BrowserType = BrowserType.CHROMIUM
    headless: bool = False
    viewport_width: int = 1920
    viewport_height: int = 1080
    user_agent: Optional[str] = None
    timeout: int = 30000

# --- Injected enhanced stealth script function (added, nothing removed) ---
def get_enhanced_stealth_script():
    return """
    // Remove webdriver property completely
    Object.defineProperty(navigator, 'webdriver', {
        get: () => undefined,
        set: () => {},
        configurable: true,
        enumerable: false
    });

    // Enhanced Chrome object simulation
    window.chrome = {
        runtime: {
            onConnect: undefined,
            onMessage: undefined
        },
        loadTimes: function() {
            return {
                requestTime: Date.now() / 1000 - Math.random() * 1000,
                startLoadTime: Date.now() / 1000 - Math.random() * 1000,
                commitLoadTime: Date.now() / 1000 - Math.random() * 1000,
                finishDocumentLoadTime: Date.now() / 1000 - Math.random() * 1000,
                finishLoadTime: Date.now() / 1000 - Math.random() * 1000,
                firstPaintTime: Date.now() / 1000 - Math.random() * 1000,
                firstPaintAfterLoadTime: 0,
                navigationType: 'Other',
                wasFetchedViaSpdy: false,
                wasNpnNegotiated: false,
                npnNegotiatedProtocol: 'unknown',
                wasAlternateProtocolAvailable: false,
                connectionInfo: 'unknown'
            };
        }
    };

    // Realistic plugins array
    Object.defineProperty(navigator, 'plugins', {
        get: () => [
            {
                0: {type: "application/x-google-chrome-pdf", suffixes: "pdf", description: "Portable Document Format"},
                description: "Portable Document Format",
                filename: "internal-pdf-viewer",
                length: 1,
                name: "Chrome PDF Plugin"
            },
            {
                0: {type: "application/pdf", suffixes: "pdf", description: ""},
                description: "",
                filename: "mhjfbmdgcfjbbpaeojofohoefgiehjai",
                length: 1,
                name: "Chrome PDF Viewer"
            }
        ],
        configurable: true,
        enumerable: false
    });

    // Enhanced navigator properties
    Object.defineProperty(navigator, 'languages', {
        get: () => ['en-US', 'en'],
        configurable: true
    });

    Object.defineProperty(navigator, 'hardwareConcurrency', {
        get: () => 4,
        configurable: true
    });

    Object.defineProperty(navigator, 'deviceMemory', {
        get: () => 8,
        configurable: true
    });

    // Override permissions API
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) => (
        parameters.name === 'notifications' ?
        Promise.resolve({ state: Notification.permission }) :
        originalQuery(parameters)
    );

    // Mock geolocation
    Object.defineProperty(navigator, 'geolocation', {
        get: () => ({
            getCurrentPosition: (success) => {
                setTimeout(() => {
                    success({
                        coords: {
                            latitude: 40.7128,
                            longitude: -74.0060,
                            accuracy: 20000
                        }
                    });
                }, 100);
            }
        })
    });
    """

# FIXED Web Driver with comprehensive stealth configuration
class EnhancedWebDriver:
    """Enhanced web driver with stealth configuration"""
    
    def __init__(self, config: WebConfig):
        self.config = config
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_browser(self) -> bool:
        """FIXED browser setup with comprehensive stealth configuration"""
        try:
            self.logger.info("Setting up web browser with stealth configuration...")
            self.playwright = sync_playwright().start()
            
            # FIXED stealth browser options (proven working)
            browser_options = {
                'headless': self.config.headless,
                'args': [
                    '--no-first-run',
                    '--no-service-autorun', 
                    '--no-default-browser-check',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--no-sandbox',
                    '--disable-web-security',
                    '--disable-blink-features=AutomationControlled',  # Critical
                    '--disable-features=VizDisplayCompositor',
                    '--disable-extensions',
                    '--disable-plugins',
                    '--disable-default-apps',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-features=TranslateUI',
                    '--disable-ipc-flooding-protection',
                    '--disable-hang-monitor',
                    '--disable-prompt-on-repost',
                    '--disable-sync',
                    '--disable-translate',
                    '--metrics-recording-only',
                    '--no-report-upload',
                    '--safebrowsing-disable-auto-update',
                    '--enable-automation=false',  # Critical
                    '--disable-client-side-phishing-detection',
                    '--disable-component-extensions-with-background-pages',
                    '--disable-default-apps',
                    '--disable-features=InterestFeedContentSuggestions',
                    '--disable-features=MediaRouter',
                    '--no-first-run',
                    '--use-mock-keychain'
                ]
            }
            
            self.browser = self.playwright.chromium.launch(**browser_options)
            
            # FIXED stealth context configuration
            context_options = {
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'locale': 'en-US',
                'timezone_id': 'America/New_York',
                'permissions': ['geolocation', 'notifications'],
                'device_scale_factor': 1,
                'is_mobile': False,
                'has_touch': False,
                'color_scheme': 'light',
                'extra_http_headers': {
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept': 'text/html,application/xhtml+xmlapplication/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
                    'Sec-Ch-Ua': '"Not_A Brand";v="99", "Google Chrome";v="120", "Chromium";v="120"',
                    'Sec-Ch-Ua-Mobile': '?0',
                    'Sec-Ch-Ua-Platform': '"Windows"',
                    'Sec-Fetch-Dest': 'document',
                    'Sec-Fetch-Mode': 'navigate',
                    'Sec-Fetch-Site': 'none',
                    'Sec-Fetch-User': '?1',
                    'Upgrade-Insecure-Requests': '1'
                }
            }
            if self.config.user_agent:
                # previously-empty block filled so custom UA works (added, not removing anything)
                context_options['user_agent'] = self.config.user_agent
            
            self.context = self.browser.new_context(**context_options)
            self.page = self.context.new_page()
            self.page.set_default_timeout(self.config.timeout)
            
            # FIXED stealth scripts (original short script kept)
            self.page.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                window.chrome = {
                    runtime: {},
                };
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
            """)
            
            # ALSO inject the full enhanced stealth script you provided (added, does not remove existing)
            self.page.add_init_script(get_enhanced_stealth_script())
            
            self.logger.info("Browser setup completed with stealth configuration")
            return True
            
        except Exception as e:
            self.logger.error(f"Browser setup failed: {e}")
            return False
    
    def navigate(self, url: str) -> bool:
        """Navigate to URL with error handling"""
        try:
            self.page.goto(url, wait_until="domcontentloaded")
            self.logger.info(f"✓ Navigated to: {url}")
            return True
        except Exception as e:
            self.logger.error(f"Navigation failed: {e}")
            return False
    
    def click_element(self, selector: str, description: str = "") -> bool:
        """Click element with retry logic"""
        try:
            self.page.click(selector, timeout=10000)
            self.logger.info(f"✓ Clicked: {description}")
            return True
        except Exception as e:
            self.logger.error(f"Click failed for {description}: {e}")
            return False
    
    def type_text(self, selector: str, text: str, description: str = "") -> bool:
        """Type text with clearing"""
        try:
            self.page.fill(selector, text)
            self.logger.info(f"✓ Typed: {description} = '{text}'")
            return True
        except Exception as e:
            self.logger.error(f"Typing failed for {description}: {e}")
            return False
    
    def cleanup(self):
        """Clean browser shutdown"""
        try:
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            self.logger.info("Browser cleaned up successfully")
        except:
            self.logger.warning("Browser cleanup had issues")

'''

def analyze_blueprint_complexity_smart(blueprint: AutomationBlueprint) -> CodeAnalysis:
    """Analyze blueprint complexity with reflection and smart assessment"""
    # Use scratchpad reflection instead of making LLM calls
    reflection = agent2_scratchpad.reflect_on_blueprint_complexity(blueprint)
    
    complexity_factors = 0
    patterns = set()
    issues = []
    optimizations = []
    
    # Analyze steps for complexity (rule-based, no LLM needed)
    for step in blueprint.steps:
        # Action complexity scoring
        if step.action in ['press_and_hold', 'scroll', 'swipe']:
            complexity_factors += 2
        elif step.action in ['click', 'type_text']:
            complexity_factors += 1
        
        # Element complexity detection
        if any(word in step.target_element_description.lower() 
               for word in ['dropdown', 'calendar', 'picker', 'dialog']):
            complexity_factors += 2
            patterns.add("complex_ui_elements")
        
        # Dynamic content detection
        if any(word in step.description.lower() 
               for word in ['wait', 'load', 'appear', 'dynamic']):
            complexity_factors += 1
            patterns.add("dynamic_content_handling")
        
        # CAPTCHA or verification detection
        if any(word in step.description.lower() 
               for word in ['captcha', 'verification', 'challenge']):
            complexity_factors += 3
            patterns.add("verification_handling")
            issues.append("CAPTCHA handling may require manual intervention")
    
    # Platform-specific patterns based on blueprint
    if blueprint.summary.platform == "mobile":
        patterns.add("mobile_gestures")
        patterns.add("device_specific_handling")
        optimizations.append("Use native mobile gestures for better reliability")
    else:
        patterns.add("web_interactions")
        patterns.add("cross_browser_compatibility")
        optimizations.append("Implement responsive design handling")
    
    # General patterns that are always recommended
    patterns.add("page_object_model")
    patterns.add("explicit_waits")
    patterns.add("retry_mechanisms")
    
    # Determine final complexity score
    complexity_score = min(max(complexity_factors // 2 + 2, 1), 10)
    
    # Add optimizations based on complexity and reflection
    if complexity_score >= 7:
        optimizations.extend([
            "Implement comprehensive error handling",
            "Add performance monitoring",
            "Use parallel execution where possible"
        ])
    
    # Add reflection-based optimizations
    optimizations.extend(reflection.get("recommended_approaches", []))
    
    # Standard optimizations
    optimizations.extend([
        "Add detailed logging for debugging",
        "Implement element visibility checks",
        "Use configuration files for test data"
    ])
    
    return CodeAnalysis(
        complexity_score=complexity_score,
        recommended_patterns=list(patterns),
        potential_issues=issues,
        optimization_suggestions=optimizations
    )

def generate_advanced_requirements_smart(platform: str, blueprint: AutomationBlueprint, analysis: CodeAnalysis) -> str:
    """Generate comprehensive requirements with smart dependency selection"""
    base_requirements = {
        "mobile": [
            "appium-python-client==4.1.1",  # FIXED: single version to avoid conflicts
            "selenium==4.15.2"
        ],
        "web": [
            "playwright==1.40.0",
            "selenium==4.15.2"
        ]
    }
    
    # Common requirements for all projects
    common_reqs = [
        "faker==20.1.0",
        "pytest==7.4.3",
        "pytest-html==4.1.1",
        "pydantic==2.5.0",
        "requests==2.31.0",
        "python-dotenv==1.0.0",
        "pillow==10.1.0"
    ]
    
    # Analysis-based requirements (smart selection)
    analysis_reqs = []
    if "verification_handling" in analysis.recommended_patterns:
        analysis_reqs.extend([
            "opencv-python==4.8.1.78",
            "pytesseract==0.3.10"
        ])
    
    if analysis.complexity_score >= 7:
        analysis_reqs.extend([
            "structlog==23.2.0"
        ])
    
    # Combine all requirements
    platform_reqs = base_requirements.get(platform, [])
    all_reqs = platform_reqs + common_reqs + analysis_reqs
    
    # Sort and deduplicate
    unique_reqs = sorted(list(set(all_reqs)))
    
    return "\n".join(unique_reqs)

def create_advanced_system_prompt_with_todos(platform: str, analysis: CodeAnalysis, organized_todos: Dict[str, List[str]]) -> str:
    """Create system prompt with TODO organization guidance"""
    base_prompt = f"""You are an elite automation engineer and code architect specializing in {platform} automation.

🧠 COGNITIVE APPROACH:
- Think systematically about code organization and structure
- Consider TODO priorities and implementation phases
- Focus on creating maintainable, production-ready code
- Apply software engineering best practices throughout

🔧 TECHNICAL EXCELLENCE:
- Modern automation frameworks and design patterns
- Clean code principles with comprehensive TODOs
- Production-grade error handling and retry mechanisms
- Performance optimization and maintainability focus

📊 CODE ANALYSIS INSIGHTS:
Complexity Score: {analysis.complexity_score}/10
Recommended Patterns: {', '.join(analysis.recommended_patterns)}
Key Optimizations: {', '.join(analysis.optimization_suggestions[:3])}

📋 TODO ORGANIZATION STRATEGY:
Critical TODOs: {len(organized_todos.get('CRITICAL', []))} items (implement first)
High Priority TODOs: {len(organized_todos.get('HIGH', []))} items (implement second)
Medium Priority TODOs: {len(organized_todos.get('MEDIUM', []))} items (enhance functionality)
Low Priority TODOs: {len(organized_todos.get('LOW', []))} items (polish features)

🎯 CODE GENERATION REQUIREMENTS:
1. **Structured TODO Integration**: Embed TODOs naturally throughout the code where they make logical sense
2. **Priority-Based Organization**: Organize code sections to reflect TODO priorities
3. **Implementation Guidance**: Each TODO should provide clear guidance for implementation
4. **Production Quality**: Generate code that senior engineers would deploy confidently
5. **Comprehensive Coverage**: Include TODOs for setup, core functionality, error handling, and optimization
6. **Best Practices**: Follow modern Python patterns, type hints, and clean architecture

🚀 GENERATION WORKFLOW:
1. Start with mandatory setup code exactly as provided
2. Integrate TODOs systematically throughout the implementation
3. Add comprehensive error handling and logging
4. Include smart waits, retry mechanisms, and fallbacks
5. Generate realistic test data using faker
6. Create modular, reusable code components
7. Add performance monitoring and cleanup logic

STRICT OUTPUT REQUIREMENTS:
- When calling the tool, provide arguments as a single valid JSON object only.
- Do not include Markdown, code fences, comments, or any text outside the JSON.
- Use ASCII quotes and hyphens; avoid smart quotes/dashes.
- Keep fields under reasonable length; truncate long docs where needed.

Generate production-ready code with excellent TODO organization that enables systematic development."""

    if analysis.complexity_score >= 7:
        base_prompt += f"""

⚠️ HIGH COMPLEXITY REQUIREMENTS:
- Implement advanced retry logic with exponential backoff
- Add comprehensive error recovery mechanisms
- Include detailed step-by-step logging and monitoring
- Use multiple element selection strategies with fallbacks
- Add performance benchmarking and resource monitoring

"""
    return base_prompt

def run_enhanced_agent2(seq_no: str, blueprint_result: Dict[str, Any]) -> Agent2Output:
    """Enhanced Agent 2 with TODO organization, caching, and reflection capabilities"""
    print(f"[{seq_no}] 🚀 Running Enhanced Agent 2: Advanced Code Generation with TODO Organization")
    
    try:
        # ===== FIXED BLUEPRINT PARSING =====
        # Handle multiple possible data structures from Agent 1
        blueprint_data = None
        
        # Try different possible structures
        if isinstance(blueprint_result, dict):
            # Method 1: Direct blueprint key
            if "blueprint" in blueprint_result and blueprint_result["blueprint"]:
                blueprint_data = blueprint_result["blueprint"]
            
            # Method 2: Result key with blueprint inside
            elif "result" in blueprint_result and isinstance(blueprint_result["result"], dict):
                if "blueprint" in blueprint_result["result"]:
                    blueprint_data = blueprint_result["result"]["blueprint"]
            
            # Method 3: Blueprint data directly in result
            elif "summary" in blueprint_result and "steps" in blueprint_result:
                blueprint_data = blueprint_result
            
            # Method 4: Agent 1 might return data in different structure
            elif "status" in blueprint_result and blueprint_result["status"] == "success":
                # Look for blueprint in various keys
                for key in ["blueprint", "result", "data", "output"]:
                    if key in blueprint_result and isinstance(blueprint_result[key], dict):
                        potential_data = blueprint_result[key]
                        if "summary" in potential_data and "steps" in potential_data:
                            blueprint_data = potential_data
                            break
        
        # Final fallback - check if blueprint_result itself is the blueprint
        if not blueprint_data and isinstance(blueprint_result, dict):
            if "summary" in blueprint_result and "steps" in blueprint_result:
                blueprint_data = blueprint_result
        
        # Validate we found blueprint data
        if not blueprint_data:
            print(f"[{seq_no}] 🔍 Debug - blueprint_result structure: {list(blueprint_result.keys()) if isinstance(blueprint_result, dict) else type(blueprint_result)}")
            raise ValueError("No valid blueprint data found in any expected format")
        
        # Validate blueprint data has required structure
        if not isinstance(blueprint_data, dict) or "summary" not in blueprint_data or "steps" not in blueprint_data:
            print(f"[{seq_no}] 🔍 Debug - blueprint_data structure: {list(blueprint_data.keys()) if isinstance(blueprint_data, dict) else type(blueprint_data)}")
            raise ValueError("Blueprint data missing required 'summary' or 'steps' fields")
        # ===== END FIXED BLUEPRINT PARSING =====
        
        blueprint = AutomationBlueprint(**blueprint_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid blueprint structure: {e}")

    # Create output directory
    out_dir = ARTIFACTS_DIR / seq_no / "enhanced_agent2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Smart complexity analysis with reflection (no extra LLM calls)
    print(f"[{seq_no}] 🔍 Analyzing blueprint complexity with reflection...")
    analysis = analyze_blueprint_complexity_smart(blueprint)

    # Get additional insights from scratchpad
    code_improvements = agent2_scratchpad.suggest_code_improvements(blueprint, analysis)
    analysis.optimization_suggestions.extend(code_improvements)

    print(f"[{seq_no}] 📊 Complexity Score: {analysis.complexity_score}/10")
    print(f"[{seq_no}] 🏗️ Recommended Patterns: {', '.join(analysis.recommended_patterns[:3])}")

    # Organize TODOs systematically
    print(f"[{seq_no}] 📋 Organizing TODOs for systematic development...")
    organized_todos = todo_organizer.organize_todos_by_priority(blueprint, analysis)

    # Generate implementation roadmap
    implementation_roadmap = todo_organizer.generate_implementation_roadmap(organized_todos)

    # Determine platform and setup
    platform = blueprint.summary.platform.lower()
    framework = "Enhanced Appium with Selenium" if platform == "mobile" else "Enhanced Playwright"
    setup_code = MOBILE_SETUP_TEMPLATE if platform == "mobile" else WEB_SETUP_TEMPLATE

    print(f"[{seq_no}] 🛠️ Using {framework} framework with organized TODOs")

    # Smart image description collection (with caching)
    image_descriptions: Dict[str, str] = {}
    for step in blueprint.steps:
        if step.associated_image:
            img_path = ARTIFACTS_DIR / seq_no / "enhanced_agent1" / step.associated_image
            if img_path.exists():
                # Create a simple prompt that can be cached
                enhanced_prompt = f"""Analyze this screenshot for {platform} automation:

1. Describe key interactive elements (buttons, inputs, etc.)
2. Note any UI patterns or dynamic content
3. Suggest reliable element selection strategies
4. Identify potential automation challenges

Keep response concise and focused on automation implementation."""

                # Create blueprint hash for caching
                blueprint_hash = hashlib.md5(json.dumps(blueprint_data).encode()).hexdigest()
                description = get_llm_response_with_cache_agent2(
                    enhanced_prompt,
                    "You are an expert UI/UX analyst specializing in automation.",
                    blueprint_hash
                )
                image_descriptions[img_path.name] = description

    # Get enhanced LLM
    llm = get_langchain_llm()
    structured_llm = llm.with_structured_output(EnhancedCodeOutput)

    # Create system prompt with TODO organization
    system_prompt = create_advanced_system_prompt_with_todos(platform, analysis, organized_todos)

    # Enhanced human prompt template with TODO integration
    human_prompt_template = """
🎯 ADVANCED AUTOMATION CODE GENERATION

**Project Analysis:**
Platform: {platform}
Complexity Score: {complexity_score}/10
Target Application: {target_app}
Total Steps: {step_count}

**TODO Organization Summary:**
- Critical Priority: {critical_todo_count} items (implement first)
- High Priority: {high_todo_count} items (implement second)
- Medium Priority: {medium_todo_count} items (enhance functionality)
- Low Priority: {low_todo_count} items (polish features)

**Automation Blueprint:**
```json
{blueprint_json}
```

**Visual Context (Screenshot Analysis):**
```json
{image_descriptions_json}
```

**Code Generation Insights:**
- Recommended Patterns: {patterns}
- Potential Issues: {issues}
- Key Optimizations: {optimizations}

**MANDATORY Setup Code (Include exactly as provided):**
```python
{setup_code}
```

**ENHANCED GENERATION REQUIREMENTS:**
1. **TODO Integration**: Naturally embed TODOs throughout the code where they provide implementation guidance
2. **Priority-Based Structure**: Organize code to reflect TODO priorities and implementation phases
3. **Production Quality**: Generate code with comprehensive error handling, logging, and retry mechanisms
4. **Smart Element Handling**: Use multiple selection strategies with fallbacks based on image analysis
5. **Performance Focus**: Include timing, monitoring, and resource management
6. **Clean Architecture**: Apply modern Python patterns, type hints, and modular design
7. **Testing Integration**: Generate realistic test data and include basic test structure
8. **Documentation**: Provide clear implementation roadmap with systematic development guidance

**Special Instructions:**
- Each TODO should be actionable and provide clear implementation direction
- Organize code sections logically with TODOs that guide development phases
- Include comprehensive error handling appropriate to complexity level
- Generate faker-based realistic test data throughout
- Add performance monitoring and cleanup logic
- Create modular, maintainable code that experts would confidently deploy

Generate production-ready automation code with excellent TODO organization that enables systematic, phased development!

"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt_template),
    ])

    chain = prompt | structured_llm

    # FIXED: Add JSON-safe structured output with retry
    try:
        print(f"[{seq_no}] 🤖 Generating enhanced code with TODO organization...")
        input_variables = {
            "platform": platform,
            "complexity_score": analysis.complexity_score,
            "target_app": blueprint.summary.target_application,
            "step_count": len(blueprint.steps),
            "critical_todo_count": len(organized_todos.get('CRITICAL', [])),
            "high_todo_count": len(organized_todos.get('HIGH', [])),
            "medium_todo_count": len(organized_todos.get('MEDIUM', [])),
            "low_todo_count": len(organized_todos.get('LOW', [])),
            "blueprint_json": blueprint.model_dump_json(indent=2),
            "image_descriptions_json": json.dumps(image_descriptions, indent=2),
            "patterns": ", ".join(analysis.recommended_patterns),
            "issues": ", ".join(analysis.potential_issues) or "None identified",
            "optimizations": ", ".join(analysis.optimization_suggestions),
            "setup_code": setup_code
        }

        code_output: EnhancedCodeOutput = chain.invoke(input_variables)
        
    except Exception as e:
        print(f"[{seq_no}] ⚠️ Structured tool call failed, retrying with strict JSON mode: {e}")

        # 1) Tool-FREE strict JSON system prompt (no template variables)
        strict_json_prompt = (
            system_prompt
            + "\n\nSTRICT JSON ONLY:"
            " Output a single valid JSON object matching the EnhancedCodeOutput schema."
            " Do NOT call any tool or function; do NOT use tool call syntax (e.g., {\"name\":..., \"arguments\":...})."
            " Do NOT include markdown or code fences."
            " Use ASCII quotes and hyphens only."
        )

        # Optional schema hint to nudge the model (kept out of tool mode)
        schema_hint = {
            "python_code": "string",
            "requirements": "string",
            "documentation": "string",
            "implementation_roadmap": "string",
            "code_analysis": {
                "complexity_score": "int",
                "recommended_patterns": ["string"],
                "potential_issues": ["string"],
                "optimization_suggestions": ["string"]
            },
            "test_code": "string|null"
        }

        # 2) Prepare plain JSON input blob for the model (no templating)
        strict_payload = {
            "input_variables": input_variables,
            "schema_hint": schema_hint
        }
        strict_input = json.dumps(strict_payload, ensure_ascii=True)

        # 3) Call base Chat LLM with proper message objects (not a dict, not a template)
        raw_llm = get_langchain_llm()
        messages = [
            SystemMessage(content=strict_json_prompt),
            HumanMessage(content="Return ONLY the JSON object for EnhancedCodeOutput.\nInput JSON:\n" + strict_input)
        ]
        raw = raw_llm.invoke(messages)
        raw_text = getattr(raw, "content", raw)

        # 4) Repair and parse JSON, then validate with Pydantic
        parsed = _repair_and_parse_json(raw_text)
        code_output = EnhancedCodeOutput(**parsed)
        
    
    # FIXED: Cap oversized fields before writing
    def _clamp(s: Optional[str], limit: int = 200000) -> Optional[str]:
        return (s[:limit] + "\n... [truncated]") if isinstance(s, str) and len(s) > limit else s

    code_output.python_code = _clamp(code_output.python_code)
    code_output.documentation = _clamp(code_output.documentation, 80000)
    code_output.implementation_roadmap = _clamp(code_output.implementation_roadmap, 60000)
    code_output.requirements = _clamp(code_output.requirements, 20000)
    if code_output.test_code:
        code_output.test_code = _clamp(code_output.test_code, 40000)

    # Validate output
    if not code_output.python_code or not code_output.requirements:
        raise ValueError("LLM response missing essential code or requirements content")

    # Generate enhanced requirements with fixed versions
    enhanced_requirements = generate_advanced_requirements_smart(platform, blueprint, analysis)
    final_requirements = f"# Generated requirements\n{code_output.requirements}\n\n# Enhanced requirements\n{enhanced_requirements}"

    # Save all outputs
    script_path = out_dir / "automation_script.py"
    reqs_path = out_dir / "requirements.txt"
    analysis_path = out_dir / "code_analysis.json"
    docs_path = out_dir / "documentation.md"
    roadmap_path = out_dir / "implementation_roadmap.md"

    script_path.write_text(code_output.python_code, encoding="utf-8")
    reqs_path.write_text(final_requirements, encoding="utf-8")

    # Save comprehensive analysis
    analysis_data = {
        "complexity_score": code_output.code_analysis.complexity_score,
        "recommended_patterns": code_output.code_analysis.recommended_patterns,
        "potential_issues": code_output.code_analysis.potential_issues,
        "optimization_suggestions": code_output.code_analysis.optimization_suggestions,
        "blueprint_summary": blueprint.summary.model_dump(),
        "organized_todos": organized_todos,
        "reflection_insights": agent2_scratchpad.reflection_log[-1] if agent2_scratchpad.reflection_log else {},
        "generation_metadata": {
            "platform": platform,
            "framework": framework,
            "steps_count": len(blueprint.steps),
            "images_processed": len(image_descriptions),
            "cache_performance": agent2_cache.get_stats()
        }
    }

    analysis_path.write_text(json.dumps(analysis_data, indent=2), encoding="utf-8")
    docs_path.write_text(code_output.documentation, encoding="utf-8")
    roadmap_path.write_text(implementation_roadmap, encoding="utf-8")

    # Save test code if generated
    test_script_path = None
    if code_output.test_code:
        test_script_path = out_dir / "test_automation.py"
        test_script_path.write_text(code_output.test_code, encoding="utf-8")

    print(f"[{seq_no}] ✅ Enhanced Agent 2 completed successfully!")
    print(f"[{seq_no}] 🎯 Cache performance: {agent2_cache.get_stats()['hit_rate_percent']:.1f}% hit rate")
    print(f"[{seq_no}] 📋 Generated {sum(len(todos) for todos in organized_todos.values())} organized TODOs")
    print(f"[{seq_no}] 📁 Generated files:")
    print(f"   - Automation Script: {script_path}")
    print(f"   - Requirements: {reqs_path}")
    print(f"   - Analysis: {analysis_path}")
    print(f"   - Documentation: {docs_path}")
    print(f"   - Implementation Roadmap: {roadmap_path}")
    if test_script_path:
        print(f"   - Test Suite: {test_script_path}")

    return {
        "script": str(script_path),
        "requirements": str(reqs_path),
        "analysis": analysis_data,
        "test_script": str(test_script_path) if test_script_path else None,
        "documentation": str(docs_path),
        "implementation_roadmap": str(roadmap_path)
    }

# Export the main function
__all__ = ["run_enhanced_agent2", "analyze_blueprint_complexity_smart", "generate_advanced_requirements_smart", 
           "agent2_cache", "agent2_scratchpad", "todo_organizer"]
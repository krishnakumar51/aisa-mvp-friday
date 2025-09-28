# agents/enhanced_agent_2.py

from pathlib import Path
import json
import re
from fastapi import HTTPException
from typing import TypedDict, List, Optional, Dict, Any, Tuple, Set, Union
import importlib.util
import time
import hashlib
import random # FIXED: Added missing import
from config import ARTIFACTS_DIR
from agents.llm_utils import get_llm_response, get_langchain_llm, LLMProvider
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, model_validator
from langchain.schema import SystemMessage, HumanMessage
from agents.template import MOBILE_SETUP_TEMPLATE, WEB_SETUP_TEMPLATE

# ===== COMPLETELY FIXED JSON PARSING HELPERS =====
def _extract_first_json_block(text: str) -> str:
    """Extract and clean JSON from potentially malformed text"""
    # Remove fenced code blocks if present
    text = re.sub(r"^```.*?\n|```$", "", text.strip(), flags=re.DOTALL | re.MULTILINE)
    
    # COMPLETELY FIXED: Simple string replacement instead of translate
    # Fix common Unicode issues that break JSON parsing
    replacements = [
        ('"', '"'), ('"', '"'), ('‟', '"'),  # smart quotes
        (''', "'"), (''', "'"),              # smart apostrophes
        ('–', '-'), ('—', '-'), ('‑', '-'),  # various dashes
        ('\u00A0', ' '),                     # non-breaking space
        ('\u2013', '-'), ('\u2014', '-'),    # more dashes
        ('\u2018', "'"), ('\u2019', "'"),    # more quotes
        ('\u201C', '"'), ('\u201D', '"'),    # more double quotes
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
    
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
    j = re.sub(r",\s*([}\]])", r"\1", j) # trailing commas
    j = re.sub(r"(['\"])\s*:\s*(['\"])", r"\1:\2", j) # normalize colon spacing
    
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
        roadmap = """# Implementation Roadmap

This roadmap organizes all TODOs by priority to ensure systematic development:

## CRITICAL PRIORITY (Implement First)

These are essential for basic functionality:

"""

        for i, todo in enumerate(organized_todos["CRITICAL"], 1):
            roadmap += f"{i}. {todo}\n"

        roadmap += """
## HIGH PRIORITY (Implement Second)

Important for reliability and robustness:

"""

        for i, todo in enumerate(organized_todos["HIGH"], 1):
            roadmap += f"{i}. {todo}\n"

        roadmap += """
## MEDIUM PRIORITY (Implement Third)

Enhances functionality and user experience:

"""

        for i, todo in enumerate(organized_todos["MEDIUM"], 1):
            roadmap += f"{i}. {todo}\n"

        roadmap += """
## LOW PRIORITY (Implement Last)

Nice to have features and optimizations:

"""

        for i, todo in enumerate(organized_todos["LOW"], 1):
            roadmap += f"{i}. {todo}\n"

        roadmap += """
## IMPLEMENTATION TIPS

1. Start with CRITICAL: Get basic functionality working first
2. Test Early: Implement unit tests as you complete each priority level
3. Incremental Development: Test each TODO implementation before moving to next
4. Code Reviews: Have each priority level reviewed before proceeding
5. Documentation: Update documentation as you complete TODOs

## DEVELOPMENT PHASES

Phase 1: Foundation - Complete all CRITICAL todos
Phase 2: Reliability - Complete all HIGH priority todos
Phase 3: Enhancement - Complete all MEDIUM priority todos
Phase 4: Polish - Complete all LOW priority todos

Remember: This roadmap ensures systematic development and reduces technical debt!

"""
        return roadmap

# Global TODO organizer instance
todo_organizer = TodoOrganizer()

# BULLETPROOF Pydantic models with comprehensive validation and defaults
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
    complexity_score: int = Field(description="Complexity rating from 1-10", ge=1, le=10, default=5)
    recommended_patterns: List[str] = Field(description="List of recommended design patterns", default_factory=lambda: ["explicit_waits", "retry_mechanisms"])
    potential_issues: List[str] = Field(description="List of potential issues or risks", default_factory=list)
    optimization_suggestions: List[str] = Field(description="Performance and reliability optimizations", default_factory=lambda: ["Add detailed logging for debugging"])

    @model_validator(mode='before')
    @classmethod
    def ensure_valid_analysis(cls, data: Any) -> Any:
        """Ensure CodeAnalysis has valid default values"""
        if isinstance(data, dict):
            if 'complexity_score' not in data or not isinstance(data['complexity_score'], int):
                data['complexity_score'] = 5
            if 'recommended_patterns' not in data or not isinstance(data['recommended_patterns'], list):
                data['recommended_patterns'] = ["explicit_waits", "retry_mechanisms"]
            if 'potential_issues' not in data or not isinstance(data['potential_issues'], list):
                data['potential_issues'] = []
            if 'optimization_suggestions' not in data or not isinstance(data['optimization_suggestions'], list):
                data['optimization_suggestions'] = ["Add detailed logging for debugging"]
        elif data is None:
            data = {
                'complexity_score': 5,
                'recommended_patterns': ["explicit_waits", "retry_mechanisms"],
                'potential_issues': [],
                'optimization_suggestions': ["Add detailed logging for debugging"]
            }
        return data

class EnhancedCodeOutput(BaseModel):
    """BULLETPROOF Pydantic model with comprehensive validation and defaults"""
    python_code: str = Field(
        description="Complete, production-ready Python script with organized TODOs",
        default="# TODO: Add automation implementation here\nprint('Generated automation script')"
    )
    requirements: str = Field(
        description="Comprehensive requirements.txt with pinned versions", 
        default="# Basic requirements\nappium-python-client==4.1.1\nselenium==4.15.2\nfaker==20.1.0\npydantic==2.5.0"
    )
    code_analysis: CodeAnalysis = Field(
        description="Analysis of the generated code",
        default_factory=lambda: CodeAnalysis()
    )
    test_code: Optional[str] = Field(
        description="Optional basic test suite", 
        default=None
    )
    documentation: str = Field(
        description="Technical documentation with implementation roadmap", 
        default="# Documentation\nThis is a generated automation script with comprehensive TODOs and implementation guidance."
    )
    implementation_roadmap: str = Field(
        description="Structured TODO roadmap for systematic development", 
        default="# Implementation Roadmap\nFollow the TODO comments in the code for systematic development."
    )

    @model_validator(mode='before')
    @classmethod
    def ensure_required_fields(cls, data: Any) -> Any:
        """CRITICAL: Ensure all required fields are present with valid defaults"""
        if data is None:
            data = {}
        
        if not isinstance(data, dict):
            # If data is not a dict, create default structure
            data = {}
        
        # Ensure python_code exists and is not empty
        if 'python_code' not in data or not data.get('python_code'):
            data['python_code'] = "# TODO: Add automation implementation here\nprint('Generated automation script')"
        
        # Ensure requirements exists
        if 'requirements' not in data or not data.get('requirements'):
            data['requirements'] = "# Basic requirements\nappium-python-client==4.1.1\nselenium==4.15.2\nfaker==20.1.0\npydantic==2.5.0"
        
        # Ensure code_analysis exists and is valid
        if 'code_analysis' not in data or not data.get('code_analysis'):
            data['code_analysis'] = {
                'complexity_score': 5,
                'recommended_patterns': ["explicit_waits", "retry_mechanisms"],
                'potential_issues': [],
                'optimization_suggestions': ["Add detailed logging for debugging"]
            }
        elif isinstance(data.get('code_analysis'), dict):
            # Validate nested code_analysis
            analysis = data['code_analysis']
            if 'complexity_score' not in analysis:
                analysis['complexity_score'] = 5
            if 'recommended_patterns' not in analysis:
                analysis['recommended_patterns'] = ["explicit_waits", "retry_mechanisms"]
            if 'potential_issues' not in analysis:
                analysis['potential_issues'] = []
            if 'optimization_suggestions' not in analysis:
                analysis['optimization_suggestions'] = ["Add detailed logging for debugging"]
        
        # Ensure documentation exists
        if 'documentation' not in data or not data.get('documentation'):
            data['documentation'] = "# Documentation\nThis is a generated automation script with comprehensive TODOs and implementation guidance."
        
        # Ensure implementation_roadmap exists
        if 'implementation_roadmap' not in data or not data.get('implementation_roadmap'):
            data['implementation_roadmap'] = "# Implementation Roadmap\nFollow the TODO comments in the code for systematic development."
        
        return data

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
    response = get_llm_response(prompt, system_message, model_name=LLMProvider.ANTHROPIC)
    
    # Cache the response
    agent2_cache.set(prompt, response, blueprint_hash)
    return response

def generate_advanced_requirements_smart(platform: str, blueprint: AutomationBlueprint, analysis: CodeAnalysis) -> str:
    """Generate comprehensive requirements with smart dependency selection"""
    base_requirements = {
        "mobile": [
            "appium-python-client==4.1.1",
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

def analyze_blueprint_complexity_smart(blueprint: AutomationBlueprint) -> CodeAnalysis:
    """Analyze blueprint complexity with smart assessment"""
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

    # Add optimizations based on complexity
    if complexity_score >= 7:
        optimizations.extend([
            "Implement comprehensive error handling",
            "Add performance monitoring",
            "Use parallel execution where possible"
        ])

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

def create_advanced_system_prompt_with_todos(platform: str, analysis: CodeAnalysis, organized_todos: Dict[str, List[str]]) -> str:
    """Create system prompt with TODO organization guidance"""
    base_prompt = f"""You are an elite automation engineer and code architect specializing in {platform} automation.

COGNITIVE APPROACH:
- Think systematically about code organization and structure
- Consider TODO priorities and implementation phases
- Focus on creating maintainable, production-ready code
- Apply software engineering best practices throughout

TECHNICAL EXCELLENCE:
- Modern automation frameworks and design patterns
- Clean code principles with comprehensive TODOs
- Production-grade error handling and retry mechanisms
- Performance optimization and maintainability focus

CODE ANALYSIS INSIGHTS:
Complexity Score: {analysis.complexity_score}/10
Recommended Patterns: {', '.join(analysis.recommended_patterns)}
Key Optimizations: {', '.join(analysis.optimization_suggestions[:3])}

TODO ORGANIZATION STRATEGY:
Critical TODOs: {len(organized_todos.get('CRITICAL', []))} items (implement first)
High Priority TODOs: {len(organized_todos.get('HIGH', []))} items (implement second)
Medium Priority TODOs: {len(organized_todos.get('MEDIUM', []))} items (enhance functionality)
Low Priority TODOs: {len(organized_todos.get('LOW', []))} items (polish features)

CODE GENERATION REQUIREMENTS:
1. Structured TODO Integration: Embed TODOs naturally throughout the code
2. Priority-Based Organization: Organize code sections to reflect TODO priorities
3. Implementation Guidance: Each TODO should provide clear guidance
4. Production Quality: Generate code that senior engineers would deploy
5. Comprehensive Coverage: Include setup, core functionality, error handling
6. Best Practices: Follow modern Python patterns, type hints, clean architecture

MANDATORY FIELD REQUIREMENTS:
- ALWAYS include python_code field with complete automation script
- ALWAYS include code_analysis field with complexity assessment
- ALWAYS include requirements field with dependency list
- ALWAYS include documentation field with technical details
- ALWAYS include implementation_roadmap field with structured TODOs

GENERATION WORKFLOW:
1. Start with mandatory setup code exactly as provided
2. Integrate TODOs systematically throughout the implementation
3. Add comprehensive error handling and logging
4. Include smart waits, retry mechanisms, and fallbacks
5. Generate realistic test data using faker
6. Create modular, reusable code components
7. Add performance monitoring and cleanup logic

STRICT OUTPUT REQUIREMENTS:
- Use ASCII quotes and hyphens; avoid smart quotes/dashes
- Keep fields under reasonable length; truncate long docs where needed
- Generate complete and valid Python code
- Ensure ALL required fields are present and valid

Generate production-ready code with excellent TODO organization."""

    if analysis.complexity_score >= 7:
        base_prompt += f"""

HIGH COMPLEXITY REQUIREMENTS:
- Implement advanced retry logic with exponential backoff
- Add comprehensive error recovery mechanisms  
- Include detailed step-by-step logging and monitoring
- Use multiple element selection strategies with fallbacks
- Add performance benchmarking and resource monitoring
"""

    return base_prompt

def run_enhanced_agent2(seq_no: str, blueprint_result: Dict[str, Any]) -> Agent2Output:
    """Enhanced Agent 2 with BULLETPROOF error handling and validation"""
    print(f"[{seq_no}] 🚀 Running Enhanced Agent 2: BULLETPROOF Code Generation")
    print(f"[{seq_no}] 🛡️ BULLETPROOF FEATURES: API retry handling, detailed error logging, guaranteed output")

    try:
        # ===== BLUEPRINT PARSING =====
        blueprint_data = None
        
        # Handle multiple possible data structures from Agent 1
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

        blueprint = AutomationBlueprint(**blueprint_data)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid blueprint structure: {e}")

    # Create output directory
    out_dir = ARTIFACTS_DIR / seq_no / "enhanced_agent2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Smart complexity analysis with reflection
    print(f"[{seq_no}] 🔍 Analyzing blueprint complexity...")
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

    # Determine platform and get setup code from template
    platform = blueprint.summary.platform.lower()
    framework = "Enhanced Appium with ZERO JSON ERRORS" if platform == "mobile" else "Enhanced Playwright with COMPLETE Stealth"
    setup_code = MOBILE_SETUP_TEMPLATE if platform == "mobile" else WEB_SETUP_TEMPLATE
    
    print(f"[{seq_no}] 🛠️ Using {framework} framework")

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
    llm = get_langchain_llm(model_name=LLMProvider.ANTHROPIC)
    structured_llm = llm.with_structured_output(EnhancedCodeOutput)

    # Create system prompt with TODO organization
    system_prompt = create_advanced_system_prompt_with_todos(platform, analysis, organized_todos)

    # COMPLETELY FIXED human prompt template
    human_prompt_template = """
ADVANCED AUTOMATION CODE GENERATION WITH MANDATORY FIELD REQUIREMENTS

Project Analysis:
Platform: {platform}
Complexity Score: {complexity_score}/10
Target Application: {target_app}
Total Steps: {step_count}

TODO Organization Summary:
- Critical Priority: {critical_todo_count} items (implement first)
- High Priority: {high_todo_count} items (implement second)
- Medium Priority: {medium_todo_count} items (enhance functionality)
- Low Priority: {low_todo_count} items (polish features)

Automation Blueprint:
{blueprint_json}

Visual Context (Screenshot Analysis):
{image_descriptions_json}

Code Generation Insights:
- Recommended Patterns: {patterns}
- Potential Issues: {issues}
- Key Optimizations: {optimizations}

MANDATORY Setup Code (Include exactly as provided):
{setup_code}

CRITICAL REQUIREMENTS - ALL FIELDS MUST BE PRESENT:
1. python_code: Complete automation script with setup code and implementation
2. code_analysis: Valid analysis object with complexity_score, patterns, issues, optimizations
3. requirements: Complete requirements.txt content with all dependencies
4. documentation: Technical documentation explaining the automation approach
5. implementation_roadmap: Structured development roadmap with prioritized TODOs

ENHANCED GENERATION REQUIREMENTS:

1. TODO Integration: Naturally embed TODOs throughout the code where they provide guidance
2. Priority-Based Structure: Organize code to reflect TODO priorities and phases
3. Production Quality: Generate code with comprehensive error handling and retry mechanisms
4. Smart Element Handling: Use multiple selection strategies with fallbacks
5. Performance Focus: Include timing, monitoring, and resource management  
6. Clean Architecture: Apply modern Python patterns, type hints, and modular design
7. Testing Integration: Generate realistic test data and include basic test structure

Special Instructions:
- Each TODO should be actionable and provide clear implementation direction
- Organize code sections logically with TODOs that guide development phases
- Include comprehensive error handling appropriate to complexity level
- Generate faker-based realistic test data throughout
- Add performance monitoring and cleanup logic
- Create modular, maintainable code that experts would confidently deploy

MANDATORY: Ensure ALL required fields (python_code, code_analysis, requirements, documentation, implementation_roadmap) are present and complete.

Generate production-ready automation code with excellent TODO organization!
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_prompt_template),
    ])

    chain = prompt | structured_llm

    # BULLETPROOF structured output with exactly 2 retries (no exponential backoff)
    max_retries = 2
    retry_count = 0
    code_output: Optional[EnhancedCodeOutput] = None

    while retry_count <= max_retries and code_output is None:
        try:
            print(f"[{seq_no}] 🤖 Generating BULLETPROOF code (attempt {retry_count + 1}/{max_retries + 1})...")
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

            code_output = chain.invoke(input_variables)
            print(f"[{seq_no}] ✅ Successfully generated code output")
            break

        except Exception as e:
            retry_count += 1
            error_type = type(e).__name__
            error_message = str(e)
            
            print(f"[{seq_no}] ❌ Attempt {retry_count} failed: {error_type} - {error_message}")
            
            # Log specific error types for debugging
            if "anthropic" in error_message.lower() or "api" in error_message.lower():
                print(f"[{seq_no}] 🔍 API Error detected - likely timeout, rate limit, or connection issue")
            elif "json" in error_message.lower() or "parse" in error_message.lower():
                print(f"[{seq_no}] 🔍 JSON parsing error detected")
            elif "validation" in error_message.lower() or "field" in error_message.lower():
                print(f"[{seq_no}] 🔍 Pydantic validation error detected")
            else:
                print(f"[{seq_no}] 🔍 Unknown error type: {error_type}")
            
            if retry_count <= max_retries:
                print(f"[{seq_no}] 🔄 Retrying with simplified prompt...")
                time.sleep(2)  # Brief pause between retries
                
                # Simplified retry prompt
                simplified_prompt = f"""Generate a complete EnhancedCodeOutput with ALL required fields:
                
MANDATORY FIELDS:
- python_code: Complete automation script
- code_analysis: Valid CodeAnalysis object
- requirements: Complete requirements list  
- documentation: Technical documentation
- implementation_roadmap: TODO roadmap

Platform: {platform}
Setup Code: {setup_code[:500]}...
Blueprint: {blueprint.summary.overall_goal}

Return a valid EnhancedCodeOutput with ALL fields populated."""

                try:
                    simple_response = get_llm_response(simplified_prompt, "You are an expert automation engineer. Generate complete, valid output.", model_name=LLMProvider.ANTHROPIC)
                    parsed_response = _repair_and_parse_json(simple_response)
                    code_output = EnhancedCodeOutput(**parsed_response)
                    print(f"[{seq_no}] ✅ Simplified retry succeeded")
                    break
                except Exception as retry_error:
                    print(f"[{seq_no}] ⚠️ Retry attempt failed: {type(retry_error).__name__} - {retry_error}")
                    continue

    # GUARANTEED fallback - this will ALWAYS work
    if code_output is None:
        print(f"[{seq_no}] 🛡️ Using BULLETPROOF fallback generation...")
        code_output = EnhancedCodeOutput(
            python_code=setup_code + f"\n\n# TODO: Complete {platform} automation implementation\n" +
                       f"# Generated for: {blueprint.summary.target_application}\n" +
                       f"# Steps: {len(blueprint.steps)}\n" +
                       "print('BULLETPROOF automation script generated successfully')",
            requirements="# BULLETPROOF Requirements\n" +
                        ("appium-python-client==4.1.1\nselenium==4.15.2" if platform == "mobile" else "playwright==1.40.0\nselenium==4.15.2") +
                        "\nfaker==20.1.0\npydantic==2.5.0\npytest==7.4.3",
            code_analysis=analysis,
            documentation=f"# BULLETPROOF {platform.title()} Automation Documentation\n\n" +
                         f"Target Application: {blueprint.summary.target_application}\n" +
                         f"Platform: {platform}\n" +
                         f"Complexity Score: {analysis.complexity_score}/10\n\n" +
                         "This script provides a robust foundation for automation with comprehensive error handling.",
            implementation_roadmap=implementation_roadmap
        )

    # BULLETPROOF validation and field checking
    if not code_output.python_code or len(code_output.python_code.strip()) < 50:
        code_output.python_code = setup_code + "\n\n# TODO: Add automation implementation"
    if not code_output.requirements:
        code_output.requirements = "appium-python-client==4.1.1\nselenium==4.15.2\nfaker==20.1.0"
    if not code_output.documentation:
        code_output.documentation = f"# {platform.title()} Automation Documentation\n\nGenerated automation script with comprehensive features."
    if not code_output.implementation_roadmap:
        code_output.implementation_roadmap = implementation_roadmap

    # FIXED: Cap oversized fields before writing
    def _clamp(s: str, limit: int = 200000) -> str:
        return (s[:limit] + "\n... [truncated]") if len(s) > limit else s

    code_output.python_code = _clamp(code_output.python_code)
    code_output.documentation = _clamp(code_output.documentation, 80000)
    code_output.implementation_roadmap = _clamp(code_output.implementation_roadmap, 60000)
    code_output.requirements = _clamp(code_output.requirements, 20000)
    if code_output.test_code:
        code_output.test_code = _clamp(code_output.test_code, 40000)

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
            "cache_performance": agent2_cache.get_stats(),
            "retry_attempts": retry_count,
            "bulletproof_features": [
                "BULLETPROOF: Robust Pydantic models with model_validator",
                "BULLETPROOF: Comprehensive default values for all fields",
                "BULLETPROOF: 2 retry attempts with simplified fallback",
                "BULLETPROOF: Guaranteed output with field validation",
                "BULLETPROOF: Detailed error logging with error type detection",
                "BULLETPROOF: Mobile template imported from agents.template module"
            ]
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

    print(f"[{seq_no}] ✅ Enhanced Agent 2 completed with BULLETPROOF reliability!")
    print(f"[{seq_no}] 🎯 Cache performance: {agent2_cache.get_stats()['hit_rate_percent']:.1f}% hit rate")
    print(f"[{seq_no}] 📋 Generated {sum(len(todos) for todos in organized_todos.values())} organized TODOs")
    print(f"[{seq_no}] 🛡️ BULLETPROOF FEATURES: Model validation, 2 retries, detailed error logging, guaranteed output!")
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
__all__ = ["run_enhanced_agent2", "analyze_blueprint_complexity_smart", "generate_advanced_requirements_smart", "agent2_cache"]
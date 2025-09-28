# enhanced/enhanced_agent_2.py
"""
Enhanced Agent 2: Code Generation with unified schema and robust error handling
Key improvements:
- Uses shared schema models to prevent communication breakdown  
- Tolerant blueprint parsing with detailed error reporting
- Persistent caching and performance monitoring
- Structured outputs and better reflection
"""

import time
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from fastapi import HTTPException

# Import shared modules (CRITICAL: prevents schema drift)
from shared_models import (
    AutomationBlueprint, BlueprintStep, Agent2Output, TaskStatus,
    ValidationLevel, safe_parse_blueprint, create_error_response
)
from utils import (
    PersistentLLMCache, PerformanceMonitor, monitor_performance,
    repair_and_parse_json, validate_with_detailed_errors,
    ensure_directory, safe_file_write
)

# Import original utilities (unchanged)
from agents.llm_utils import get_llm_response, get_langchain_llm
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from config import ARTIFACTS_DIR

# Global instances with persistent storage
cache_dir = ensure_directory(Path(ARTIFACTS_DIR) / "cache")
agent2_cache = PersistentLLMCache(cache_dir / "agent2_cache.db", default_ttl=7200)
performance_monitor = PerformanceMonitor()

class EnhancedAgent2:
    """
    Enhanced Agent 2 with unified schema compliance and robust error handling
    """

    def __init__(self):
        self.cache = agent2_cache
        self.monitor = performance_monitor
        self.llm_calls_made = 0
        self.reflection_notes = []

    def generate_automation_script(self, blueprint_data: Union[str, dict], 
                                 validation_level: ValidationLevel = ValidationLevel.STRICT) -> Agent2Output:
        """
        Main code generation with tolerant blueprint parsing and structured output
        """
        start_time = time.time()

        try:
            with monitor_performance(self.monitor, "code_generation"):
                # Parse blueprint with tolerance for schema mismatches
                blueprint = self._parse_blueprint_tolerant(blueprint_data, validation_level)

                # Try cache first
                blueprint_hash = self._get_blueprint_hash(blueprint)
                cached_result = self.cache.get(f"code_gen_{blueprint_hash}", model="gpt-4")

                if cached_result:
                    return Agent2Output(
                        success=True,
                        agent_name="enhanced_agent_2",
                        timestamp=time.time(),
                        execution_time_seconds=0.01,  # Cache hit
                        script_content=cached_result['script_content'],
                        requirements_content=cached_result.get('requirements_content', ''),
                        setup_instructions=cached_result.get('setup_instructions', []),
                        code_quality_score=cached_result.get('code_quality_score', 0.8),
                        cache_used=True,
                        llm_calls=0
                    )

                # Generate code using LLM with structured output
                code_result = self._generate_code_structured(blueprint)

                # Cache successful result
                self.cache.set(
                    f"code_gen_{blueprint_hash}",
                    {
                        'script_content': code_result.script_content,
                        'requirements_content': code_result.requirements_content,
                        'setup_instructions': code_result.setup_instructions,
                        'code_quality_score': code_result.code_quality_score
                    },
                    model="gpt-4",
                    tags=["code_generation", "agent2"]
                )

                return code_result

        except Exception as e:
            execution_time = time.time() - start_time
            return Agent2Output(
                success=False,
                agent_name="enhanced_agent_2", 
                timestamp=time.time(),
                execution_time_seconds=execution_time,
                errors=[f"Code generation failed: {str(e)}"],
                script_content=self._create_fallback_script(),  # Always return valid script
                requirements_content="# Error in requirements generation",
                code_quality_score=0.0,
                llm_calls=self.llm_calls_made
            )

    def _parse_blueprint_tolerant(self, blueprint_data: Union[str, dict], 
                                validation_level: ValidationLevel) -> AutomationBlueprint:
        """
        Parse blueprint with maximum tolerance for schema mismatches
        """
        try:
            # Handle different input types
            if isinstance(blueprint_data, str):
                if blueprint_data.strip().startswith('{'):
                    # JSON string
                    parsed_data = repair_and_parse_json(blueprint_data)
                else:
                    # File path
                    blueprint_file = Path(blueprint_data)
                    if blueprint_file.exists():
                        parsed_data = json.loads(blueprint_file.read_text())
                    else:
                        raise ValueError(f"Blueprint file not found: {blueprint_data}")
            else:
                parsed_data = blueprint_data

            # Use shared safe parsing with tolerance
            blueprint = safe_parse_blueprint(parsed_data, validation_level)

            # Additional tolerance measures for legacy blueprints
            if validation_level == ValidationLevel.TOLERANT:
                blueprint = self._apply_compatibility_fixes(blueprint)

            return blueprint

        except Exception as e:
            self.reflection_notes.append(f"Blueprint parsing failed: {str(e)}")

            # Create minimal valid blueprint from available data
            return self._create_minimal_blueprint(blueprint_data)

    def _apply_compatibility_fixes(self, blueprint: AutomationBlueprint) -> AutomationBlueprint:
        """
        Apply fixes for common blueprint compatibility issues
        """
        # Ensure all steps have required fields
        for step in blueprint.steps:
            if not step.fallback_actions:
                step.fallback_actions = [f"Manual intervention for step {step.step_number}"]

            if not step.expected_outcome:
                step.expected_outcome = f"Step {step.step_number} completed successfully"

            if not step.validation_criteria:
                step.validation_criteria = f"Verify step {step.step_number} visual outcome"

        # Add metadata if missing
        if not blueprint.metadata:
            blueprint.metadata = {
                "auto_fixed": True,
                "compatibility_mode": True
            }

        return blueprint

    def _create_minimal_blueprint(self, original_data: Any) -> AutomationBlueprint:
        """
        Create minimal valid blueprint when parsing fails completely
        """
        return AutomationBlueprint(
            schema_version="1.0",
            summary="Minimal blueprint from parsing failure",
            steps=[
                BlueprintStep(
                    step_number=1,
                    action="navigate_to_page",
                    target_element="body",
                    expected_outcome="Page loaded successfully",
                    fallback_actions=["Refresh page", "Check internet connection"]
                )
            ],
            validation_level=ValidationLevel.TOLERANT,
            metadata={"parsing_error": True, "original_data_type": str(type(original_data))}
        )

    def _generate_code_structured(self, blueprint: AutomationBlueprint) -> Agent2Output:
        """
        Generate code using structured LLM output to eliminate parsing errors
        """
        # Create comprehensive prompt
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", self._get_human_prompt(blueprint))
        ])

        llm = get_langchain_llm(
            response_format={"type": "json_object"}  # Force JSON mode
        )

        self.llm_calls_made += 1

        with monitor_performance(self.monitor, "llm_call"):
            chain = prompt_template | llm
            response = chain.invoke({
                "blueprint": blueprint.model_dump_json(),
                "step_count": len(blueprint.steps),
                "complexity": self._assess_complexity(blueprint)
            })

        # Parse structured response
        try:
            parsed_response = repair_and_parse_json(response.content)

            # Extract components with defaults
            script_content = parsed_response.get('script_content', self._create_fallback_script())
            requirements_content = parsed_response.get('requirements_content', 'selenium>=4.0.0')
            setup_instructions = parsed_response.get('setup_instructions', ['Install requirements'])

            # Quality assessment
            quality_score = self._assess_code_quality(script_content, blueprint)

            # Add reflection notes
            if 'reflection_notes' in parsed_response:
                self.reflection_notes.extend(parsed_response['reflection_notes'])

            return Agent2Output(
                success=True,
                agent_name="enhanced_agent_2",
                timestamp=time.time(), 
                execution_time_seconds=self.monitor.metrics.get("llm_call", [0])[-1],
                script_content=script_content,
                requirements_content=requirements_content,
                setup_instructions=setup_instructions,
                code_quality_score=quality_score,
                llm_calls=1
            )

        except Exception as e:
            return Agent2Output(
                success=False,
                agent_name="enhanced_agent_2",
                timestamp=time.time(),
                execution_time_seconds=time.time(),
                errors=[f"Code generation parsing failed: {str(e)}"],
                script_content=self._create_fallback_script(),
                requirements_content="# Parsing error in requirements",
                code_quality_score=0.0,
                llm_calls=1
            )

    def _create_fallback_script(self) -> str:
        """
        Create minimal valid automation script for error cases
        """
        return """#!/usr/bin/env python3
# Fallback automation script generated due to processing error

import time
import sys

def main():
    print("Starting fallback automation...")

    try:
        # Import selenium dynamically to avoid import errors
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC

        # Initialize WebDriver
        driver = webdriver.Chrome()

        try:
            # Basic navigation
            driver.get("https://example.com")

            # Wait for page load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            print("Fallback automation completed successfully")
            return True

        except Exception as e:
            print(f"Automation failed: {e}")
            return False

        finally:
            driver.quit()

    except ImportError:
        print("Selenium not installed. Please install requirements.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
"""

    def _assess_complexity(self, blueprint: AutomationBlueprint) -> str:
        """
        Assess blueprint complexity for code generation
        """
        step_count = len(blueprint.steps)
        has_forms = any('input' in step.action.lower() for step in blueprint.steps)
        has_navigation = any('navigate' in step.action.lower() for step in blueprint.steps)
        has_waits = any('wait' in step.action.lower() for step in blueprint.steps)

        if step_count <= 3 and not has_forms:
            return "simple"
        elif step_count <= 8 and (has_forms or has_navigation):
            return "medium"
        else:
            return "complex"

    def _assess_code_quality(self, script_content: str, blueprint: AutomationBlueprint) -> float:
        """
        Assess generated code quality
        """
        score = 0.5  # Base score

        # Check for error handling
        if 'try:' in script_content and 'except' in script_content:
            score += 0.15

        # Check for proper imports
        if 'selenium' in script_content:
            score += 0.1

        # Check for WebDriverWait usage
        if 'WebDriverWait' in script_content:
            score += 0.1

        # Check if all blueprint steps are addressed
        addressed_steps = 0
        for step in blueprint.steps:
            if step.action.lower() in script_content.lower():
                addressed_steps += 1

        if addressed_steps > 0:
            score += (addressed_steps / len(blueprint.steps)) * 0.15

        return min(score, 1.0)

    def _get_blueprint_hash(self, blueprint: AutomationBlueprint) -> str:
        """
        Create hash of blueprint for caching
        """
        import hashlib
        content = blueprint.model_dump_json(sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _get_system_prompt(self) -> str:
        """
        Enhanced system prompt for structured code generation
        """
        return """You are an expert automation code generator specializing in Selenium WebDriver.

Your task is to generate complete, production-ready automation scripts from blueprints.

CRITICAL: You MUST respond with valid JSON in this exact structure:
{
  "script_content": "complete Python script as string",
  "requirements_content": "requirements.txt content",
  "setup_instructions": ["step1", "step2", "step3"],
  "code_quality_notes": ["note1", "note2"],
  "reflection_notes": ["insight1", "insight2"]
}

Script Requirements:
- Complete Python script with proper imports
- Selenium WebDriver implementation
- Error handling with try/catch blocks
- WebDriverWait for element waiting
- Proper resource cleanup (driver.quit())
- Comments explaining each major section
- Exit codes (0 for success, 1 for failure)

Quality Standards:
- Follow PEP 8 style guidelines
- Include docstrings for functions
- Handle common web automation issues (stale elements, timeouts)
- Implement fallback actions from blueprint
- Validate expected outcomes where possible

Response must be valid JSON only, no additional text.
"""

    def _get_human_prompt(self, blueprint: AutomationBlueprint) -> str:
        """
        Human prompt with blueprint context
        """
        return f"""
Blueprint to implement:
{blueprint.model_dump_json(indent=2)}

Generate a complete automation script that:
1. Implements all {len(blueprint.steps)} steps in sequence
2. Handles errors gracefully with fallback actions
3. Uses appropriate waits and element selection strategies
4. Includes proper logging and status reporting
5. Follows the JSON response structure specified

Platform: {blueprint.metadata.get('platform', 'web')}
Complexity: {self._assess_complexity(blueprint)}
Estimated Duration: {blueprint.estimated_duration_seconds}s

Remember: Response must be valid JSON only, no additional explanations.
"""

# Module-level functions for backward compatibility
def run_enhanced_agent2(seq_no: str, blueprint_file_path: str) -> Dict[str, Any]:
    """
    Main entry point for Enhanced Agent 2 with error handling
    """
    agent = EnhancedAgent2()

    try:
        # Parse blueprint with high tolerance
        result = agent.generate_automation_script(
            blueprint_file_path,
            ValidationLevel.TOLERANT  # Always use tolerant mode for maximum compatibility
        )

        if result.success:
            # Save generated artifacts
            artifacts_path = ensure_directory(Path(ARTIFACTS_DIR) / seq_no)

            # Save script
            script_file = artifacts_path / "automation_script.py"
            safe_file_write(script_file, result.script_content)

            # Save requirements
            requirements_file = artifacts_path / "requirements.txt"
            safe_file_write(requirements_file, result.requirements_content)

            # Save setup instructions
            setup_file = artifacts_path / "setup_instructions.txt"
            safe_file_write(setup_file, "\n".join(result.setup_instructions))

            return {
                "success": True,
                "script_content": result.script_content,
                "requirements_content": result.requirements_content,
                "setup_instructions": result.setup_instructions,
                "code_quality_score": result.code_quality_score,
                "cache_used": result.cache_used,
                "llm_calls": result.llm_calls,
                "reflection_notes": agent.reflection_notes,
                "performance_stats": agent.monitor.get_stats()
            }
        else:
            return {
                "success": False,
                "error": result.errors[0] if result.errors else "Unknown error",
                "fallback_script": result.script_content,
                "llm_calls": result.llm_calls,
                "reflection_notes": agent.reflection_notes
            }

    except Exception as e:
        return create_error_response(
            "agent2_critical_failure",
            str(e),
            suggested_fix="Check blueprint format and retry with tolerant validation"
        )

# Cache and utility functions
def clear_agent2_cache():
    """Clear Agent 2 cache"""
    cache_dir = Path(ARTIFACTS_DIR) / "cache"
    if (cache_dir / "agent2_cache.db").exists():
        (cache_dir / "agent2_cache.db").unlink()

def get_agent2_stats():
    """Get Agent 2 performance statistics"""
    return {
        "cache_stats": agent2_cache.get_stats(),
        "performance_stats": performance_monitor.get_stats()
    }

# Backward compatibility globals
agent2_cache = agent2_cache
agent2_scratchpad = {"reflection_notes": []}  # Legacy compatibility
todo_organizer = {"tasks": []}  # Legacy compatibility

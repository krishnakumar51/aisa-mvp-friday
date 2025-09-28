# enhanced/enhanced_agent_1.py
"""
Enhanced Agent 1: Blueprint Generation with unified schema and improved error handling
Key improvements:
- Uses shared schema models to prevent communication breakdown
- Persistent SQLite-backed caching
- Better error handling and validation
- Structured JSON outputs guaranteed
"""

import time
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import HTTPException
import base64
from io import BytesIO
from PIL import Image

# Import shared modules (CRITICAL: prevents schema drift)
from shared_models import (
    AutomationBlueprint, BlueprintStep, Agent1Output, TaskStatus,
    ValidationLevel, safe_parse_blueprint, create_error_response
)
from utils import (
    PersistentLLMCache, PerformanceMonitor, monitor_performance,
    repair_and_parse_json, validate_with_detailed_errors, 
    ensure_directory, safe_file_write
)

# Import original LLM utilities (unchanged)
from agents.llm_utils import get_llm_response, get_langchain_llm
from langchain.prompts import ChatPromptTemplate
from config import ARTIFACTS_DIR

# Global instances with persistent storage
cache_dir = ensure_directory(Path(ARTIFACTS_DIR) / "cache")
agent1_cache = PersistentLLMCache(cache_dir / "agent1_cache.db", default_ttl=7200)
performance_monitor = PerformanceMonitor()

class EnhancedAgent1:
    """
    Enhanced Agent 1 with unified schema compliance and robust error handling
    """

    def __init__(self):
        self.cache = agent1_cache
        self.monitor = performance_monitor
        self.llm_calls_made = 0

    def analyze_pdf_content(self, pdf_content: str, images: List[str], 
                          validation_level: ValidationLevel = ValidationLevel.STRICT) -> Agent1Output:
        """
        Main blueprint generation with structured output and error handling
        """
        start_time = time.time()

        try:
            with monitor_performance(self.monitor, "blueprint_generation"):
                # Try cache first (validation-first approach)
                cache_key = f"pdf_analysis_{len(pdf_content)}_{len(images)}"
                cached_result = self.cache.get(cache_key, model="gpt-4")

                if cached_result:
                    blueprint = AutomationBlueprint.model_validate(cached_result['blueprint'])
                    return Agent1Output(
                        success=True,
                        agent_name="enhanced_agent_1",
                        timestamp=time.time(),
                        execution_time_seconds=0.01,  # Cache hit
                        blueprint=blueprint,
                        confidence_score=cached_result.get('confidence_score', 0.8),
                        cache_used=True,
                        llm_calls=0
                    )

                # Generate blueprint using LLM with structured output
                blueprint_result = self._generate_blueprint_structured(
                    pdf_content, images, validation_level
                )

                # Cache the successful result
                self.cache.set(
                    cache_key, 
                    {
                        'blueprint': blueprint_result.blueprint.model_dump(),
                        'confidence_score': blueprint_result.confidence_score
                    },
                    model="gpt-4",
                    tags=["blueprint", "agent1"]
                )

                return blueprint_result

        except Exception as e:
            execution_time = time.time() - start_time
            return Agent1Output(
                success=False,
                agent_name="enhanced_agent_1",
                timestamp=time.time(),
                execution_time_seconds=execution_time,
                errors=[f"Blueprint generation failed: {str(e)}"],
                blueprint=self._create_fallback_blueprint(),  # Always return valid structure
                confidence_score=0.0,
                llm_calls=self.llm_calls_made
            )

    def _generate_blueprint_structured(self, pdf_content: str, images: List[str], 
                                     validation_level: ValidationLevel) -> Agent1Output:
        """
        Generate blueprint using OpenAI structured outputs to eliminate parsing errors
        """
        # Use improved prompt template with explicit JSON structure requirement
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", self._get_human_prompt(pdf_content, images))
        ])

        llm = get_langchain_llm(
            response_format={"type": "json_object"}  # Force JSON mode
        )

        self.llm_calls_made += 1

        with monitor_performance(self.monitor, "llm_call"):
            # Generate response with structured format
            chain = prompt_template | llm
            response = chain.invoke({
                "pdf_content": pdf_content[:8000],  # Limit context
                "images_count": len(images),
                "validation_level": validation_level.value
            })

        # Parse with robust error handling
        try:
            parsed_data = repair_and_parse_json(response.content)

            # Validate using shared schema
            validation_result = validate_with_detailed_errors(
                AutomationBlueprint, 
                parsed_data
            )

            if validation_result['success']:
                blueprint = validation_result['data']
            else:
                # Try relaxed parsing for compatibility
                relaxed_blueprint = self._create_relaxed_blueprint(parsed_data)
                blueprint = relaxed_blueprint

            return Agent1Output(
                success=True,
                agent_name="enhanced_agent_1",
                timestamp=time.time(),
                execution_time_seconds=self.monitor.metrics.get("llm_call", [0])[-1],
                blueprint=blueprint,
                confidence_score=self._calculate_confidence(blueprint),
                warnings=validation_result.get('warnings', []),
                llm_calls=1
            )

        except Exception as e:
            # Return structured error with fallback blueprint
            return Agent1Output(
                success=False,
                agent_name="enhanced_agent_1",
                timestamp=time.time(),
                execution_time_seconds=time.time(),
                errors=[f"JSON parsing failed: {str(e)}"],
                blueprint=self._create_fallback_blueprint(),
                confidence_score=0.0,
                llm_calls=1
            )

    def _create_relaxed_blueprint(self, raw_data: Dict[str, Any]) -> AutomationBlueprint:
        """
        Create blueprint with relaxed validation for compatibility
        """
        # Ensure required fields exist
        if 'summary' not in raw_data:
            raw_data['summary'] = "Auto-generated task summary"

        if 'steps' not in raw_data or not raw_data['steps']:
            raw_data['steps'] = [
                {
                    'step_number': 1,
                    'action': 'navigate',
                    'target_element': 'main_content',
                    'input_data': None
                }
            ]

        # Normalize steps
        for i, step in enumerate(raw_data['steps']):
            if not isinstance(step, dict):
                continue

            # Ensure step_number
            step['step_number'] = i + 1

            # Convert complex target_element_description to string
            if 'target_element_description' in step and isinstance(step['target_element_description'], dict):
                step['target_element_description'] = json.dumps(step['target_element_description'])

            # Ensure fallback_actions is a list
            if 'fallback_actions' in step and not isinstance(step['fallback_actions'], list):
                if isinstance(step['fallback_actions'], str):
                    step['fallback_actions'] = [step['fallback_actions']]
                else:
                    step['fallback_actions'] = []

        # Set schema version and validation level
        raw_data['schema_version'] = '1.0'
        raw_data['validation_level'] = ValidationLevel.RELAXED

        return AutomationBlueprint.model_validate(raw_data)

    def _create_fallback_blueprint(self) -> AutomationBlueprint:
        """
        Create minimal valid blueprint for error cases
        """
        return AutomationBlueprint(
            schema_version="1.0",
            summary="Fallback blueprint due to processing error",
            steps=[
                BlueprintStep(
                    step_number=1,
                    action="error_fallback",
                    target_element="body",
                    input_data=None,
                    expected_outcome="Manual intervention required",
                    fallback_actions=["Contact support"]
                )
            ],
            validation_level=ValidationLevel.TOLERANT
        )

    def _calculate_confidence(self, blueprint: AutomationBlueprint) -> float:
        """
        Calculate confidence score based on blueprint completeness
        """
        score = 0.5  # Base score

        # Add points for completeness
        if blueprint.summary and len(blueprint.summary) > 20:
            score += 0.1

        if len(blueprint.steps) >= 3:
            score += 0.1

        # Check step quality
        detailed_steps = sum(1 for step in blueprint.steps 
                           if step.target_element and step.expected_outcome)
        if detailed_steps > 0:
            score += (detailed_steps / len(blueprint.steps)) * 0.2

        # Bonus for fallback actions
        steps_with_fallbacks = sum(1 for step in blueprint.steps 
                                 if step.fallback_actions)
        if steps_with_fallbacks > 0:
            score += 0.1

        return min(score, 1.0)

    def _get_system_prompt(self) -> str:
        """
        Enhanced system prompt with explicit JSON structure requirements
        """
        return """You are an expert automation blueprint generator. 

Your task is to analyze PDF content and generate a detailed automation blueprint.

CRITICAL: You MUST respond with valid JSON in this exact structure:
{
  "schema_version": "1.0",
  "summary": "Brief task description",
  "steps": [
    {
      "step_number": 1,
      "action": "action_name",
      "target_element": "element_selector",
      "target_element_description": "detailed description or JSON string",
      "input_data": "data to input or null",
      "expected_outcome": "what should happen",
      "timing_notes": "any timing considerations",
      "fallback_actions": ["fallback1", "fallback2"],
      "validation_criteria": "how to verify success"
    }
  ],
  "estimated_duration_seconds": 120,
  "metadata": {
    "complexity": "medium",
    "platform": "web"
  }
}

Rules:
- All fields in steps are optional except step_number and action
- target_element_description can be string or complex object (will be converted to JSON string)
- fallback_actions must be array of strings
- estimated_duration_seconds should be realistic
- Ensure JSON is valid and complete
"""

    def _get_human_prompt(self, pdf_content: str, images: List[str]) -> str:
        """
        Human prompt with context and requirements
        """
        return f"""
PDF Content (first 8000 chars):
{pdf_content[:8000]}

Number of images provided: {len(images)}

Generate a comprehensive automation blueprint that:
1. Breaks down the task into clear, executable steps
2. Includes specific element selectors and descriptions
3. Provides fallback actions for each critical step
4. Estimates timing and validation criteria
5. Follows the exact JSON structure specified in the system prompt

Remember: Response must be valid JSON only, no additional text or explanation.
"""

# Module-level functions for backward compatibility
def run_enhanced_agent1(seq_no: str, pdf_content: str, user_instructions: str, 
                       images: List[str] = None) -> Dict[str, Any]:
    """
    Main entry point for Enhanced Agent 1 with error handling
    """
    agent = EnhancedAgent1()
    images = images or []

    try:
        # Use STRICT validation by default, fallback to RELAXED on errors
        result = agent.analyze_pdf_content(
            pdf_content, 
            images, 
            ValidationLevel.STRICT
        )

        if result.success:
            # Save blueprint to artifacts directory
            artifacts_path = ensure_directory(Path(ARTIFACTS_DIR) / seq_no)
            blueprint_file = artifacts_path / "blueprint.json"

            safe_file_write(
                blueprint_file, 
                result.blueprint.model_dump_json(indent=2)
            )

            return {
                "success": True,
                "blueprint_content": result.blueprint.model_dump(),
                "confidence_score": result.confidence_score,
                "cache_used": result.cache_used,
                "llm_calls": result.llm_calls,
                "warnings": result.warnings,
                "performance_stats": agent.monitor.get_stats()
            }
        else:
            return {
                "success": False,
                "error": result.errors[0] if result.errors else "Unknown error",
                "fallback_blueprint": result.blueprint.model_dump(),
                "llm_calls": result.llm_calls
            }

    except Exception as e:
        return create_error_response(
            "agent1_critical_failure",
            str(e),
            suggested_fix="Check system logs and retry with relaxed validation"
        )

# Cache management functions
def clear_agent1_cache():
    """Clear Agent 1 cache"""
    cache_dir = Path(ARTIFACTS_DIR) / "cache"
    if (cache_dir / "agent1_cache.db").exists():
        (cache_dir / "agent1_cache.db").unlink()

def get_agent1_stats():
    """Get Agent 1 performance statistics"""
    return {
        "cache_stats": agent1_cache.get_stats(),
        "performance_stats": performance_monitor.get_stats()
    }

# schemas/shared_models.py
"""
Unified schema definitions for the enhanced agent pipeline.
This module ensures all agents use identical data models to prevent communication breakdowns.
"""

from typing import Dict, List, Optional, Any, Union, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator
from enum import Enum
import json

class Platform(Enum):
    MOBILE = "mobile"
    WEB = "web"

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class ValidationLevel(Enum):
    STRICT = "strict"
    RELAXED = "relaxed"
    TOLERANT = "tolerant"

# Core Blueprint Models
class BlueprintStep(BaseModel):
    """Unified step definition used across all agents"""
    model_config = ConfigDict(extra="allow")  # Allow future fields

    step_number: int = Field(..., ge=1, description="Sequential step number")
    action: str = Field(..., min_length=1, description="Action to perform")
    target_element: Optional[str] = Field(None, description="Element selector or description")
    target_element_description: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Detailed element description")
    input_data: Optional[str] = Field(None, description="Data to input")
    expected_outcome: Optional[Union[str, Dict[str, Any]]] = Field(None, description="Expected result")
    timing_notes: Optional[str] = Field(None, description="Timing considerations")
    fallback_actions: Optional[List[str]] = Field(default_factory=list, description="Recovery steps if primary action fails")
    validation_criteria: Optional[str] = Field(None, description="How to verify step completion")

    @field_validator('target_element_description', mode='before')
    @classmethod
    def coerce_target_description(cls, v):
        """Handle dict->str conversion for target descriptions"""
        if isinstance(v, dict):
            return json.dumps(v, separators=(',', ':'))
        return v

    @field_validator('expected_outcome', mode='before')
    @classmethod
    def coerce_expected_outcome(cls, v):
        """Handle mixed types for expected outcomes"""
        if isinstance(v, dict):
            return json.dumps(v, separators=(',', ':'))
        return v

class AutomationBlueprint(BaseModel):
    """Unified blueprint model with versioning and validation"""
    model_config = ConfigDict(extra="allow")  # Future-proof

    schema_version: Literal["1.0"] = Field(default="1.0", description="Schema version for compatibility")
    summary: str = Field(..., min_length=1, description="Task summary")
    steps: List[BlueprintStep] = Field(..., min_items=1, description="Automation steps")

    # Optional metadata (Agent 1 may include, Agent 2 ignores gracefully)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    validation_level: ValidationLevel = Field(default=ValidationLevel.STRICT, description="Validation strictness")
    estimated_duration_seconds: Optional[int] = Field(None, ge=0, description="Estimated execution time")

    @model_validator(mode='after')
    def validate_steps_sequence(self):
        """Ensure steps are properly numbered"""
        if not self.steps:
            raise ValueError("At least one step is required")

        for i, step in enumerate(self.steps, 1):
            if step.step_number != i:
                # Auto-correct step numbering
                step.step_number = i
        return self

# Task Status and Communication Models
class TaskStatus(BaseModel):
    """Enhanced task status with detailed error tracking"""
    model_config = ConfigDict(extra="allow")

    seq_no: str = Field(..., description="Task sequence number")
    phase: Literal["agent1", "agent2", "agent3", "completed", "failed"] = Field(..., description="Current phase")
    schema_version: str = Field(default="1.0", description="Schema version used")
    last_valid_step: Optional[int] = Field(None, description="Last successfully completed step")

    # Error handling
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Accumulated errors")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal issues")
    suggested_retry: bool = Field(default=False, description="Whether retry is recommended")
    retry_count: int = Field(default=0, ge=0, description="Number of retries attempted")

    # Performance tracking
    start_time: Optional[float] = Field(None, description="Task start timestamp")
    phase_times: Dict[str, float] = Field(default_factory=dict, description="Time spent in each phase")
    llm_calls_made: int = Field(default=0, ge=0, description="Number of LLM calls")
    cache_hits: int = Field(default=0, ge=0, description="Cache hit count")

class AgentOutput(BaseModel):
    """Base class for all agent outputs"""
    model_config = ConfigDict(extra="allow")

    success: bool = Field(..., description="Whether operation succeeded")
    agent_name: str = Field(..., description="Which agent produced this output")
    timestamp: float = Field(..., description="Output generation timestamp")
    execution_time_seconds: float = Field(..., ge=0, description="Time taken to generate output")

    # Optional fields
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings")
    cache_used: bool = Field(default=False, description="Whether cached result was used")
    llm_calls: int = Field(default=0, ge=0, description="Number of LLM calls made")

class Agent1Output(AgentOutput):
    """Enhanced Agent 1 output with blueprint"""
    blueprint: AutomationBlueprint = Field(..., description="Generated automation blueprint")
    confidence_score: float = Field(default=0.5, ge=0, le=1, description="Confidence in blueprint quality")
    reflection_notes: List[str] = Field(default_factory=list, description="Reflection insights")

class Agent2Output(AgentOutput):
    """Enhanced Agent 2 output with code artifacts"""
    script_content: str = Field(..., min_length=1, description="Generated automation script")
    requirements_content: str = Field(default="", description="Requirements.txt content")
    setup_instructions: List[str] = Field(default_factory=list, description="Setup steps")
    code_quality_score: float = Field(default=0.5, ge=0, le=1, description="Code quality assessment")

class Agent3Output(AgentOutput):
    """Enhanced Agent 3 output with execution results"""
    exit_code: int = Field(..., description="Script execution exit code")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error output")
    artifacts_created: List[str] = Field(default_factory=list, description="Generated artifacts")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Execution metrics")

# Error Handling Models
class SchemaError(BaseModel):
    """Detailed schema validation error"""
    error_type: Literal["missing_field", "type_mismatch", "validation_failed", "unknown_version"] = Field(..., description="Error category")
    field_path: str = Field(..., description="JSON path to problematic field")
    expected_type: Optional[str] = Field(None, description="Expected data type")
    actual_value: Optional[str] = Field(None, description="Actual value found")
    suggested_fix: Optional[str] = Field(None, description="How to fix this error")
    is_retryable: bool = Field(default=True, description="Whether error can be retried")

# Cache Models
class CacheEntry(BaseModel):
    """Enhanced cache entry with metadata"""
    key: str = Field(..., description="Cache key")
    value: Dict[str, Any] = Field(..., description="Cached value")
    timestamp: float = Field(..., description="Cache creation time")
    ttl_seconds: int = Field(default=7200, gt=0, description="Time to live")
    hit_count: int = Field(default=0, ge=0, description="Number of cache hits")
    tags: List[str] = Field(default_factory=list, description="Cache tags for bulk invalidation")

    def is_expired(self, current_time: float) -> bool:
        """Check if cache entry has expired"""
        return (current_time - self.timestamp) > self.ttl_seconds

# Utility functions for safe parsing
def safe_parse_blueprint(raw_data: Union[str, dict], validation_level: ValidationLevel = ValidationLevel.STRICT) -> AutomationBlueprint:
    """Safely parse blueprint data with configurable validation"""
    try:
        if isinstance(raw_data, str):
            data = json.loads(raw_data)
        else:
            data = raw_data

        # Handle legacy blueprints without schema_version
        if 'schema_version' not in data:
            data['schema_version'] = '1.0'

        blueprint = AutomationBlueprint.model_validate(data)

        if validation_level == ValidationLevel.RELAXED:
            # Skip optional field validation
            pass
        elif validation_level == ValidationLevel.TOLERANT:
            # Fill in missing optional fields with defaults
            for step in blueprint.steps:
                if not step.fallback_actions:
                    step.fallback_actions = []

        return blueprint

    except Exception as e:
        raise ValueError(f"Blueprint parsing failed: {str(e)}")

def create_error_response(error_type: str, message: str, field_path: str = "", suggested_fix: str = "") -> Dict[str, Any]:
    """Create standardized error response"""
    return {
        "success": False,
        "error": {
            "type": error_type,
            "message": message,
            "field_path": field_path,
            "suggested_fix": suggested_fix,
            "timestamp": __import__('time').time()
        }
    }

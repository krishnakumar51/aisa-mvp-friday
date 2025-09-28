# agents/enhanced_agent_3.py

from pathlib import Path
import sys
import os
import subprocess
import json
import time
import shutil
import re
from typing import Dict, List, Optional, Any
from fastapi import HTTPException
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field
import logging
import hashlib
from config import ARTIFACTS_DIR
from agents.llm_utils import get_llm_response, get_langchain_llm, LLMProvider

# ENHANCED: Tavily Search Integration (for intelligent analysis)
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    print("⚠️ Tavily not available - using fallback analysis")
    TAVILY_AVAILABLE = False

class Platform(Enum):
    MOBILE = "mobile"
    WEB = "web"

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class StepStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    PENDING = "pending"

@dataclass
class ExecutionConfig:
    platform: Platform
    use_enhanced_agent: bool = True
    enable_monitoring: bool = True
    enable_reporting: bool = True
    timeout_seconds: int = 300
    retry_attempts: int = 1

# ENHANCED: Pydantic models for evaluation (Problem 4 solution)
class StepEvaluation(BaseModel):
    """Evaluation model for individual automation steps"""
    step_no: int = Field(ge=1, description="Step number starting from 1")
    step_name: str = Field(min_length=1, max_length=200)
    status: StepStatus
    review: Optional[str] = Field(default=None, max_length=1000)
    logs_path: str = Field(..., description="Path to step log file")
    screenshot_start: Optional[str] = None
    screenshot_final: Optional[str] = None
    duration_seconds: float = Field(ge=0, le=300)
    error_details: Optional[str] = Field(default=None, max_length=2000)
    retry_count: int = Field(default=0, ge=0, le=10)
    confidence_score: float = Field(ge=0.0, le=1.0)
    evaluation_method: str = Field(..., description="Method used for evaluation")

    class Config:
        use_enum_values = True

class EvaluationReport(BaseModel):
    """Complete evaluation report for automation run"""
    execution_metadata: Dict[str, Any]
    step_evaluations: List[StepEvaluation] = Field(default_factory=list)
    overall_assessment: Dict[str, Any]
    
    class Config:
        use_enum_values = True
        validate_assignment = True

# CRITICAL: LLM-Powered Intelligent Agent 3 with REAL THINKING capabilities
class LLMIntelligentAgent3:
    """
    LLM-POWERED REAL Intelligent Agent that uses LLM to THINK, ANALYZE, and provide insights.
    
    This agent uses LLM throughout for:
    - Intelligent log analysis and pattern recognition
    - Screenshot comparison with visual understanding
    - Code analysis with context awareness
    - Step evaluation with reasoning
    - Recommendation generation with expert knowledge
    - Real-time decision making and adaptation
    """

    def __init__(self, seq_no: str):
        self.seq_no = seq_no
        self.artifacts_dir = ARTIFACTS_DIR / seq_no
        self.agent3_dir = self.artifacts_dir / "enhanced_agent3"
        self.evaluation_file = self.agent3_dir / "evaluate.json"
        self.screenshots_dir = self.agent3_dir / "screenshots"
        self.logs_dir = self.agent3_dir / "logs"
        
        # LLM-powered intelligence components
        self.log_watcher = LLMLogAnalyzer(self.logs_dir, seq_no)
        self.screenshot_analyzer = LLMScreenshotAnalyzer(self.screenshots_dir, seq_no)
        self.code_analyzer = LLMCodeAnalyzer(self.artifacts_dir, seq_no)
        self.tavily_researcher = LLMTavilyResearcher(seq_no) if TAVILY_AVAILABLE else None
        
        # LLM instance for real-time thinking
        self.llm = get_langchain_llm(model_name=LLMProvider.GROQ)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def watch_and_evaluate_realtime_with_llm(self, blueprint_data: Dict = None) -> EvaluationReport:
        """
        REAL-TIME LLM-powered monitoring and evaluation with INTELLIGENT THINKING.
        
        This method continuously uses LLM to:
        1. THINK about what's happening in logs
        2. ANALYZE screenshots with visual understanding  
        3. REASON about code execution patterns
        4. EVALUATE step success with expert judgment
        5. GENERATE intelligent recommendations
        """
        print(f"[{self.seq_no}] 🧠 Starting LLM-POWERED REAL Intelligent Monitoring...")
        print(f"[{self.seq_no}] 🤔 LLM will THINK and ANALYZE everything in real-time...")
        print(f"[{self.seq_no}] 👁️ Watching: logs, screenshots, code execution WITH LLM INTELLIGENCE")
        
        # Initialize evaluation report with LLM analysis
        evaluation_report = self._initialize_llm_evaluation_report(blueprint_data)
        
        # Start LLM-powered monitoring
        monitoring_active = True
        step_evaluations = []
        monitoring_cycle = 0
        
        while monitoring_active:
            try:
                monitoring_cycle += 1
                print(f"[{self.seq_no}] 🔄 LLM Monitoring Cycle {monitoring_cycle}")
                
                # 1. LLM-POWERED LOG ANALYSIS - Real intelligent thinking
                recent_logs = self.log_watcher.get_recent_logs()
                if recent_logs:
                    print(f"[{self.seq_no}] 🤔 LLM analyzing {len(recent_logs)} new log entries...")
                    log_analysis = self.log_watcher.analyze_logs_with_llm_intelligence(recent_logs)
                    print(f"[{self.seq_no}] 📋 LLM Log Analysis: {log_analysis['llm_assessment']}")
                    
                    # Store LLM analysis results for comprehensive reporting
                    if not hasattr(self, 'llm_log_analyses'):
                        self.llm_log_analyses = []
                    self.llm_log_analyses.append(log_analysis)
                
                # 2. LLM-POWERED SCREENSHOT ANALYSIS - Visual intelligence
                screenshot_pairs = self.screenshot_analyzer.detect_new_screenshot_pairs()
                for pair in screenshot_pairs:
                    print(f"[{self.seq_no}] 👁️ LLM analyzing screenshot pair: {pair.get('step_name', 'unknown')}")
                    screenshot_analysis = self.screenshot_analyzer.analyze_screenshots_with_llm_intelligence(pair)
                    print(f"[{self.seq_no}] 🖼️ LLM Screenshot Analysis: {screenshot_analysis['llm_conclusion']}")
                    
                    # Store LLM screenshot analyses
                    if not hasattr(self, 'llm_screenshot_analyses'):
                        self.llm_screenshot_analyses = []
                    self.llm_screenshot_analyses.append(screenshot_analysis)
                
                # 3. LLM-POWERED CODE EXECUTION ANALYSIS - Context-aware thinking
                execution_status = self.code_analyzer.detect_execution_patterns_with_llm()
                if execution_status['new_patterns'] or execution_status['llm_should_analyze']:
                    print(f"[{self.seq_no}] 💻 LLM analyzing code execution patterns...")
                    code_analysis = self.code_analyzer.analyze_execution_with_llm_intelligence(execution_status)
                    print(f"[{self.seq_no}] 🧠 LLM Code Analysis: {code_analysis['llm_assessment']}")
                    
                    # Store LLM code analyses
                    if not hasattr(self, 'llm_code_analyses'):
                        self.llm_code_analyses = []
                    self.llm_code_analyses.append(code_analysis)
                
                # 4. LLM-POWERED STEP EVALUATION - Expert-level judgment
                completed_steps = self._detect_completed_steps_with_llm(recent_logs, screenshot_pairs)
                for step_data in completed_steps:
                    print(f"[{self.seq_no}] 🎯 LLM evaluating completed step {step_data['step_no']}...")
                    step_evaluation = self._evaluate_step_with_llm_intelligence(step_data)
                    step_evaluations.append(step_evaluation)
                    
                    print(f"[{self.seq_no}] ✅ Step {step_evaluation.step_no}: {step_evaluation.status} (Confidence: {step_evaluation.confidence_score:.2f})")
                    
                    # If step failed, use LLM + Tavily for intelligent recommendations
                    if step_evaluation.status == StepStatus.FAILURE:
                        print(f"[{self.seq_no}] 🔧 LLM generating intelligent recommendations for failed step...")
                        recommendations = self._generate_llm_intelligent_recommendations(step_evaluation, step_data)
                        step_evaluation.review = recommendations
                        print(f"[{self.seq_no}] 💡 LLM Recommendations generated for failed step")
                
                # 5. LLM-POWERED COMPLETION CHECK - Intelligent decision making
                if self._llm_check_execution_completed(recent_logs, step_evaluations):
                    monitoring_active = False
                    print(f"[{self.seq_no}] 🏁 LLM determined execution completed - finalizing evaluation")
                
                # Wait before next monitoring cycle
                time.sleep(3)  # Slightly longer for LLM processing
                
            except KeyboardInterrupt:
                print(f"[{self.seq_no}] ⏹️ Monitoring stopped by user")
                monitoring_active = False
            except Exception as e:
                print(f"[{self.seq_no}] ⚠️ LLM Monitoring error: {e}")
                time.sleep(5)  # Continue monitoring despite errors

        # LLM-POWERED FINAL ASSESSMENT
        print(f"[{self.seq_no}] 🧠 LLM generating comprehensive final assessment...")
        evaluation_report.step_evaluations = step_evaluations
        evaluation_report.overall_assessment = self._generate_llm_overall_assessment(step_evaluations)
        
        # Save comprehensive LLM-powered evaluation
        self._save_llm_evaluation_report(evaluation_report)
        
        print(f"[{self.seq_no}] 🎉 LLM-POWERED REAL Intelligent Evaluation completed!")
        print(f"[{self.seq_no}] 📊 LLM evaluated {len(step_evaluations)} steps with expert intelligence")
        print(f"[{self.seq_no}] 📁 Comprehensive LLM report saved: {self.evaluation_file}")
        
        return evaluation_report

    def _detect_completed_steps_with_llm(self, recent_logs: List[str], screenshot_pairs: List[Dict]) -> List[Dict[str, Any]]:
        """Use LLM to intelligently detect completed automation steps"""
        if not recent_logs and not screenshot_pairs:
            return []
            
        log_content = "\n".join(recent_logs) if recent_logs else "No recent logs"
        screenshot_info = json.dumps([{k: v for k, v in pair.items() if k != 'binary_data'} 
                                    for pair in screenshot_pairs], indent=2)
        
        # LLM-powered step detection
        step_detection_prompt = f"""You are an expert automation analyst. Analyze the following execution data to identify completed automation steps.

LOG CONTENT:
{log_content}

SCREENSHOT PAIRS INFO:
{screenshot_info}

Your task:
1. Identify automation steps that have been completed based on logs and screenshots
2. Extract step numbers, names, and completion evidence
3. Be intelligent about inferring step completion even with incomplete data

Respond with JSON format:
{{
    "completed_steps": [
        {{
            "step_no": <number>,
            "step_name": "<descriptive_name>",
            "completion_evidence": "<why_you_think_it_completed>",
            "confidence": <0.0-1.0>
        }}
    ],
    "analysis_reasoning": "<your_thinking_process>"
}}"""

        try:
            llm_response = get_llm_response(step_detection_prompt, 
                "You are an expert automation analyst specializing in step detection and execution monitoring.", model_name=LLMProvider.GROQ)
            
            # Parse LLM response
            llm_data = json.loads(llm_response)
            completed_steps = []
            
            for step_info in llm_data.get("completed_steps", []):
                step_data = {
                    "step_no": step_info.get("step_no", 1),
                    "step_name": step_info.get("step_name", f"Step {step_info.get('step_no', 1)}"),
                    "logs_content": log_content,
                    "screenshots": self._find_step_screenshots(step_info.get("step_no", 1), screenshot_pairs),
                    "completion_detected": True,
                    "llm_evidence": step_info.get("completion_evidence", ""),
                    "llm_confidence": step_info.get("confidence", 0.5),
                    "llm_reasoning": llm_data.get("analysis_reasoning", "")
                }
                completed_steps.append(step_data)
                
            return completed_steps
            
        except Exception as e:
            print(f"[{self.seq_no}] ⚠️ LLM step detection failed: {e}, using fallback")
            return self._fallback_detect_completed_steps(recent_logs, screenshot_pairs)

    def _evaluate_step_with_llm_intelligence(self, step_data: Dict[str, Any]) -> StepEvaluation:
        """Use LLM to comprehensively evaluate step success with expert judgment"""
        step_no = step_data["step_no"]
        step_name = step_data["step_name"]
        logs_content = step_data.get("logs_content", "")
        screenshots = step_data.get("screenshots", {})
        llm_evidence = step_data.get("llm_evidence", "")
        
        # Prepare screenshot information for LLM
        screenshot_info = "No screenshots available"
        if screenshots.get("start") or screenshots.get("final"):
            screenshot_info = f"Screenshots available - Start: {bool(screenshots.get('start'))}, Final: {bool(screenshots.get('final'))}"
            
            # Add basic screenshot analysis
            if screenshots.get("start") and screenshots.get("final"):
                try:
                    start_path = Path(screenshots["start"])
                    final_path = Path(screenshots["final"]) 
                    if start_path.exists() and final_path.exists():
                        start_size = start_path.stat().st_size
                        final_size = final_path.stat().st_size
                        size_change = ((final_size - start_size) / start_size * 100) if start_size > 0 else 0
                        screenshot_info += f"\nScreenshot size change: {size_change:.1f}%"
                except:
                    pass

        # LLM-powered comprehensive step evaluation
        evaluation_prompt = f"""You are an expert automation engineer evaluating the success of an automation step. Use your expertise to provide a comprehensive assessment.

STEP INFORMATION:
- Step Number: {step_no}
- Step Name: {step_name}
- Completion Evidence: {llm_evidence}

LOG ANALYSIS:
{logs_content[:2000] if logs_content else "No logs available"}

SCREENSHOT ANALYSIS:
{screenshot_info}

Your task is to evaluate this step's success using expert judgment. Consider:

1. **Log Patterns**: Look for success indicators (✓, success, completed) vs error patterns (error, failed, timeout, exception)
2. **Screenshot Evidence**: Consider if visual changes suggest successful action
3. **Timing**: Reasonable execution time vs timeouts
4. **Context**: Does the step make sense in the automation flow?
5. **Error Recovery**: Any retry attempts or error handling?

Provide your expert evaluation in JSON format:
{{
    "success_assessment": "<SUCCESS|FAILURE|WARNING>",
    "confidence_score": <0.0-1.0>,
    "reasoning": "<detailed_explanation_of_your_analysis>",
    "success_factors": ["<factor1>", "<factor2>"],
    "failure_factors": ["<factor1>", "<factor2>"],
    "recommendations": ["<recommendation1>", "<recommendation2>"],
    "duration_assessment": "<NORMAL|TOO_FAST|TOO_SLOW>",
    "overall_health": "<EXCELLENT|GOOD|FAIR|POOR>"
}}"""

        try:
            llm_response = get_llm_response(evaluation_prompt,
                "You are a senior automation engineer with 10+ years of experience in test automation, debugging, and quality assurance.", model_name=LLMProvider.GROQ)
            
            # Parse LLM evaluation
            llm_eval = json.loads(llm_response)
            
            # Convert LLM assessment to StepStatus
            assessment = llm_eval.get("success_assessment", "WARNING").upper()
            if assessment == "SUCCESS":
                status = StepStatus.SUCCESS
                review = None
            elif assessment == "FAILURE":
                status = StepStatus.FAILURE
                review = "LLM Analysis: " + llm_eval.get("reasoning", "Step failed based on expert analysis")
            else:
                status = StepStatus.WARNING
                review = "LLM Analysis: " + llm_eval.get("reasoning", "Step partially successful - review needed")
            
            # Extract duration (simplified)
            duration = self._extract_step_duration_llm(logs_content, step_no)
            
            return StepEvaluation(
                step_no=step_no,
                step_name=step_name,
                status=status,
                review=review,
                logs_path=str(self.logs_dir / f"step_{step_no:03d}.log"),
                screenshot_start=screenshots.get("start"),
                screenshot_final=screenshots.get("final"),
                duration_seconds=duration,
                confidence_score=llm_eval.get("confidence_score", 0.5),
                evaluation_method="LLM_Expert_Analysis"
            )
            
        except Exception as e:
            print(f"[{self.seq_no}] ⚠️ LLM step evaluation failed: {e}, using fallback")
            return self._fallback_evaluate_step(step_data)

    def _generate_llm_intelligent_recommendations(self, step_evaluation: StepEvaluation, step_data: Dict[str, Any]) -> str:
        """Use LLM + Tavily to generate intelligent, actionable recommendations"""
        logs_content = step_data.get("logs_content", "")
        llm_evidence = step_data.get("llm_evidence", "")
        screenshots = step_data.get("screenshots", {})
        
        # Prepare context for LLM
        context = f"""
FAILED STEP DETAILS:
- Step: {step_evaluation.step_no} - {step_evaluation.step_name}
- Confidence: {step_evaluation.confidence_score}
- Evidence: {llm_evidence}

RECENT LOGS:
{logs_content[-1000:] if logs_content else "No logs available"}

SCREENSHOTS:
- Start: {bool(screenshots.get('start'))}
- Final: {bool(screenshots.get('final'))}
"""

        # First, use Tavily for research if available
        tavily_insights = []
        if self.tavily_researcher:
            print(f"[{self.seq_no}] 🔍 Using Tavily to research step failure solutions...")
            tavily_insights = self.tavily_researcher.research_step_failure_with_llm(
                step_evaluation.step_name, logs_content[-500:] if logs_content else ""
            )

        # Prepare Tavily context
        tavily_context = ""
        if tavily_insights:
            tavily_context = f"\nTAVILY RESEARCH RESULTS:\n" + "\n".join(tavily_insights[:3])

        # LLM-powered recommendation generation
        recommendation_prompt = f"""You are a senior automation engineer and debugging expert. A automation step has failed and you need to provide actionable recommendations.

{context}
{tavily_context}

Based on your expertise and the available information, provide specific, actionable recommendations to fix this failed step. Consider:

1. **Root Cause Analysis**: What likely caused the failure?
2. **Immediate Fixes**: Quick solutions to try first
3. **Robust Solutions**: More comprehensive fixes for reliability
4. **Prevention**: How to prevent similar failures
5. **Alternative Approaches**: Different ways to achieve the same goal

Provide your expert recommendations in clear, actionable format. Focus on practical solutions that a developer can implement immediately.

Format your response as specific bullet points, each with a clear action to take."""

        try:
            llm_recommendations = get_llm_response(recommendation_prompt,
                "You are a world-class automation engineer and debugging specialist with extensive experience in fixing automation failures.", model_name=LLMProvider.GROQ)
            
            # Combine LLM recommendations with Tavily insights
            combined_recommendations = []
            
            if llm_recommendations:
                combined_recommendations.append("🤖 LLM Expert Analysis:")
                combined_recommendations.append(llm_recommendations)
            
            if tavily_insights:
                combined_recommendations.append("\n🔍 Research-Based Solutions:")
                combined_recommendations.extend([f"• {insight}" for insight in tavily_insights[:3]])
            
            return "\n".join(combined_recommendations) if combined_recommendations else "Manual review needed - insufficient data for automated analysis"
            
        except Exception as e:
            print(f"[{self.seq_no}] ⚠️ LLM recommendation generation failed: {e}")
            return f"Step failed with confidence {step_evaluation.confidence_score:.2f}. Manual debugging required."

    def _llm_check_execution_completed(self, recent_logs: List[str], step_evaluations: List[StepEvaluation]) -> bool:
        """Use LLM to intelligently determine if automation execution has completed"""
        log_content = "\n".join(recent_logs[-20:]) if recent_logs else "No recent logs"  # Last 20 lines
        
        evaluation_summary = []
        for eval in step_evaluations[-5:]:  # Last 5 evaluations
            evaluation_summary.append(f"Step {eval.step_no}: {eval.status}")
        
        completion_check_prompt = f"""You are an automation execution monitor. Determine if this automation run has completed based on the evidence.

RECENT LOG CONTENT (last 20 lines):
{log_content}

RECENT STEP EVALUATIONS:
{chr(10).join(evaluation_summary) if evaluation_summary else "No step evaluations yet"}

Look for completion indicators such as:
- "automation complete", "execution finished", "all steps done"
- "cleanup complete", "teardown finished" 
- Success/failure summary statements
- Process termination messages
- Final status reports

Respond with JSON:
{{
    "execution_completed": <true|false>,
    "reasoning": "<why_you_think_execution_is_or_isnt_completed>",
    "confidence": <0.0-1.0>
}}"""

        try:
            llm_response = get_llm_response(completion_check_prompt,
                "You are an expert at monitoring automation execution and determining completion status.", model_name=LLMProvider.GROQ)
            
            llm_result = json.loads(llm_response)
            completed = llm_result.get("execution_completed", False)
            reasoning = llm_result.get("reasoning", "No reasoning provided")
            
            if completed:
                print(f"[{self.seq_no}] 🤔 LLM determined execution completed: {reasoning}")
            
            return completed
            
        except Exception as e:
            print(f"[{self.seq_no}] ⚠️ LLM completion check failed: {e}, using fallback")
            return self._fallback_check_completion(recent_logs)

    def _generate_llm_overall_assessment(self, step_evaluations: List[StepEvaluation]) -> Dict[str, Any]:
        """Use LLM to generate comprehensive overall assessment"""
        if not step_evaluations:
            return {
                "success_rate": 0.0,
                "critical_failures": 0,
                "total_steps": 0,
                "performance_rating": "no_data",
                "llm_assessment": "No steps were evaluated",
                "recommendations": ["No steps to analyze"]
            }

        # Prepare evaluation summary for LLM
        eval_summary = []
        for eval in step_evaluations:
            eval_summary.append({
                "step": eval.step_no,
                "status": eval.status,
                "confidence": eval.confidence_score,
                "duration": eval.duration_seconds
            })

        assessment_prompt = f"""You are a senior automation engineer reviewing the results of an automation run. Provide a comprehensive assessment.

STEP EVALUATION RESULTS:
{json.dumps(eval_summary, indent=2)}

SUMMARY STATISTICS:
- Total Steps: {len(step_evaluations)}
- Successful: {sum(1 for e in step_evaluations if e.status == StepStatus.SUCCESS)}
- Failed: {sum(1 for e in step_evaluations if e.status == StepStatus.FAILURE)}
- Warnings: {sum(1 for e in step_evaluations if e.status == StepStatus.WARNING)}

Provide your expert assessment in JSON format:
{{
    "overall_rating": "<EXCELLENT|GOOD|FAIR|POOR>",
    "success_rate_analysis": "<analysis_of_success_rate>",
    "key_strengths": ["<strength1>", "<strength2>"],
    "key_weaknesses": ["<weakness1>", "<weakness2>"],
    "critical_issues": ["<issue1>", "<issue2>"],
    "performance_analysis": "<analysis_of_timing_and_efficiency>",
    "reliability_assessment": "<assessment_of_overall_reliability>",
    "priority_improvements": ["<improvement1>", "<improvement2>"],
    "next_steps": ["<step1>", "<step2>"],
    "executive_summary": "<brief_summary_for_stakeholders>"
}}"""

        try:
            llm_response = get_llm_response(assessment_prompt,
                "You are a senior automation engineer with extensive experience in automation quality assessment and reporting.", model_name=LLMProvider.GROQ)
            
            llm_assessment = json.loads(llm_response)
            
            # Calculate basic metrics
            successful_steps = sum(1 for step in step_evaluations if step.status == StepStatus.SUCCESS)
            failed_steps = sum(1 for step in step_evaluations if step.status == StepStatus.FAILURE)
            warning_steps = sum(1 for step in step_evaluations if step.status == StepStatus.WARNING)
            success_rate = successful_steps / len(step_evaluations)
            
            return {
                "success_rate": success_rate,
                "critical_failures": failed_steps,
                "total_steps": len(step_evaluations),
                "successful_steps": successful_steps,
                "warning_steps": warning_steps,
                "failed_steps": failed_steps,
                "performance_rating": llm_assessment.get("overall_rating", "UNKNOWN").lower(),
                "llm_assessment": llm_assessment,
                "recommendations": (
                    llm_assessment.get("priority_improvements", []) + 
                    llm_assessment.get("next_steps", [])
                )[:10]  # Limit to 10 recommendations
            }
            
        except Exception as e:
            print(f"[{self.seq_no}] ⚠️ LLM overall assessment failed: {e}, using fallback")
            return self._fallback_overall_assessment(step_evaluations)

    # Helper methods for LLM operations
    def _initialize_llm_evaluation_report(self, blueprint_data: Dict = None) -> EvaluationReport:
        """Initialize evaluation report with LLM-powered metadata"""
        execution_metadata = {
            "seqno": self.seq_no,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "agent_version": "llm_intelligent_agent_3",
            "llm_capabilities": [
                "Real-time LLM log analysis with intelligent reasoning",
                "LLM-powered screenshot comparison with visual understanding",
                "LLM-based code execution pattern analysis",
                "Expert-level LLM step evaluation with confidence scoring",
                "LLM + Tavily intelligent recommendation generation",
                "LLM-powered execution completion detection",
                "Comprehensive LLM-based overall assessment"
            ],
            "intelligence_level": "LLM_EXPERT_ANALYSIS"
        }

        overall_assessment = {
            "success_rate": 0.0,
            "critical_failures": 0,
            "total_steps": 0,
            "performance_rating": "pending",
            "recommendations": ["LLM-powered monitoring in progress..."],
            "llm_assessment": "LLM analysis will provide comprehensive evaluation"
        }

        return EvaluationReport(
            execution_metadata=execution_metadata,
            step_evaluations=[],
            overall_assessment=overall_assessment
        )

    def _save_llm_evaluation_report(self, report: EvaluationReport):
        """Save comprehensive LLM-powered evaluation report"""
        self.evaluation_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main report with LLM analyses
        report_dict = report.model_dump()
        
        # Add comprehensive LLM analysis results
        if hasattr(self, 'llm_log_analyses'):
            report_dict["comprehensive_llm_analyses"] = {
                "log_analyses": self.llm_log_analyses,
                "screenshot_analyses": getattr(self, 'llm_screenshot_analyses', []),
                "code_analyses": getattr(self, 'llm_code_analyses', [])
            }
        
        self.evaluation_file.write_text(json.dumps(report_dict, indent=2), encoding='utf-8')

        # Save detailed LLM analysis files
        analysis_dir = self.evaluation_file.parent / "detailed_llm_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Save comprehensive LLM analysis results
        if hasattr(self, 'llm_log_analyses'):
            (analysis_dir / "llm_log_analysis.json").write_text(
                json.dumps(self.llm_log_analyses, indent=2), encoding='utf-8'
            )
            
        if hasattr(self, 'llm_screenshot_analyses'):
            (analysis_dir / "llm_screenshot_analysis.json").write_text(
                json.dumps(self.llm_screenshot_analyses, indent=2), encoding='utf-8'
            )
            
        if hasattr(self, 'llm_code_analyses'):
            (analysis_dir / "llm_code_analysis.json").write_text(
                json.dumps(self.llm_code_analyses, indent=2), encoding='utf-8'
            )

        print(f"[{self.seq_no}] 📁 Comprehensive LLM-powered evaluation saved: {self.evaluation_file}")

    # Fallback methods (simplified versions without LLM)
    def _fallback_detect_completed_steps(self, recent_logs: List[str], screenshot_pairs: List[Dict]) -> List[Dict[str, Any]]:
        """Simple fallback step detection without LLM"""
        completed_steps = []
        log_content = "\n".join(recent_logs) if recent_logs else ""
        
        step_patterns = [r"✓.*[Ss]tep\s+(\d+)", r"[Ss]tep\s+(\d+).*success", r"✅.*[Ss]tep\s+(\d+)"]
        step_numbers = set()
        
        for pattern in step_patterns:
            matches = re.finditer(pattern, log_content, re.IGNORECASE)
            for match in matches:
                step_num = int(match.group(1))
                step_numbers.add(step_num)
        
        for step_num in step_numbers:
            completed_steps.append({
                "step_no": step_num,
                "step_name": f"Step {step_num}",
                "logs_content": log_content,
                "screenshots": self._find_step_screenshots(step_num, screenshot_pairs),
                "completion_detected": True,
                "llm_evidence": "Pattern-based detection",
                "llm_confidence": 0.6
            })
            
        return completed_steps

    def _fallback_evaluate_step(self, step_data: Dict[str, Any]) -> StepEvaluation:
        """Simple fallback step evaluation without LLM"""
        step_no = step_data["step_no"]
        step_name = step_data["step_name"] 
        logs_content = step_data.get("logs_content", "")
        screenshots = step_data.get("screenshots", {})
        
        # Simple pattern matching
        success_patterns = ["✓", "success", "completed", "done"]
        error_patterns = ["error", "failed", "timeout", "exception"]
        
        success_count = sum(logs_content.lower().count(p) for p in success_patterns)
        error_count = sum(logs_content.lower().count(p) for p in error_patterns)
        
        if error_count > success_count:
            status = StepStatus.FAILURE
            confidence = 0.4
            review = "Pattern-based analysis detected potential failure"
        elif success_count > 0:
            status = StepStatus.SUCCESS  
            confidence = 0.7
            review = None
        else:
            status = StepStatus.WARNING
            confidence = 0.3
            review = "Unclear status from pattern analysis"
        
        return StepEvaluation(
            step_no=step_no,
            step_name=step_name,
            status=status,
            review=review,
            logs_path=str(self.logs_dir / f"step_{step_no:03d}.log"),
            screenshot_start=screenshots.get("start"),
            screenshot_final=screenshots.get("final"),
            duration_seconds=5.0,
            confidence_score=confidence,
            evaluation_method="Pattern_Based_Fallback"
        )

    def _fallback_check_completion(self, recent_logs: List[str]) -> bool:
        """Simple fallback completion check without LLM"""
        log_content = '\n'.join(recent_logs) if recent_logs else ""
        completion_patterns = ["automation.*complete", "execution.*finished", "all steps.*done"]
        return any(re.search(pattern, log_content, re.IGNORECASE) for pattern in completion_patterns)

    def _fallback_overall_assessment(self, step_evaluations: List[StepEvaluation]) -> Dict[str, Any]:
        """Simple fallback overall assessment without LLM"""
        successful_steps = sum(1 for step in step_evaluations if step.status == StepStatus.SUCCESS)
        failed_steps = sum(1 for step in step_evaluations if step.status == StepStatus.FAILURE)
        warning_steps = sum(1 for step in step_evaluations if step.status == StepStatus.WARNING)
        success_rate = successful_steps / len(step_evaluations)
        
        if success_rate >= 0.9:
            rating = "excellent"
        elif success_rate >= 0.7:
            rating = "good"
        elif success_rate >= 0.5:
            rating = "fair"
        else:
            rating = "poor"
        
        return {
            "success_rate": success_rate,
            "critical_failures": failed_steps,
            "total_steps": len(step_evaluations),
            "successful_steps": successful_steps,
            "warning_steps": warning_steps,
            "failed_steps": failed_steps,
            "performance_rating": rating,
            "recommendations": ["Pattern-based analysis - manual review recommended"]
        }

    def _find_step_screenshots(self, step_no: int, screenshot_pairs: List[Dict]) -> Dict[str, str]:
        """Find screenshots for a specific step"""
        step_screenshots = {"start": None, "final": None}
        step_pattern = f"step_{step_no:03d}"
        
        for pair in screenshot_pairs:
            if step_pattern in pair.get("step_name", "").lower():
                step_screenshots["start"] = pair.get("start")
                step_screenshots["final"] = pair.get("final")
                break
                
        return step_screenshots

    def _extract_step_duration_llm(self, logs_content: str, step_no: int) -> float:
        """Extract step duration from logs"""
        duration_patterns = [
            rf"duration[:\s]+(\d+\.?\d*)",
            rf"took[:\s]+(\d+\.?\d*)",
            rf"completed in[:\s]+(\d+\.?\d*)"
        ]
        
        for pattern in duration_patterns:
            match = re.search(pattern, logs_content, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
                    
        return 5.0  # Default duration

# LLM-POWERED SUPPORTING CLASSES for intelligent monitoring

class LLMLogAnalyzer:
    """LLM-powered real-time log file monitoring and analysis"""
    
    def __init__(self, logs_dir: Path, seq_no: str):
        self.logs_dir = logs_dir
        self.seq_no = seq_no
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.last_read_positions = {}
        
    def get_recent_logs(self) -> List[str]:
        """Get recent log entries since last read"""
        recent_logs = []
        
        # Find all log files
        log_files = list(self.logs_dir.glob("*.log"))
        log_files.extend(list(self.logs_dir.glob("*.txt")))
        
        for log_file in log_files:
            if not log_file.exists():
                continue
                
            file_key = str(log_file)
            last_position = self.last_read_positions.get(file_key, 0)
            
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(last_position)
                    new_content = f.read()
                    current_position = f.tell()
                    
                    if new_content.strip():
                        recent_logs.extend(new_content.strip().split('\n'))
                        
                    self.last_read_positions[file_key] = current_position
                    
            except Exception as e:
                print(f"Error reading log file {log_file}: {e}")
                continue
                
        return recent_logs

    def analyze_logs_with_llm_intelligence(self, recent_logs: List[str]) -> Dict[str, Any]:
        """LLM-powered intelligent log analysis with deep reasoning"""
        log_content = "\n".join(recent_logs)
        
        # LLM-powered log analysis
        log_analysis_prompt = f"""You are an expert automation engineer analyzing execution logs. Provide intelligent analysis of these recent log entries.

LOG CONTENT:
{log_content}

Your analysis should cover:
1. **Execution Status**: What's currently happening in the automation
2. **Error Detection**: Any errors, warnings, or concerning patterns
3. **Success Indicators**: Positive progress and successful operations  
4. **Performance Insights**: Timing, efficiency, resource usage
5. **Risk Assessment**: Potential issues or failure predictions
6. **Actionable Insights**: What should be done based on these logs

Respond with JSON format:
{{
    "llm_assessment": "<PROGRESSING_WELL|ISSUES_DETECTED|CRITICAL_ERRORS|UNCLEAR>",
    "current_status": "<detailed_status_description>",
    "errors_found": [
        {{
            "type": "<error_type>",
            "severity": "<LOW|MEDIUM|HIGH|CRITICAL>", 
            "description": "<error_description>",
            "suggested_action": "<what_to_do>"
        }}
    ],
    "success_indicators": [
        {{
            "type": "<success_type>",
            "description": "<what_went_well>"
        }}
    ],
    "performance_analysis": "<timing_and_efficiency_assessment>",
    "risk_predictions": ["<potential_risk1>", "<potential_risk2>"],
    "actionable_recommendations": ["<recommendation1>", "<recommendation2>"],
    "confidence": <0.0-1.0>,
    "expert_summary": "<brief_expert_assessment>"
}}"""

        try:
            llm_response = get_llm_response(log_analysis_prompt,
                "You are a senior automation engineer with expertise in log analysis, debugging, and predictive analysis.", model_name=LLMProvider.GROQ)
            
            return json.loads(llm_response)
            
        except Exception as e:
            print(f"[{self.seq_no}] ⚠️ LLM log analysis failed: {e}")
            return {
                "llm_assessment": "ANALYSIS_FAILED",
                "current_status": f"LLM analysis error: {str(e)}",
                "errors_found": [],
                "success_indicators": [],
                "confidence": 0.0,
                "expert_summary": "Failed to analyze logs with LLM"
            }

class LLMScreenshotAnalyzer:
    """LLM-powered screenshot comparison and visual analysis"""
    
    def __init__(self, screenshots_dir: Path, seq_no: str):
        self.screenshots_dir = screenshots_dir
        self.seq_no = seq_no
        self.screenshots_dir.mkdir(parents=True, exist_ok=True)
        self.processed_screenshots = set()
        
    def detect_new_screenshot_pairs(self) -> List[Dict[str, str]]:
        """Detect new before/after screenshot pairs"""
        pairs = []
        
        # Find all screenshots
        all_screenshots = list(self.screenshots_dir.glob("*.png"))
        all_screenshots.extend(list(self.screenshots_dir.glob("*.jpg")))
        
        # Group by step
        step_groups = {}
        for screenshot in all_screenshots:
            if screenshot.name in self.processed_screenshots:
                continue
                
            # Extract step info from filename
            match = re.match(r'step_(\d+)_(.+)_(start|final)\.(png|jpg)', screenshot.name)
            if match:
                step_num, step_name, phase, ext = match.groups()
                key = f"step_{step_num}_{step_name}"
                
                if key not in step_groups:
                    step_groups[key] = {"step_name": step_name}
                    
                step_groups[key][phase] = str(screenshot)
                self.processed_screenshots.add(screenshot.name)
        
        # Find complete pairs
        for key, group in step_groups.items():
            if "start" in group and "final" in group:
                pairs.append(group)
                
        return pairs

    def analyze_screenshots_with_llm_intelligence(self, screenshot_pair: Dict[str, str]) -> Dict[str, Any]:
        """LLM-powered intelligent screenshot analysis"""
        step_name = screenshot_pair.get("step_name", "unknown")
        start_screenshot = screenshot_pair.get("start")
        final_screenshot = screenshot_pair.get("final")
        
        # Prepare screenshot analysis context
        screenshot_info = {
            "step_name": step_name,
            "has_start": bool(start_screenshot),
            "has_final": bool(final_screenshot),
            "file_analysis": {}
        }
        
        # Basic file analysis
        if start_screenshot and final_screenshot:
            try:
                start_path = Path(start_screenshot)
                final_path = Path(final_screenshot)
                
                if start_path.exists() and final_path.exists():
                    start_size = start_path.stat().st_size
                    final_size = final_path.stat().st_size
                    
                    screenshot_info["file_analysis"] = {
                        "start_size_kb": round(start_size / 1024, 2),
                        "final_size_kb": round(final_size / 1024, 2),
                        "size_change_percent": round(((final_size - start_size) / start_size * 100), 2) if start_size > 0 else 0,
                        "both_files_exist": True
                    }
                else:
                    screenshot_info["file_analysis"]["both_files_exist"] = False
            except Exception as e:
                screenshot_info["file_analysis"]["error"] = str(e)

        # LLM-powered screenshot analysis
        screenshot_analysis_prompt = f"""You are an expert automation engineer analyzing before/after screenshots to determine if an automation step was successful.

SCREENSHOT PAIR ANALYSIS:
Step Name: {step_name}
Screenshot Information: {json.dumps(screenshot_info, indent=2)}

Based on the available information about these screenshots, provide your expert analysis:

1. **Visual Change Assessment**: What do the file size changes suggest about visual changes?
2. **Success Likelihood**: How likely is it that the automation step succeeded?
3. **Failure Indicators**: What would suggest the step failed?
4. **Analysis Confidence**: How confident are you in this assessment?
5. **Recommendations**: What should be done based on this analysis?

Important considerations:
- Significant size changes often indicate successful UI interactions
- Very small changes might suggest the action didn't take effect
- Missing screenshots indicate capture failures
- File size can correlate with visual complexity changes

Respond with JSON format:
{{
    "llm_conclusion": "<SUCCESS_LIKELY|FAILURE_LIKELY|UNCLEAR|MISSING_DATA>",
    "visual_change_assessment": "<assessment_of_visual_changes>",
    "success_probability": <0.0-1.0>,
    "failure_indicators": ["<indicator1>", "<indicator2>"],
    "success_indicators": ["<indicator1>", "<indicator2>"],
    "analysis_confidence": <0.0-1.0>,
    "expert_recommendations": ["<recommendation1>", "<recommendation2>"],
    "technical_details": "<detailed_technical_analysis>",
    "next_steps": ["<step1>", "<step2>"]
}}"""

        try:
            llm_response = get_llm_response(screenshot_analysis_prompt,
                "You are a senior automation engineer with expertise in visual analysis, UI automation, and screenshot-based debugging.", model_name=LLMProvider.GROQ)
            
            llm_analysis = json.loads(llm_response)
            
            # Add basic info to LLM analysis
            llm_analysis.update({
                "step_name": step_name,
                "start_screenshot": start_screenshot,
                "final_screenshot": final_screenshot,
                "file_info": screenshot_info["file_analysis"]
            })
            
            return llm_analysis
            
        except Exception as e:
            print(f"[{self.seq_no}] ⚠️ LLM screenshot analysis failed: {e}")
            return {
                "llm_conclusion": "ANALYSIS_FAILED",
                "visual_change_assessment": f"LLM analysis error: {str(e)}",
                "success_probability": 0.0,
                "analysis_confidence": 0.0,
                "expert_recommendations": ["Manual screenshot review needed"]
            }

class LLMCodeAnalyzer:
    """LLM-powered code execution pattern analysis"""
    
    def __init__(self, artifacts_dir: Path, seq_no: str):
        self.artifacts_dir = artifacts_dir
        self.seq_no = seq_no
        self.last_analysis_time = 0
        
    def detect_execution_patterns_with_llm(self) -> Dict[str, Any]:
        """Detect code execution patterns with LLM analysis trigger"""
        current_time = time.time()
        
        # Only analyze if enough time has passed
        if current_time - self.last_analysis_time < 10:  # Every 10 seconds for LLM
            return {"new_patterns": [], "llm_should_analyze": False}
            
        self.last_analysis_time = current_time
        
        patterns_detected = []
        code_content = {}
        
        # Look for automation script and analyze its content
        script_path = self.artifacts_dir / "enhanced_agent2" / "automation_script.py"
        if script_path.exists():
            try:
                script_content = script_path.read_text(encoding='utf-8')
                code_content["automation_script"] = script_content[-2000:]  # Last 2000 chars
                
                # Basic pattern detection
                if "exception" in script_content.lower():
                    patterns_detected.append("exception_handling_present")
                if "retry" in script_content.lower():
                    patterns_detected.append("retry_mechanisms_present")
                if "screenshot" in script_content.lower():
                    patterns_detected.append("screenshot_capture_present")
                if "def " in script_content:
                    patterns_detected.append("function_definitions_present")
                    
            except Exception as e:
                patterns_detected.append("script_read_error")
                code_content["error"] = str(e)

        return {
            "new_patterns": patterns_detected,
            "llm_should_analyze": len(patterns_detected) > 0 or len(code_content) > 0,
            "code_content": code_content
        }

    def analyze_execution_with_llm_intelligence(self, execution_status: Dict[str, Any]) -> Dict[str, Any]:
        """LLM-powered code execution analysis"""
        patterns = execution_status.get("new_patterns", [])
        code_content = execution_status.get("code_content", {})
        
        # LLM-powered code analysis
        code_analysis_prompt = f"""You are a senior automation engineer analyzing code execution patterns. Provide expert analysis of the automation code and its execution.

DETECTED PATTERNS:
{json.dumps(patterns, indent=2)}

CODE CONTENT (recent sections):
{json.dumps(code_content, indent=2)}

Your analysis should cover:
1. **Code Quality Assessment**: How well is the automation code structured?
2. **Execution Health**: Are there signs of healthy or problematic execution?
3. **Performance Indicators**: Any performance concerns or optimizations?
4. **Risk Assessment**: Potential execution risks or failure points
5. **Best Practices**: How well does the code follow automation best practices?
6. **Recommendations**: Specific improvements for better reliability

Respond with JSON format:
{{
    "llm_assessment": "<HEALTHY_EXECUTION|PERFORMANCE_ISSUES|CODE_PROBLEMS|EXECUTION_RISKS>",
    "code_quality_rating": "<EXCELLENT|GOOD|FAIR|POOR>",
    "execution_health": "<HEALTHY|CONCERNING|PROBLEMATIC>",
    "strengths": ["<strength1>", "<strength2>"],
    "weaknesses": ["<weakness1>", "<weakness2>"],
    "performance_insights": "<performance_analysis>",
    "risk_factors": ["<risk1>", "<risk2>"],
    "optimization_recommendations": ["<rec1>", "<rec2>"],
    "confidence": <0.0-1.0>,
    "expert_summary": "<brief_code_analysis_summary>"
}}"""

        try:
            llm_response = get_llm_response(code_analysis_prompt,
                "You are a senior automation engineer with expertise in code analysis, performance optimization, and automation reliability.", model_name=LLMProvider.GROQ)
            
            return json.loads(llm_response)
            
        except Exception as e:
            print(f"[{self.seq_no}] ⚠️ LLM code analysis failed: {e}")
            return {
                "llm_assessment": "ANALYSIS_FAILED",
                "code_quality_rating": "UNKNOWN",
                "execution_health": "UNKNOWN",
                "confidence": 0.0,
                "expert_summary": f"LLM analysis error: {str(e)}"
            }

class LLMTavilyResearcher:
    """LLM-enhanced Tavily search integration for intelligent error resolution"""
    
    def __init__(self, seq_no: str):
        self.seq_no = seq_no
        self.client = TavilyClient() if TAVILY_AVAILABLE else None
        
    def research_step_failure_with_llm(self, step_name: str, error_context: str) -> List[str]:
        """Use LLM to enhance Tavily research for step failures"""
        if not self.client:
            return [f"Tavily not available - manual research needed for '{step_name}' failure"]
        
        # First, use LLM to create better search queries
        query_generation_prompt = f"""You are an expert at creating effective search queries for automation debugging. 

FAILED STEP: {step_name}
ERROR CONTEXT: {error_context[-300:] if error_context else "No error context"}

Generate 2-3 specific, effective search queries that would help find solutions to this automation failure. Focus on:
1. The specific automation technology (Selenium, Appium, Playwright)
2. The type of failure or error
3. Practical solutions and fixes

Respond with JSON:
{{
    "search_queries": ["<query1>", "<query2>", "<query3>"],
    "reasoning": "<why_these_queries_will_be_effective>"
}}"""

        try:
            # Get LLM-generated search queries
            llm_response = get_llm_response(query_generation_prompt,
                "You are an expert automation engineer who excels at finding solutions to technical problems.", model_name=LLMProvider.GROQ)
            
            query_data = json.loads(llm_response)
            search_queries = query_data.get("search_queries", [f"automation {step_name} error solution"])
            
            # Use Tavily to research with LLM-generated queries
            all_solutions = []
            for query in search_queries[:2]:  # Limit to 2 queries
                try:
                    response = self.client.search(query, max_results=2)
                    
                    for result in response.get('results', []):
                        title = result.get('title', '')
                        content = result.get('content', '')
                        
                        if content and len(content) > 50:  # Meaningful content
                            # Use LLM to extract actionable solutions from search results
                            solution_extraction_prompt = f"""Extract actionable solutions from this search result for the automation failure.

ORIGINAL PROBLEM: {step_name} failed
ERROR CONTEXT: {error_context[-200:] if error_context else "No context"}

SEARCH RESULT:
Title: {title}
Content: {content[:800]}

Extract 1-2 specific, actionable solutions that directly address this automation problem. Focus on practical steps that can be implemented immediately.

Respond with just the solutions, one per line, starting with "•"."""

                            try:
                                solution_response = get_llm_response(solution_extraction_prompt,
                                    "You are an expert at extracting actionable solutions from technical documentation.", model_name=LLMProvider.GROQ)
                                
                                if solution_response and len(solution_response.strip()) > 10:
                                    all_solutions.append(f"Research-based: {solution_response.strip()}")
                                    
                            except:
                                # Fallback to raw content
                                solution = content[:200] + "..." if len(content) > 200 else content
                                all_solutions.append(f"From {title}: {solution}")
                                
                except Exception as e:
                    print(f"[{self.seq_no}] Tavily search failed for query '{query}': {e}")
                    continue
                    
            return all_solutions[:3]  # Return top 3 solutions
            
        except Exception as e:
            print(f"[{self.seq_no}] ⚠️ LLM Tavily research failed: {e}")
            return [f"Research failed for step '{step_name}' - manual investigation needed"]

# Execution Intelligence System (Enhanced from original)
class ExecutionIntelligence:
    """Smart execution system that learns from previous runs and optimizes future executions"""

    def __init__(self):
        self.execution_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.common_issues: Dict[str, int] = {}
        self.optimization_suggestions: List[str] = []

    def record_execution(self, seq_no: str, platform: str, success: bool,
                         duration: float, issues: List[str] = None):
        """Record execution results for learning"""
        execution_record = {
            "seq_no": seq_no,
            "platform": platform,
            "success": success,
            "duration": duration,
            "timestamp": time.time(),
            "issues": issues or []
        }
        
        self.execution_history.append(execution_record)
        
        # Update performance metrics
        if platform not in self.performance_metrics:
            self.performance_metrics[platform] = []
        self.performance_metrics[platform].append(duration)
        
        # Track common issues
        for issue in (issues or []):
            self.common_issues[issue] = self.common_issues.get(issue, 0) + 1

    def get_platform_insights(self, platform: str) -> Dict[str, Any]:
        """Get insights for specific platform based on execution history"""
        platform_executions = [ex for ex in self.execution_history if ex["platform"] == platform]
        
        if not platform_executions:
            return {"insights": "No previous executions for this platform"}

        success_rate = sum(1 for ex in platform_executions if ex["success"]) / len(platform_executions)
        avg_duration = sum(ex["duration"] for ex in platform_executions) / len(platform_executions)
        
        # Common issues for this platform
        platform_issues = {}
        for ex in platform_executions:
            for issue in ex.get("issues", []):
                platform_issues[issue] = platform_issues.get(issue, 0) + 1

        insights = {
            "total_executions": len(platform_executions),
            "success_rate": success_rate,
            "average_duration": avg_duration,
            "common_issues": dict(sorted(platform_issues.items(), key=lambda x: x[1], reverse=True)[:5]),
            "recommendations": self._generate_recommendations(platform, success_rate, platform_issues)
        }
        
        return insights

    def _generate_recommendations(self, platform: str, success_rate: float,
                                 issues: Dict[str, int]) -> List[str]:
        """Generate recommendations based on execution history"""
        recommendations = []
        
        if success_rate < 0.8:
            recommendations.append("Consider increasing timeout values for better reliability")
            recommendations.append("Add more robust error handling and retry mechanisms")
        
        if "timeout" in str(issues).lower():
            recommendations.append("Implement adaptive timeout based on system performance")
        
        if "dependency" in str(issues).lower():
            recommendations.append("Create dependency pre-check before execution")
        
        if platform == "mobile":
            if "appium" in str(issues).lower():
                recommendations.append("Add Appium server health check before execution")
                recommendations.append("Verify noReset=True capability is correctly set")
            recommendations.append("Consider device-specific optimizations")
        else:  # web
            if "browser" in str(issues).lower():
                recommendations.append("Add browser compatibility checks")
                recommendations.append("Verify stealth configuration is properly applied")
                recommendations.append("Implement headless mode for better stability")
        
        return recommendations

# Global execution intelligence instance
execution_intelligence = ExecutionIntelligence()

class TerminalExecutionEngine:
    """Enhanced execution engine with LLM-POWERED intelligent evaluation capabilities"""

    def __init__(self, seq_no: str, config: ExecutionConfig):
        self.seq_no = seq_no
        self.config = config
        self.artifacts_dir = ARTIFACTS_DIR / seq_no
        self.agent3_dir = self.artifacts_dir / "enhanced_agent3"
        self.agent3_dir.mkdir(parents=True, exist_ok=True)

        # Determine source directories (enhanced or regular)
        self.agent1_dir = self.artifacts_dir / ("enhanced_agent1" if config.use_enhanced_agent else "agent1")
        self.agent2_dir = self.artifacts_dir / ("enhanced_agent2" if config.use_enhanced_agent else "agent2")
        
        # Initialize LLM-POWERED intelligent agent
        self.intelligent_agent = LLMIntelligentAgent3(seq_no)

    def validate_prerequisites_smart(self) -> Dict[str, Any]:
        """Enhanced validation with intelligent recommendations"""
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "file_status": {},
            "intelligence_insights": {}
        }

        # Get platform insights from execution intelligence
        platform_insights = execution_intelligence.get_platform_insights(self.config.platform.value)
        validation_results["intelligence_insights"] = platform_insights

        # Check for required files
        required_files = {
            "script": self.agent2_dir / "automation_script.py",
            "requirements": self.agent2_dir / "requirements.txt"
        }

        # Enhanced agent files
        if self.config.use_enhanced_agent:
            required_files.update({
                "blueprint": self.agent1_dir / "blueprint.json",
                "analysis": self.agent2_dir / "code_analysis.json",
                "documentation": self.agent2_dir / "documentation.md",
                "roadmap": self.agent2_dir / "implementation_roadmap.md"
            })

        for file_type, file_path in required_files.items():
            if file_path.exists():
                validation_results["file_status"][file_type] = {
                    "exists": True,
                    "size": file_path.stat().st_size,
                    "path": str(file_path),
                    "modified": file_path.stat().st_mtime
                }
            else:
                validation_results["valid"] = False
                validation_results["issues"].append(f"Missing {file_type} file: {file_path}")
                validation_results["file_status"][file_type] = {"exists": False, "path": str(file_path)}

        # Platform-specific validations with intelligence
        if self.config.platform == Platform.MOBILE:
            self._validate_mobile_prerequisites(validation_results, platform_insights)
        else:
            self._validate_web_prerequisites(validation_results, platform_insights)

        # Check Python environment
        validation_results["python_version"] = sys.version
        validation_results["python_executable"] = sys.executable

        # Add intelligent recommendations based on history
        if platform_insights and "recommendations" in platform_insights:
            validation_results["recommendations"] = platform_insights["recommendations"]

        return validation_results

    def _validate_mobile_prerequisites(self, validation_results: Dict, insights: Dict):
        """Mobile-specific validation with intelligent recommendations"""
        # Check for Appium server availability
        try:
            result = subprocess.run(["appium", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                validation_results["appium_version"] = result.stdout.strip()
                # Intelligent recommendation based on history
                if insights.get("common_issues", {}).get("appium_connection", 0) > 2:
                    validation_results["warnings"].append("Previous Appium connection issues detected - consider server restart")
            else:
                validation_results["warnings"].append("Appium not found or not working")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            validation_results["warnings"].append("Appium command not available")

        # Check for ADB
        try:
            result = subprocess.run(["adb", "version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                validation_results["adb_available"] = True
            else:
                validation_results["warnings"].append("ADB not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            validation_results["warnings"].append("ADB command not found")

    def _validate_web_prerequisites(self, validation_results: Dict, insights: Dict):
        """Web-specific validation with intelligent recommendations"""
        # Check for browser availability
        browsers = ["chromium", "firefox", "webkit"]
        available_browsers = []
        
        for browser in browsers:
            try:
                if browser == "chromium":
                    result = subprocess.run(["google-chrome", "--version"],
                                          capture_output=True, text=True, timeout=5)
                elif browser == "firefox":
                    result = subprocess.run(["firefox", "--version"],
                                          capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    available_browsers.append(browser)
            except:
                continue

        validation_results["available_browsers"] = available_browsers
        if not available_browsers:
            validation_results["warnings"].append("No browsers detected for web automation")

        # Intelligent recommendation based on browser issues
        if insights.get("common_issues", {}).get("browser_crash", 0) > 1:
            validation_results["warnings"].append("Previous browser crashes detected - consider headless mode")

    def create_virtual_environment_smart(self) -> bool:
        """Enhanced virtual environment creation with intelligent optimization"""
        venv_dir = self.agent3_dir / "venv"
        python_exe = sys.executable

        try:
            print(f"[{self.seq_no}] 🧠 Creating optimized virtual environment...")
            
            # Get platform insights for optimization
            platform_insights = execution_intelligence.get_platform_insights(self.config.platform.value)

            # Remove existing venv if present
            if venv_dir.exists():
                shutil.rmtree(venv_dir)

            # Create new venv with timeout based on historical performance
            expected_duration = platform_insights.get("average_duration", 30)
            timeout = max(60, int(expected_duration * 1.5))

            result = subprocess.run([python_exe, "-m", "venv", str(venv_dir)],
                                  capture_output=True, text=True, timeout=timeout)
            
            if result.returncode != 0:
                print(f"Venv creation failed: {result.stderr}")
                return False

            # Get venv python path
            if sys.platform == "win32":
                venv_python = venv_dir / "Scripts" / "python.exe"
                venv_pip = venv_dir / "Scripts" / "pip.exe"
            else:
                venv_python = venv_dir / "bin" / "python"
                venv_pip = venv_dir / "bin" / "pip"

            # Upgrade pip with intelligent timeout
            subprocess.run([str(venv_pip), "install", "--upgrade", "pip"], timeout=60, check=False)

            # Install requirements with intelligent error handling
            req_file = self.agent2_dir / "requirements.txt"
            if req_file.exists():
                print(f"[{self.seq_no}] 📦 Installing requirements with intelligent optimization...")
                
                # Fix duplicate package issues first
                requirements_content = req_file.read_text().strip()
                requirements_lines = []
                seen_packages = set()
                
                for line in requirements_content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        package_name = line.split('==')[0].split('>=')[0].split('<=')[0].lower()
                        if package_name not in seen_packages:
                            requirements_lines.append(line)
                            seen_packages.add(package_name)

                # Write cleaned requirements
                clean_req_file = self.agent3_dir / "requirements_clean.txt"
                clean_req_file.write_text('\n'.join(requirements_lines))

                # Install cleaned requirements
                result = subprocess.run([str(venv_pip), "install", "-r", str(clean_req_file)],
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    print(f"Requirements installation failed: {result.stderr}")
                    return False

            # Platform-specific installations with intelligence
            if self.config.platform == Platform.WEB:
                try:
                    print(f"[{self.seq_no}] 🎭 Installing Playwright browsers...")
                    subprocess.run([str(venv_python), "-m", "playwright", "install"],
                                 timeout=300, check=False)
                except subprocess.TimeoutExpired:
                    print("Playwright browser installation timed out")

            return True

        except Exception as e:
            print(f"Virtual environment setup failed: {e}")
            # Record failure for intelligence
            execution_intelligence.record_execution(
                self.seq_no, self.config.platform.value, False, 0,
                [f"venv_setup_failure: {str(e)}"]
            )
            return False

    def copy_generated_script(self) -> bool:
        """Copy the generated script to agent3 directory for execution"""
        try:
            source_script = self.agent2_dir / "automation_script.py"
            target_script = self.agent3_dir / "automation_script.py"

            if not source_script.exists():
                print(f"[{self.seq_no}] ❌ Source script not found: {source_script}")
                return False

            # Copy the script
            shutil.copy2(source_script, target_script)

            # Copy requirements if they exist
            source_req = self.agent2_dir / "requirements.txt"
            target_req = self.agent3_dir / "requirements.txt"
            if source_req.exists():
                shutil.copy2(source_req, target_req)

            print(f"[{self.seq_no}] ✅ Script copied successfully to agent3 directory")
            return True

        except Exception as e:
            print(f"[{self.seq_no}] ❌ Failed to copy script: {e}")
            return False

    def launch_mobile_terminals_with_llm_intelligence(self) -> Dict[str, Any]:
        """Launch Appium server terminal and execution terminal for mobile with LLM-powered monitoring"""
        try:
            print(f"[{self.seq_no}] 📱 Launching Mobile Automation Terminals with LLM-POWERED Intelligence...")

            # Get venv paths
            if sys.platform == "win32":
                venv_python = self.agent3_dir / "venv" / "Scripts" / "python.exe"
                activate_script = self.agent3_dir / "venv" / "Scripts" / "activate.bat"
            else:
                venv_python = self.agent3_dir / "venv" / "bin" / "python"
                activate_script = self.agent3_dir / "venv" / "bin" / "activate"

            # 1. Launch Appium Server Terminal
            if sys.platform == "win32":
                appium_cmd = [
                    "cmd", "/c", "start", "cmd", "/k",
                    f"title Appium Server - Task {self.seq_no} && echo Starting Appium Server for Task {self.seq_no}... && appium --allow-cors --log-timestamp --log-level info"
                ]
            else:
                appium_cmd = [
                    "gnome-terminal", "--title", f"Appium Server - {self.seq_no}",
                    "--", "bash", "-c",
                    f"echo 'Starting Appium Server for Task {self.seq_no}...'; appium --allow-cors --log-timestamp --log-level info; read -p 'Press Enter to close...'"
                ]

            appium_process = subprocess.Popen(appium_cmd)
            print(f"[{self.seq_no}] 🚀 Appium Server Terminal launched (PID: {appium_process.pid})")
            
            # Wait a bit for Appium to start
            time.sleep(5)

            # 2. Launch Execution Terminal with LLM-POWERED intelligent monitoring
            script_path = self.agent3_dir / "automation_script.py"
            
            if sys.platform == "win32":
                exec_cmd = [
                    "cmd", "/c", "start", "cmd", "/k",
                    f"title Mobile Automation - Task {self.seq_no} && cd /d {self.agent3_dir} && call {activate_script} && echo Running Mobile Automation Script with LLM-POWERED Intelligence... && python {script_path} && pause"
                ]
            else:
                exec_cmd = [
                    "gnome-terminal", "--title", f"Mobile Automation - {self.seq_no}",
                    "--", "bash", "-c",
                    f"cd {self.agent3_dir} && source {activate_script} && echo 'Running Mobile Automation Script with LLM-POWERED Intelligence...' && python {script_path}; read -p 'Press Enter to close...'"
                ]

            exec_process = subprocess.Popen(exec_cmd)
            print(f"[{self.seq_no}] 🚀 Mobile Execution Terminal launched (PID: {exec_process.pid})")

            # 3. Start LLM-POWERED Intelligent Monitoring in background
            print(f"[{self.seq_no}] 🧠 Starting LLM-POWERED Intelligent Agent monitoring...")
            
            # Load blueprint for intelligent analysis
            blueprint_data = None
            if (self.agent1_dir / "blueprint.json").exists():
                try:
                    blueprint_data = json.loads((self.agent1_dir / "blueprint.json").read_text())
                except:
                    pass

            # Start LLM-powered intelligent monitoring (this will run continuously)
            evaluation_report = self.intelligent_agent.watch_and_evaluate_realtime_with_llm(blueprint_data)

            return {
                "appium_server_pid": appium_process.pid,
                "execution_pid": exec_process.pid,
                "llm_intelligent_monitoring": True,
                "evaluation_report": evaluation_report.model_dump(),
                "status": "terminals_launched_with_llm_intelligence"
            }

        except Exception as e:
            print(f"[{self.seq_no}] ❌ Failed to launch mobile terminals: {e}")
            return {"error": str(e), "status": "failed"}

    def launch_web_terminals_with_llm_intelligence(self) -> Dict[str, Any]:
        """Launch web automation terminal with LLM-POWERED intelligent monitoring"""
        try:
            print(f"[{self.seq_no}] 🌐 Launching Web Automation Terminal with LLM-POWERED Intelligence...")

            # Get venv paths
            if sys.platform == "win32":
                venv_python = self.agent3_dir / "venv" / "Scripts" / "python.exe"
                activate_script = self.agent3_dir / "venv" / "Scripts" / "activate.bat"
            else:
                venv_python = self.agent3_dir / "venv" / "bin" / "python"
                activate_script = self.agent3_dir / "venv" / "bin" / "activate"

            script_path = self.agent3_dir / "automation_script.py"

            # Launch Execution Terminal with LLM-POWERED intelligent monitoring
            if sys.platform == "win32":
                exec_cmd = [
                    "cmd", "/c", "start", "cmd", "/k",
                    f"title Web Automation - Task {self.seq_no} && cd /d {self.agent3_dir} && call {activate_script} && echo Running Web Automation Script with LLM-POWERED Intelligence... && python {script_path} && pause"
                ]
            else:
                exec_cmd = [
                    "gnome-terminal", "--title", f"Web Automation - {self.seq_no}",
                    "--", "bash", "-c",
                    f"cd {self.agent3_dir} && source {activate_script} && echo 'Running Web Automation Script with LLM-POWERED Intelligence...' && python {script_path}; read -p 'Press Enter to close...'"
                ]

            exec_process = subprocess.Popen(exec_cmd)
            print(f"[{self.seq_no}] 🚀 Web Execution Terminal launched (PID: {exec_process.pid})")

            # Start LLM-POWERED Intelligent Monitoring
            print(f"[{self.seq_no}] 🧠 Starting LLM-POWERED Intelligent Agent monitoring...")
            
            # Load blueprint for intelligent analysis
            blueprint_data = None
            if (self.agent1_dir / "blueprint.json").exists():
                try:
                    blueprint_data = json.loads((self.agent1_dir / "blueprint.json").read_text())
                except:
                    pass

            # Start LLM-powered intelligent monitoring
            evaluation_report = self.intelligent_agent.watch_and_evaluate_realtime_with_llm(blueprint_data)

            return {
                "execution_pid": exec_process.pid,
                "llm_intelligent_monitoring": True,
                "evaluation_report": evaluation_report.model_dump(),
                "status": "terminal_launched_with_llm_intelligence"
            }

        except Exception as e:
            print(f"[{self.seq_no}] ❌ Failed to launch web terminal: {e}")
            return {"error": str(e), "status": "failed"}

def run_enhanced_agent3(seq_no: str, platform: str, use_enhanced: bool = True) -> dict:
    """Enhanced Agent 3 with LLM-POWERED intelligent monitoring and evaluation"""
    print(f"[{seq_no}] 🚀 Running LLM-POWERED REAL Intelligent Enhanced Agent 3")
    print(f"[{seq_no}] 🧠 LLM-POWERED Intelligence Features:")
    print(f"[{seq_no}]    🤔 LLM thinking and reasoning for all analysis")
    print(f"[{seq_no}]    👁️ LLM-powered real-time log analysis with expert insights")
    print(f"[{seq_no}]    🖼️ LLM-powered screenshot comparison with visual understanding")
    print(f"[{seq_no}]    💻 LLM-powered code execution pattern analysis")
    print(f"[{seq_no}]    🔍 LLM + Tavily search for intelligent error resolution")
    print(f"[{seq_no}]    📊 LLM-powered comprehensive step-by-step evaluation")
    print(f"[{seq_no}]    🎯 LLM-powered expert recommendations and insights")
    
    start_time = time.time()
    issues = []

    try:
        # Create execution config
        platform_enum = Platform.MOBILE if platform.lower() == "mobile" else Platform.WEB
        config = ExecutionConfig(
            platform=platform_enum,
            use_enhanced_agent=use_enhanced,
            enable_monitoring=True,
            enable_reporting=True
        )

        # Initialize execution engine with LLM-POWERED intelligence
        engine = TerminalExecutionEngine(seq_no, config)

        print(f"[{seq_no}] 🔍 Running intelligent prerequisites validation...")
        validation = engine.validate_prerequisites_smart()

        if not validation["valid"]:
            issues.extend(validation["issues"])
            print(f"[{seq_no}] ❌ Validation failed: {validation['issues']}")
            
            # Record failure in intelligence
            execution_intelligence.record_execution(
                seq_no, platform, False, time.time() - start_time,
                validation["issues"]
            )
            
            return {
                "status": "failed",
                "validation": validation,
                "issues": issues,
                "duration": time.time() - start_time
            }

        # Display intelligence insights
        insights = validation.get("intelligence_insights", {})
        if insights and insights.get("total_executions", 0) > 0:
            print(f"[{seq_no}] 🧠 Intelligence Insights:")
            print(f"   - Previous executions: {insights.get('total_executions', 0)}")
            print(f"   - Success rate: {insights.get('success_rate', 0):.1%}")
            print(f"   - Average duration: {insights.get('average_duration', 0):.1f}s")

        print(f"[{seq_no}] 🏗️ Creating intelligent virtual environment...")
        if not engine.create_virtual_environment_smart():
            issues.append("Failed to create virtual environment")
            execution_intelligence.record_execution(
                seq_no, platform, False, time.time() - start_time, issues
            )
            return {
                "status": "failed",
                "error": "Virtual environment creation failed",
                "issues": issues,
                "duration": time.time() - start_time
            }

        print(f"[{seq_no}] 📋 Copying generated automation script...")
        if not engine.copy_generated_script():
            issues.append("Failed to copy automation script")
            return {
                "status": "failed",
                "error": "Script copy failed",
                "issues": issues,
                "duration": time.time() - start_time
            }

        print(f"[{seq_no}] 🚀 Launching terminal execution with LLM-POWERED Intelligence...")
        
        # Launch platform-specific terminals with LLM-POWERED intelligence
        if platform_enum == Platform.MOBILE:
            terminal_result = engine.launch_mobile_terminals_with_llm_intelligence()
        else:
            terminal_result = engine.launch_web_terminals_with_llm_intelligence()

        # Record successful setup
        setup_duration = time.time() - start_time
        execution_intelligence.record_execution(
            seq_no, platform, True, setup_duration, []
        )

        print(f"[{seq_no}] ✅ LLM-POWERED REAL Intelligent Enhanced Agent 3 completed successfully!")
        print(f"[{seq_no}] 🧠 LLM-POWERED Intelligence Features Now Active:")
        print(f"       🤔 LLM thinking and reasoning for all analysis")
        print(f"       👁️ LLM-powered continuous log monitoring with expert insights")
        print(f"       🖼️ LLM-powered before/after screenshot analysis with visual understanding")
        print(f"       💻 LLM-powered real-time code execution monitoring")
        print(f"       🔍 LLM + Tavily-powered error research and intelligent solutions")
        print(f"       📊 LLM-powered step-by-step intelligent evaluation")
        print(f"       🎯 LLM-powered expert recommendations and insights")
        print(f"[{seq_no}] 📊 Setup completed in {setup_duration:.1f} seconds")
        print(f"[{seq_no}] 🚀 Terminals launched with LLM-POWERED Intelligence - monitoring active!")
        print(f"[{seq_no}] 📋 LLM-powered intelligent evaluation will be saved to: enhanced_agent3/evaluate.json")

        return {
            "status": "success",
            "validation": validation,
            "terminal_result": terminal_result,
            "intelligence_insights": insights,
            "llm_powered_intelligence_features": [
                "LLM thinking and reasoning for all analysis",
                "LLM-powered real-time log analysis with expert insights",
                "LLM-powered screenshot comparison with visual understanding", 
                "LLM-powered code execution pattern analysis",
                "LLM + Tavily search for intelligent error resolution",
                "LLM-powered comprehensive step-by-step evaluation",
                "LLM-powered expert recommendations and insights generation"
            ],
            "evaluation_file": str(engine.intelligent_agent.evaluation_file),
            "duration": setup_duration,
            "issues": issues
        }

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"LLM-POWERED REAL Intelligent Enhanced Agent 3 failed: {e}"
        issues.append(error_msg)
        
        # Record failure in intelligence
        execution_intelligence.record_execution(
            seq_no, platform, False, duration, issues
        )
        
        print(f"[{seq_no}] ❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# Export main function with backward compatibility
def run_agent3(seq_no: str, platform: str, use_enhanced: bool = True) -> dict:
    """Backward compatible wrapper for LLM-POWERED intelligent agent 3"""
    return run_enhanced_agent3(seq_no, platform, use_enhanced)

# Export LLM-POWERED intelligent classes
__all__ = [
    "run_enhanced_agent3", "run_agent3", "TerminalExecutionEngine",
    "ExecutionIntelligence", "execution_intelligence", "LLMIntelligentAgent3",
    "StepEvaluation", "EvaluationReport", "LLMLogAnalyzer", 
    "LLMScreenshotAnalyzer", "LLMCodeAnalyzer", "LLMTavilyResearcher"
]
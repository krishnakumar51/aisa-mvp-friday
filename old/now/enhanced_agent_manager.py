# enhanced/enhanced_agent_manager.py
"""
Enhanced Agent Manager: Unified coordination with shared schema and robust error handling
Key improvements:
- Uses shared schema models for all agent communication
- Centralized error handling and status management
- Persistent cross-agent intelligence sharing
- Performance monitoring and optimization
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import HTTPException

# Import shared modules (CRITICAL: prevents schema drift)
from shared_models import (
    AutomationBlueprint, Agent1Output, Agent2Output, Agent3Output,
    TaskStatus, ValidationLevel, Platform, ExecutionStatus, 
    create_error_response
)
from now.utils import (
    PersistentLLMCache, PerformanceMonitor, monitor_performance,
    ensure_directory, safe_file_write, validate_with_detailed_errors
)

# Import fixed agents (use the new unified versions)
from now.enhanced_agent_1 import run_enhanced_agent1, get_agent1_stats
from now.enhanced_agent_2 import run_enhanced_agent2, get_agent2_stats
from now.enhanced_agent_3 import run_enhanced_agent3, get_execution_intelligence_stats

from config import ARTIFACTS_DIR

# Global instances for manager coordination
cache_dir = ensure_directory(Path(ARTIFACTS_DIR) / "cache")
manager_cache = PersistentLLMCache(cache_dir / "manager_cache.db", default_ttl=3600)
performance_monitor = PerformanceMonitor()

class EnhancedAgentManager:
    """
    Unified Agent Manager with advanced coordination, error recovery, and intelligence sharing
    """

    def __init__(self):
        self.cache = manager_cache
        self.monitor = performance_monitor
        self.execution_log: Dict[str, Dict[str, Any]] = {}
        self.global_insights: Dict[str, Any] = {}

        # Load historical insights
        self._load_global_insights()

    def _load_global_insights(self):
        """Load cross-agent insights from persistent cache"""
        try:
            cached_insights = self.cache.get("global_insights", model="manager")
            if cached_insights:
                self.global_insights = cached_insights
        except Exception:
            pass  # Start fresh if loading fails

    def _save_global_insights(self):
        """Persist cross-agent insights"""
        try:
            self.cache.set(
                "global_insights",
                self.global_insights,
                model="manager",
                ttl=86400,  # 24 hours
                tags=["insights", "manager"]
            )
        except Exception:
            pass  # Don't fail pipeline if saving insights fails

    def run_enhanced_pipeline(self, seq_no: str, pdf_content: str, 
                            user_instructions: str, images: List[str] = None) -> Dict[str, Any]:
        """
        Main pipeline execution with unified error handling and status tracking
        """
        images = images or []
        start_time = time.time()

        # Initialize task status
        task_status = TaskStatus(
            seq_no=seq_no,
            phase="agent1",
            start_time=start_time
        )

        self._update_task_status(seq_no, task_status)

        try:
            with monitor_performance(self.monitor, "full_pipeline"):
                # Phase 1: Blueprint Generation
                blueprint_result = self._run_agent1_with_recovery(
                    seq_no, pdf_content, user_instructions, images, task_status
                )

                if not blueprint_result['success']:
                    return self._handle_pipeline_failure(
                        seq_no, "agent1", blueprint_result, task_status
                    )

                # Phase 2: Code Generation
                code_result = self._run_agent2_with_recovery(
                    seq_no, blueprint_result, task_status
                )

                if not code_result['success']:
                    return self._handle_pipeline_failure(
                        seq_no, "agent2", code_result, task_status
                    )

                # Phase 3: Execution
                execution_result = self._run_agent3_with_recovery(
                    seq_no, task_status
                )

                # Final status update
                task_status.phase = "completed" if execution_result['success'] else "failed"
                task_status.phase_times['total'] = time.time() - start_time
                self._update_task_status(seq_no, task_status)

                # Record insights for future optimizations
                self._record_pipeline_insights(seq_no, {
                    'agent1': blueprint_result,
                    'agent2': code_result,
                    'agent3': execution_result
                })

                return {
                    "success": execution_result['success'],
                    "seq_no": seq_no,
                    "total_time": time.time() - start_time,
                    "phases": {
                        "blueprint": blueprint_result,
                        "code": code_result,
                        "execution": execution_result
                    },
                    "task_status": task_status.model_dump(),
                    "performance_stats": self.monitor.get_stats(),
                    "insights": self.global_insights.get(seq_no, {})
                }

        except Exception as e:
            return self._handle_critical_failure(seq_no, str(e), task_status)

    def _run_agent1_with_recovery(self, seq_no: str, pdf_content: str, 
                                 user_instructions: str, images: List[str],
                                 task_status: TaskStatus) -> Dict[str, Any]:
        """
        Run Agent 1 with error recovery and validation
        """
        phase_start = time.time()

        try:
            # Check cache for similar blueprints
            cache_key = f"blueprint_{len(pdf_content)}_{len(images)}_{hash(user_instructions)}"
            cached_result = self.cache.get(cache_key, model="agent1")

            if cached_result:
                task_status.cache_hits += 1
                result = cached_result
                result['cache_used'] = True
            else:
                # Run Agent 1
                result = run_enhanced_agent1(seq_no, pdf_content, user_instructions, images)

                # Cache successful results
                if result['success']:
                    self.cache.set(
                        cache_key,
                        result,
                        model="agent1",
                        tags=["blueprint", "agent1", seq_no]
                    )

                task_status.llm_calls_made += result.get('llm_calls', 0)

            # Validate blueprint structure
            if result['success']:
                blueprint_data = result['blueprint_content']
                validation = validate_with_detailed_errors(AutomationBlueprint, blueprint_data)

                if not validation['success']:
                    # Try relaxed parsing
                    task_status.warnings.extend([
                        f"Blueprint validation failed, attempting recovery: {validation['raw_error']}"
                    ])

                    # This would normally trigger a recovery process
                    # For now, we proceed with warnings
                    result['warnings'] = validation['warnings']

            task_status.phase_times['agent1'] = time.time() - phase_start
            return result

        except Exception as e:
            task_status.errors.append(f"Agent 1 execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'suggested_fix': 'Retry with different validation level or check PDF content'
            }

    def _run_agent2_with_recovery(self, seq_no: str, blueprint_result: Dict[str, Any],
                                 task_status: TaskStatus) -> Dict[str, Any]:
        """
        Run Agent 2 with error recovery and validation
        """
        phase_start = time.time()

        try:
            # Get blueprint file path
            artifacts_path = Path(ARTIFACTS_DIR) / seq_no
            blueprint_file = artifacts_path / "blueprint.json"

            if not blueprint_file.exists():
                # Create blueprint file from result
                blueprint_content = blueprint_result.get('blueprint_content', {})
                safe_file_write(blueprint_file, json.dumps(blueprint_content, indent=2))

            # Run Agent 2 with maximum tolerance
            result = run_enhanced_agent2(seq_no, str(blueprint_file))

            task_status.llm_calls_made += result.get('llm_calls', 0)
            task_status.phase_times['agent2'] = time.time() - phase_start

            return result

        except Exception as e:
            task_status.errors.append(f"Agent 2 execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'suggested_fix': 'Check blueprint format and ensure valid JSON structure'
            }

    def _run_agent3_with_recovery(self, seq_no: str, task_status: TaskStatus) -> Dict[str, Any]:
        """
        Run Agent 3 with error recovery and intelligent retry
        """
        phase_start = time.time()

        try:
            # Get intelligent execution configuration
            intelligence_stats = get_execution_intelligence_stats()
            platform_stats = intelligence_stats.get('platforms', {}).get('web', {})

            # Configure execution based on intelligence
            config = {
                'platform': Platform.WEB,
                'timeout_seconds': platform_stats.get('timeout_seconds', 300),
                'retry_attempts': platform_stats.get('retry_attempts', 1),
                'virtual_env': True,
                'enable_monitoring': True
            }

            # Run Agent 3
            result = run_enhanced_agent3(seq_no, config)

            task_status.phase_times['agent3'] = time.time() - phase_start

            # Handle retry logic if needed
            if not result['success'] and config['retry_attempts'] > 0:
                task_status.retry_count += 1
                task_status.warnings.append(f"Execution failed, retrying ({task_status.retry_count}/{config['retry_attempts']})")

                # Simple retry with adjusted timeout
                config['timeout_seconds'] = int(config['timeout_seconds'] * 1.5)
                retry_result = run_enhanced_agent3(seq_no, config)

                if retry_result['success']:
                    result = retry_result
                    result['retry_success'] = True

            return result

        except Exception as e:
            task_status.errors.append(f"Agent 3 execution failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'suggested_fix': 'Check script file exists and system permissions'
            }

    def _handle_pipeline_failure(self, seq_no: str, failed_phase: str, 
                               failure_result: Dict[str, Any], 
                               task_status: TaskStatus) -> Dict[str, Any]:
        """
        Handle pipeline failures with detailed error reporting
        """
        task_status.phase = "failed"
        task_status.suggested_retry = True
        task_status.errors.append(f"Pipeline failed at {failed_phase}: {failure_result.get('error', 'Unknown error')}")

        self._update_task_status(seq_no, task_status)

        return {
            "success": False,
            "seq_no": seq_no,
            "failed_phase": failed_phase,
            "error": failure_result.get('error', 'Unknown error'),
            "task_status": task_status.model_dump(),
            "suggested_actions": [
                f"Review {failed_phase} logs for detailed error information",
                "Check input data format and content",
                "Retry with different validation level",
                "Contact support if issue persists"
            ]
        }

    def _handle_critical_failure(self, seq_no: str, error: str, 
                                task_status: TaskStatus) -> Dict[str, Any]:
        """
        Handle critical pipeline failures
        """
        task_status.phase = "failed"
        task_status.errors.append(f"Critical pipeline failure: {error}")

        self._update_task_status(seq_no, task_status)

        return create_error_response(
            "pipeline_critical_failure",
            error,
            field_path=f"pipeline.{seq_no}",
            suggested_fix="Check system resources and restart pipeline"
        )

    def _update_task_status(self, seq_no: str, task_status: TaskStatus):
        """
        Update and persist task status
        """
        try:
            status_file = Path(ARTIFACTS_DIR) / seq_no / "status.json"
            ensure_directory(status_file.parent)
            safe_file_write(status_file, task_status.model_dump_json(indent=2))

            # Also cache in memory for quick access
            self.execution_log[seq_no] = task_status.model_dump()

        except Exception:
            pass  # Don't fail pipeline if status update fails

    def _record_pipeline_insights(self, seq_no: str, phase_results: Dict[str, Any]):
        """
        Record insights from pipeline execution for future optimization
        """
        try:
            insights = {
                'timestamp': time.time(),
                'success_pattern': {
                    'agent1': phase_results['agent1']['success'],
                    'agent2': phase_results['agent2']['success'], 
                    'agent3': phase_results['agent3']['success']
                },
                'performance': {
                    'agent1_time': phase_results['agent1'].get('execution_time', 0),
                    'agent2_time': phase_results['agent2'].get('execution_time', 0),
                    'agent3_time': phase_results['agent3'].get('execution_time', 0)
                },
                'cache_effectiveness': {
                    'agent1_cache_hit': phase_results['agent1'].get('cache_used', False),
                    'agent2_cache_hit': phase_results['agent2'].get('cache_used', False)
                }
            }

            self.global_insights[seq_no] = insights
            self._save_global_insights()

        except Exception:
            pass  # Don't fail pipeline if insight recording fails

    def get_pipeline_status(self, seq_no: str) -> Dict[str, Any]:
        """
        Get current pipeline status with detailed information
        """
        try:
            status_file = Path(ARTIFACTS_DIR) / seq_no / "status.json"
            if status_file.exists():
                status_data = json.loads(status_file.read_text())
                return {
                    "success": True,
                    "status": status_data,
                    "insights": self.global_insights.get(seq_no, {})
                }
            else:
                return {
                    "success": False,
                    "error": f"Status file not found for task {seq_no}"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to read status: {str(e)}"
            }

    def get_system_health(self) -> Dict[str, Any]:
        """
        Get overall system health and performance metrics
        """
        try:
            # Combine stats from all agents
            agent1_stats = get_agent1_stats()
            agent2_stats = get_agent2_stats()
            agent3_stats = get_execution_intelligence_stats()

            return {
                "system_status": "healthy",
                "manager_stats": {
                    "cache_stats": self.cache.get_stats(),
                    "performance_stats": self.monitor.get_stats(),
                    "active_tasks": len(self.execution_log),
                    "global_insights_count": len(self.global_insights)
                },
                "agent_stats": {
                    "agent1": agent1_stats,
                    "agent2": agent2_stats,
                    "agent3": agent3_stats
                },
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "system_status": "degraded",
                "error": str(e),
                "timestamp": time.time()
            }

# Module-level convenience functions
def clear_all_caches():
    """Clear all agent caches"""
    cache_dir = Path(ARTIFACTS_DIR) / "cache"
    if cache_dir.exists():
        for cache_file in cache_dir.glob("*.db"):
            try:
                cache_file.unlink()
            except Exception:
                pass

def get_comprehensive_stats():
    """Get comprehensive system statistics"""
    manager = EnhancedAgentManager()
    return manager.get_system_health()

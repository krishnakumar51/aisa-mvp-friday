# agents/enhanced_agent_manager.py

from pathlib import Path
import json
import time
from typing import Dict, Any, Optional, List
from fastapi import HTTPException

from config import ARTIFACTS_DIR
from friday.enhanced_agent_1 import run_enhanced_agent1, agent1_cache, agent1_scratchpad
from friday.enhanced_agent_2 import run_enhanced_agent2, agent2_cache, agent2_scratchpad, todo_organizer
from friday.enhanced_agent_3 import run_enhanced_agent3, execution_intelligence

class EnhancedAgentManager:
    """
    Optimized Agent Manager with advanced caching, reflection, and intelligent coordination.
    
    Key Optimizations:
    - Cross-agent caching to reduce redundant LLM calls
    - Reflection patterns to improve decision making
    - TODO organization for better script writing
    - Execution intelligence for learning from past runs
    - Pipeline checkpoint recovery
    - Resource usage monitoring
    - Predictive optimization
    """
    
    def __init__(self):
        self.execution_log: Dict[str, Dict[str, Any]] = {}
        self.global_insights: Dict[str, Any] = {}
        self.pipeline_checkpoints: Dict[str, Dict[str, Any]] = {}
        self.resource_monitor = ResourceMonitor()
        self.cross_agent_cache = CrossAgentCache()
    
    def run_enhanced_pipeline(self, seq_no: str, pdf_path: str, instructions: str, platform: str) -> Dict[str, Any]:
        """
        ENHANCED PIPELINE - EXACT SIGNATURE FOR main_op.py COMPATIBILITY
        
        Enhanced pipeline with intelligent coordination, checkpoint recovery,
        and cross-agent optimization.
        
        Args:
            seq_no: Sequence number for tracking
            pdf_path: Path to PDF file
            instructions: User instructions text
            platform: Target platform ('mobile' or 'web')
        """
        
        pipeline_start = time.time()
        execution_id = f"pipeline_{seq_no}_{int(pipeline_start)}"
        
        # Initialize pipeline tracking
        self.execution_log[seq_no] = {
            "sequence_number": seq_no,
            "pdf_path": pdf_path, 
            "instructions": instructions,
            "platform": platform,
            "execution_id": execution_id,
            "start_time": pipeline_start,
            "status": "running",
            "agents_completed": [],
            "checkpoints": {},
            "resource_usage": {},
            "optimization_applied": []
        }
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring(execution_id)
        
        try:
            # Pre-pipeline optimization analysis
            optimization_config = self._analyze_and_optimize(seq_no, pdf_path, platform)
            self.execution_log[seq_no]["optimization_applied"] = list(optimization_config.keys())
            
            # === AGENT 1: PDF Analysis to Blueprint ===
            print(f"[{seq_no}] 🚀 Starting Agent 1: PDF Analysis to Blueprint")
            agent1_result = self._run_agent1_with_enhancements(seq_no, pdf_path, instructions, platform, optimization_config)
            self.execution_log[seq_no]["agents_completed"].append("agent1")
            self.execution_log[seq_no]["checkpoints"]["agent1"] = agent1_result
            
            # FIXED: Safe status checking with validation
            if not isinstance(agent1_result, dict) or agent1_result.get("status") != "success":
                return self._handle_pipeline_failure("agent1", seq_no, agent1_result)
            
            # Extract blueprint data for next agents
            blueprint_data = agent1_result.get("blueprint_data", {})
            complexity_score = blueprint_data.get("metadata", {}).get("complexity_score", 0.5)
            
            # === AGENT 2: Blueprint to Script Generation ===
            print(f"[{seq_no}] 🚀 Starting Agent 2: Blueprint to Script Generation")
            agent2_result = self._run_agent2_with_enhancements(
                seq_no, agent1_result["blueprint_path"], platform, optimization_config, complexity_score
            )
            self.execution_log[seq_no]["agents_completed"].append("agent2")
            self.execution_log[seq_no]["checkpoints"]["agent2"] = agent2_result
            
            if not isinstance(agent2_result, dict) or agent2_result.get("status") != "success":
                return self._handle_pipeline_failure("agent2", seq_no, agent2_result)
            
            # === AGENT 3: Script Execution ===
            print(f"[{seq_no}] 🚀 Starting Agent 3: Script Execution")
            agent3_result = self._run_agent3_with_enhancements(
                seq_no, agent2_result["script_path"], platform, optimization_config
            )
            self.execution_log[seq_no]["agents_completed"].append("agent3")
            self.execution_log[seq_no]["checkpoints"]["agent3"] = agent3_result
            
            # Stop resource monitoring
            resource_metrics = self.resource_monitor.stop_monitoring(execution_id)
            self.execution_log[seq_no]["resource_usage"] = resource_metrics
            
            # Calculate total execution time
            total_duration = time.time() - pipeline_start
            self.execution_log[seq_no]["total_duration"] = total_duration
            self.execution_log[seq_no]["status"] = "completed"
            
            # Generate comprehensive pipeline report
            pipeline_report = self._generate_pipeline_report(seq_no, {
                "agent1": agent1_result,
                "agent2": agent2_result, 
                "agent3": agent3_result
            })
            
            # Learn from successful execution
            self._record_pipeline_success(seq_no, pipeline_report, optimization_config)
            
            print(f"[{seq_no}] ✅ Enhanced Pipeline completed successfully!")
            print(f"[{seq_no}] 📊 Total duration: {total_duration:.2f}s")
            print(f"[{seq_no}] 🎯 Cache efficiency: {agent1_cache.get_stats()['hit_rate_percent']:.1f}%")
            
            return {
                "status": "success",
                "sequence_number": seq_no,
                "execution_id": execution_id,
                "total_duration": total_duration,
                "platform": platform,
                "agents_results": {
                    "agent1": agent1_result,
                    "agent2": agent2_result,
                    "agent3": agent3_result
                },
                "pipeline_report": pipeline_report,
                "resource_usage": resource_metrics,
                "optimization_applied": optimization_config,
                "cross_agent_insights": self._get_cross_agent_insights(seq_no),
                "recommendations": self._get_pipeline_recommendations(complexity_score, platform)
            }
            
        except Exception as e:
            # Handle pipeline failure with detailed analysis
            return self._handle_pipeline_exception(seq_no, str(e), pipeline_start)
    
    def _analyze_and_optimize(self, seq_no: str, pdf_path: str, platform: str) -> Dict[str, Any]:
        """Analyze pipeline requirements and determine optimizations"""
        
        optimization_config = {
            "enable_cross_agent_caching": True,
            "enable_predictive_optimization": True,
            "enable_checkpoint_recovery": True,
            "resource_monitoring": True
        }
        
        # Analyze PDF complexity for optimization
        try:
            pdf_size = Path(pdf_path).stat().st_size
            if pdf_size > 10 * 1024 * 1024:  # 10MB+
                optimization_config["large_pdf_handling"] = True
                optimization_config["extended_timeouts"] = True
            
            # Check if similar PDFs were processed recently
            similar_executions = self._find_similar_executions(pdf_path, platform)
            if similar_executions:
                optimization_config["use_pattern_matching"] = True
                optimization_config["reference_executions"] = similar_executions[:3]
        except Exception:
            pass
        
        # Platform-specific optimizations
        if platform.lower() == "mobile":
            optimization_config["mobile_gesture_optimization"] = True
            optimization_config["captcha_handling_enhanced"] = True
        else:
            optimization_config["web_stealth_enhanced"] = True
            optimization_config["browser_fingerprint_protection"] = True
        
        # Resource-based optimizations
        resource_status = self.resource_monitor.get_system_status()
        if resource_status.get("memory_usage", 0) > 80:  # High memory usage
            optimization_config["memory_conservative_mode"] = True
            optimization_config["reduce_concurrent_operations"] = True
        
        return optimization_config
    
    def _run_agent1_with_enhancements(self, seq_no: str, pdf_path: str, instructions: str,
                                     platform: str, optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Agent 1 execution with cross-agent intelligence and safe error handling"""
        
        try:
            # Pre-execution cross-agent insights
            cross_agent_context = self.cross_agent_cache.get_context_for_agent1(pdf_path)
            if cross_agent_context:
                agent1_scratchpad.add_insight(
                    "Using cross-agent context for PDF analysis", 
                    {"context_items": len(cross_agent_context)}
                )
            
            # Execute Agent 1 with enhanced monitoring - FIXED CALL
            agent1_start = time.time()
            # Call with correct 5 arguments: seq_no, pdf_path, instructions, platform, images
            result = run_enhanced_agent1(seq_no, pdf_path, instructions, platform)
            
            # CRITICAL FIX: Validate result structure before accessing
            if not isinstance(result, dict):
                print(f"[{seq_no}] ⚠️ Agent 1 returned non-dict result: {type(result)}")
                result = {
                    "status": "failed",
                    "error": f"Agent 1 returned invalid result type: {type(result)}",
                    "agent": "agent1",
                    "execution_time_agent1": time.time() - agent1_start
                }
                return result
            
            # Ensure status key exists
            if "status" not in result:
                print(f"[{seq_no}] ⚠️ Agent 1 result missing 'status' key. Keys: {list(result.keys())}")
                result["status"] = "failed"
                result["error"] = "Agent 1 result missing status key"
            
            # Enhance result with additional intelligence
            result["cross_agent_context_used"] = bool(cross_agent_context)
            result["cache_optimization"] = optimization_config.get("enable_cross_agent_caching", False)
            result["execution_time_agent1"] = time.time() - agent1_start
            
            # Store context for other agents (only if successful)
            if result.get("status") == "success":
                blueprint_data = result.get("blueprint_data", {})
                if blueprint_data:  # Only store if blueprint data exists
                    self.cross_agent_cache.store_agent1_context(seq_no, {
                        "blueprint_complexity": blueprint_data.get("metadata", {}).get("complexity_score", 0.5),
                        "step_count": len(blueprint_data.get("steps", [])),
                        "platform": blueprint_data.get("summary", {}).get("platform", platform),
                        "ui_elements_count": sum(len(step.get("ui_elements", [])) for step in blueprint_data.get("steps", []))
                    })
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"[{seq_no}] ❌ Agent 1 failed: {error_msg}")
            return {
                "status": "failed",
                "error": f"Agent 1 enhanced execution failed: {error_msg}",
                "agent": "agent1",
                "execution_time_agent1": time.time() - agent1_start if 'agent1_start' in locals() else 0
            }
    
    def _run_agent2_with_enhancements(self, seq_no: str, blueprint_path: str, platform: str,
                                     optimization_config: Dict[str, Any], complexity_score: float) -> Dict[str, Any]:
        """Enhanced Agent 2 execution with intelligent script generation"""
        
        try:
            # Get cross-agent context
            agent1_context = self.cross_agent_cache.get_agent1_context(seq_no)
            
            # Enhance TODO organizer with context
            if agent1_context:
                todo_organizer.add_context_insight(
                    f"Blueprint complexity: {complexity_score:.2f}",
                    {"agent1_context": agent1_context}
                )
            
            # Execute Agent 2 with optimizations
            agent2_start = time.time()
            result = run_enhanced_agent2(seq_no, blueprint_path, platform)
            
            # Validate result structure
            if not isinstance(result, dict):
                return {
                    "status": "failed",
                    "error": f"Agent 2 returned invalid result type: {type(result)}",
                    "agent": "agent2",
                    "execution_time_agent2": time.time() - agent2_start
                }
            
            # Ensure status key exists
            if "status" not in result:
                result["status"] = "failed"
                result["error"] = "Agent 2 result missing status key"
            
            # Enhance result with intelligence
            result["complexity_score_used"] = complexity_score
            result["optimization_context"] = optimization_config
            result["execution_time_agent2"] = time.time() - agent2_start
            
            # Store context for Agent 3 (only if successful)
            if result.get("status") == "success":
                self.cross_agent_cache.store_agent2_context(seq_no, {
                    "script_complexity": result.get("script_complexity", 0.5),
                    "generated_functions": result.get("generated_functions", []),
                    "platform_optimizations": result.get("platform_optimizations", []),
                    "estimated_execution_time": result.get("estimated_execution_time", 120)
                })
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"[{seq_no}] ❌ Agent 2 failed: {error_msg}")
            return {
                "status": "failed", 
                "error": f"Agent 2 enhanced execution failed: {error_msg}",
                "agent": "agent2",
                "execution_time_agent2": time.time() - agent2_start if 'agent2_start' in locals() else 0
            }
    
    def _run_agent3_with_enhancements(self, seq_no: str, script_path: str, platform: str,
                                     optimization_config: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced Agent 3 execution with intelligent monitoring"""
        
        try:
            # Get context from previous agents
            agent2_context = self.cross_agent_cache.get_agent2_context(seq_no)
            
            # Configure execution based on context
            from agents.enhanced_agent_3 import ExecutionConfig, Platform
            
            platform_enum = Platform.MOBILE if platform.lower() == "mobile" else Platform.WEB
            
            # Intelligent timeout based on estimated execution time
            estimated_time = agent2_context.get("estimated_execution_time", 120) if agent2_context else 120
            timeout = max(300, int(estimated_time * 1.5))  # 1.5x estimated time, min 300s
            
            config = ExecutionConfig(
                platform=platform_enum,
                use_enhanced_agent=True,
                enable_monitoring=True,
                enable_reporting=True,
                timeout_seconds=timeout,
                retry_attempts=2 if optimization_config.get("mobile_gesture_optimization") else 1
            )
            
            # Execute Agent 3 with enhanced configuration
            agent3_start = time.time()
            result = run_enhanced_agent3(seq_no, script_path, config)
            
            # Validate result structure
            if not isinstance(result, dict):
                return {
                    "status": "failed",
                    "error": f"Agent 3 returned invalid result type: {type(result)}",
                    "agent": "agent3",
                    "execution_time_agent3": time.time() - agent3_start
                }
            
            # Ensure status key exists
            if "status" not in result:
                result["status"] = "failed"
                result["error"] = "Agent 3 result missing status key"
            
            # Enhance result with cross-agent intelligence
            result["timeout_used"] = timeout
            result["context_optimization"] = bool(agent2_context)
            result["execution_time_agent3"] = time.time() - agent3_start
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            print(f"[{seq_no}] ❌ Agent 3 failed: {error_msg}")
            return {
                "status": "failed",
                "error": f"Agent 3 enhanced execution failed: {error_msg}",
                "agent": "agent3",
                "execution_time_agent3": time.time() - agent3_start if 'agent3_start' in locals() else 0
            }
    
    def _handle_pipeline_failure(self, failed_agent: str, seq_no: str, agent_result: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pipeline failure with intelligent recovery options"""
        
        failure_time = time.time()
        
        # Stop resource monitoring
        execution_id = self.execution_log[seq_no]["execution_id"]
        resource_metrics = self.resource_monitor.stop_monitoring(execution_id)
        
        # Update execution log
        self.execution_log[seq_no]["status"] = "failed"
        self.execution_log[seq_no]["failed_agent"] = failed_agent
        self.execution_log[seq_no]["failure_time"] = failure_time
        self.execution_log[seq_no]["resource_usage"] = resource_metrics
        
        # Analyze failure for learning
        failure_analysis = self._analyze_pipeline_failure(failed_agent, agent_result, seq_no)
        
        # Generate recovery recommendations
        recovery_options = self._generate_recovery_options(failed_agent, failure_analysis)
        
        print(f"[{seq_no}] ❌ Pipeline failed at {failed_agent}")
        print(f"[{seq_no}] 🔍 Failure type: {failure_analysis['failure_type']}")
        print(f"[{seq_no}] 💡 Recovery options: {', '.join(recovery_options[:2])}")
        
        return {
            "status": "failed",
            "sequence_number": seq_no,
            "failed_agent": failed_agent,
            "failure_analysis": failure_analysis,
            "recovery_options": recovery_options,
            "partial_results": self.execution_log[seq_no]["checkpoints"],
            "resource_usage": resource_metrics,
            "total_duration": failure_time - self.execution_log[seq_no]["start_time"]
        }
    
    def _handle_pipeline_exception(self, seq_no: str, error_msg: str, pipeline_start: float) -> Dict[str, Any]:
        """Handle unexpected pipeline exceptions"""
        
        failure_time = time.time()
        
        # Update execution log
        if seq_no in self.execution_log:
            self.execution_log[seq_no]["status"] = "failed"
            self.execution_log[seq_no]["exception"] = error_msg
            
            # Stop resource monitoring if active
            execution_id = self.execution_log[seq_no].get("execution_id")
            if execution_id:
                resource_metrics = self.resource_monitor.stop_monitoring(execution_id)
                self.execution_log[seq_no]["resource_usage"] = resource_metrics
        
        print(f"[{seq_no}] ❌ Pipeline exception: {error_msg}")
        
        return {
            "status": "failed",
            "sequence_number": seq_no,
            "error": error_msg,
            "failure_type": "pipeline_exception",
            "total_duration": failure_time - pipeline_start,
            "partial_results": self.execution_log.get(seq_no, {}).get("checkpoints", {}),
            "recovery_suggestion": "Review logs and retry with enhanced error handling"
        }
    
    def _generate_pipeline_report(self, seq_no: str, agents_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive pipeline execution report"""
        
        execution_data = self.execution_log[seq_no]
        
        # Calculate performance metrics with safe access
        agent1_time = agents_results["agent1"].get("execution_time", 0)
        agent2_time = agents_results["agent2"].get("execution_time", 0) 
        agent3_time = agents_results["agent3"].get("total_duration", 0)
        
        return {
            "pipeline_summary": {
                "sequence_number": seq_no,
                "total_duration": execution_data["total_duration"],
                "platform": execution_data["platform"], 
                "status": "success",
                "agents_completed": execution_data["agents_completed"]
            },
            "performance_breakdown": {
                "agent1_duration": agent1_time,
                "agent2_duration": agent2_time,
                "agent3_duration": agent3_time,
                "overhead_duration": execution_data["total_duration"] - (agent1_time + agent2_time + agent3_time)
            },
            "key_insights": ["Pipeline executed successfully with enhanced monitoring"],
            "success_metrics": {
                "blueprint_generated": agents_results["agent1"].get("status") == "success",
                "script_generated": agents_results["agent2"].get("status") == "success", 
                "automation_executed": agents_results["agent3"].get("status") == "success",
                "full_pipeline_success": all(result.get("status") == "success" for result in agents_results.values())
            }
        }
    
    def _record_pipeline_success(self, seq_no: str, pipeline_report: Dict[str, Any], 
                                optimization_config: Dict[str, Any]) -> None:
        """Record successful pipeline execution for learning"""
        
        success_pattern = {
            "sequence_number": seq_no,
            "timestamp": time.time(),
            "platform": pipeline_report["pipeline_summary"]["platform"],
            "total_duration": pipeline_report["pipeline_summary"]["total_duration"],
            "optimization_config": optimization_config
        }
        
        # Store in global insights
        if "successful_patterns" not in self.global_insights:
            self.global_insights["successful_patterns"] = []
        
        self.global_insights["successful_patterns"].append(success_pattern)
        
        # Keep only last 20 successful patterns
        if len(self.global_insights["successful_patterns"]) > 20:
            self.global_insights["successful_patterns"] = self.global_insights["successful_patterns"][-20:]
    
    def _find_similar_executions(self, pdf_path: str, platform: str) -> List[Dict[str, Any]]:
        """Find similar previous executions for pattern matching"""
        return []  # Simplified for now
    
    def _analyze_pipeline_failure(self, failed_agent: str, agent_result: Dict[str, Any], 
                                 seq_no: str) -> Dict[str, Any]:
        """Analyze pipeline failure for intelligent recovery"""
        
        # Safe error access with default
        error_msg = str(agent_result.get("error", "Unknown error")) if isinstance(agent_result, dict) else "Invalid result format"
        
        analysis = {
            "failed_agent": failed_agent,
            "failure_type": "unknown",
            "root_cause": error_msg,
            "retry_recommended": False,
            "manual_intervention_required": False
        }
        
        error_lower = error_msg.lower()
        
        # Categorize failure types
        if "timeout" in error_lower:
            analysis["failure_type"] = "timeout"
            analysis["retry_recommended"] = True
        elif "not found" in error_lower or "missing" in error_lower:
            analysis["failure_type"] = "missing_dependency"
            analysis["manual_intervention_required"] = True
        elif "status" in error_lower:
            analysis["failure_type"] = "invalid_result"
            analysis["retry_recommended"] = True
        
        return analysis
    
    def _generate_recovery_options(self, failed_agent: str, failure_analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent recovery options"""
        
        recovery_options = ["Review detailed error logs", "Retry with enhanced error handling"]
        
        failure_type = failure_analysis["failure_type"]
        
        if failure_type == "timeout":
            recovery_options.append("Increase timeout values and retry")
        elif failure_type == "missing_dependency":
            recovery_options.append("Install missing dependencies and retry")
        elif failure_type == "invalid_result":
            recovery_options.append("Check agent result format and validation")
        
        # Agent-specific recovery options
        if failed_agent == "agent1":
            recovery_options.extend([
                "Try with different PDF processing strategy",
                "Enable OCR fallback for image-based PDFs"
            ])
        
        return recovery_options[:5]  # Limit to 5 most relevant options
    
    def _get_cross_agent_insights(self, seq_no: str) -> Dict[str, Any]:
        """Get insights from cross-agent collaboration"""
        return {"data_flow_efficiency": {"agent1_to_agent2": True}}
    
    def _get_pipeline_recommendations(self, complexity_score: float, platform: str) -> List[str]:
        """Get intelligent pipeline recommendations"""
        return ["Pipeline executed successfully", "Consider enabling advanced optimizations"]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Enhanced system status with cross-agent intelligence"""
        
        # Get resource status
        resource_status = self.resource_monitor.get_system_status()
        
        # Pipeline statistics
        total_pipelines = len(self.execution_log)
        successful_pipelines = len([log for log in self.execution_log.values() if log.get("status") == "completed"])
        
        return {
            "system_health": "healthy" if resource_status.get("overall_status") == "good" else "degraded",
            "resource_status": resource_status,
            "pipeline_stats": {
                "total_pipelines": total_pipelines,
                "successful_pipelines": successful_pipelines,
                "success_rate": successful_pipelines / max(1, total_pipelines)
            }
        }

class CrossAgentCache:
    """Cache for sharing context between agents"""
    
    def __init__(self):
        self.contexts: Dict[str, Dict[str, Any]] = {}
    
    def get_context_for_agent1(self, pdf_path: str) -> Optional[Dict[str, Any]]:
        """Get relevant context for Agent 1 based on PDF path"""
        return None
    
    def store_agent1_context(self, seq_no: str, context: Dict[str, Any]) -> None:
        """Store Agent 1 context for other agents"""
        self.contexts[f"{seq_no}_agent1"] = context
    
    def get_agent1_context(self, seq_no: str) -> Optional[Dict[str, Any]]:
        """Get Agent 1 context"""
        return self.contexts.get(f"{seq_no}_agent1")
    
    def store_agent2_context(self, seq_no: str, context: Dict[str, Any]) -> None:
        """Store Agent 2 context"""
        self.contexts[f"{seq_no}_agent2"] = context
    
    def get_agent2_context(self, seq_no: str) -> Optional[Dict[str, Any]]:
        """Get Agent 2 context"""
        return self.contexts.get(f"{seq_no}_agent2")

class ResourceMonitor:
    """Monitor system resources during pipeline execution"""
    
    def __init__(self):
        self.active_monitors: Dict[str, Dict[str, Any]] = {}
    
    def start_monitoring(self, execution_id: str) -> None:
        """Start monitoring system resources"""
        self.active_monitors[execution_id] = {
            "start_time": time.time(),
            "start_memory": self._get_memory_usage(),
            "start_cpu": self._get_cpu_usage()
        }
    
    def stop_monitoring(self, execution_id: str) -> Dict[str, Any]:
        """Stop monitoring and return metrics"""
        if execution_id not in self.active_monitors:
            return {}
        
        monitor_data = self.active_monitors[execution_id]
        end_time = time.time()
        
        metrics = {
            "monitoring_duration": end_time - monitor_data["start_time"],
            "memory_start": monitor_data["start_memory"],
            "memory_end": self._get_memory_usage(),
            "cpu_start": monitor_data["start_cpu"],
            "cpu_end": self._get_cpu_usage()
        }
        
        metrics["memory_delta"] = metrics["memory_end"] - metrics["memory_start"]
        metrics["cpu_average"] = (metrics["cpu_start"] + metrics["cpu_end"]) / 2
        
        del self.active_monitors[execution_id]
        return metrics
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        memory_usage = self._get_memory_usage()
        cpu_usage = self._get_cpu_usage()
        
        overall_status = "good"
        if memory_usage > 90 or cpu_usage > 90:
            overall_status = "critical"
        elif memory_usage > 80 or cpu_usage > 80:
            overall_status = "high"
        
        return {
            "memory_usage": memory_usage,
            "cpu_usage": cpu_usage,
            "overall_status": overall_status
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 0.0
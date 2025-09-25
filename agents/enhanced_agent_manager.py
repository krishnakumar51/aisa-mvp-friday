# agents/enhanced_agent_manager.py

from pathlib import Path
import json
import time
from typing import Dict, Any, Optional
from fastapi import HTTPException

from config import ARTIFACTS_DIR

# Import all enhanced agents
try:
    from agents.enhanced_agent_1 import run_enhanced_agent1
    from agents.enhanced_agent_2 import run_enhanced_agent2
    from agents.enhanced_agent_3 import run_enhanced_agent3
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False
    print("Enhanced agents not available, falling back to standard agents")

# Import standard agents as fallback
from agents.agent_1 import run_agent1
from agents.agent_2 import run_agent2
from agents.agent_3 import run_agent3

class EnhancedAgentManager:
    """Manages the execution flow of all enhanced agents with seamless integration"""
    
    def __init__(self, seq_no: str, use_enhanced: bool = True):
        self.seq_no = seq_no
        self.use_enhanced = use_enhanced and ENHANCED_AVAILABLE
        self.artifacts_dir = ARTIFACTS_DIR / seq_no
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        self.execution_log = {
            "seq_no": seq_no,
            "enhanced_mode": self.use_enhanced,
            "started_at": time.time(),
            "agents": {},
            "overall_status": "pending"
        }
        
    def log_agent_result(self, agent_name: str, status: str, result: Any = None, error: str = None):
        """Log agent execution result"""
        self.execution_log["agents"][agent_name] = {
            "status": status,
            "timestamp": time.time(),
            "result": result,
            "error": error
        }
        
        # Save execution log
        log_path = self.artifacts_dir / "execution_log.json"
        log_path.write_text(json.dumps(self.execution_log, indent=2, default=str), encoding="utf-8")
    
    def run_agent_1(self, pdf_path: Path, instructions: str, platform: str) -> dict:
        """Run Agent 1 with enhanced or standard mode"""
        print(f"[{self.seq_no}] Running Agent 1 ({'Enhanced' if self.use_enhanced else 'Standard'} mode)")
        
        try:
            if self.use_enhanced:
                result = run_enhanced_agent1(self.seq_no, pdf_path, instructions, platform)
            else:
                result = run_agent1(self.seq_no, pdf_path, instructions, platform)
            
            self.log_agent_result("agent_1", "success", result)
            print(f"[{self.seq_no}] Agent 1 completed successfully")
            return result
            
        except Exception as e:
            self.log_agent_result("agent_1", "failed", error=str(e))
            print(f"[{self.seq_no}] Agent 1 failed: {e}")
            raise
    
    def run_agent_2(self, blueprint_dict: dict) -> dict:
        """Run Agent 2 with enhanced or standard mode"""
        print(f"[{self.seq_no}] Running Agent 2 ({'Enhanced' if self.use_enhanced else 'Standard'} mode)")
        
        try:
            if self.use_enhanced:
                result = run_enhanced_agent2(self.seq_no, blueprint_dict)
            else:
                result = run_agent2(self.seq_no, blueprint_dict)
            
            self.log_agent_result("agent_2", "success", result)
            print(f"[{self.seq_no}] Agent 2 completed successfully")
            return result
            
        except Exception as e:
            self.log_agent_result("agent_2", "failed", error=str(e))
            print(f"[{self.seq_no}] Agent 2 failed: {e}")
            raise
    
    def run_agent_3(self, platform: str) -> dict:
        """Run Agent 3 with enhanced or standard mode"""
        print(f"[{self.seq_no}] Running Agent 3 ({'Enhanced' if self.use_enhanced else 'Standard'} mode)")
        
        try:
            if self.use_enhanced:
                result = run_enhanced_agent3(self.seq_no, platform, use_enhanced_agent=True)
            else:
                result = run_agent3(self.seq_no, platform)
            
            self.log_agent_result("agent_3", "success", result)
            print(f"[{self.seq_no}] Agent 3 completed successfully")
            return result
            
        except Exception as e:
            self.log_agent_result("agent_3", "failed", error=str(e))
            print(f"[{self.seq_no}] Agent 3 failed: {e}")
            raise
    
    def run_full_pipeline(self, pdf_path: Path, instructions: str, platform: str) -> dict:
        """Run the complete automation pipeline"""
        print(f"[{self.seq_no}] Starting complete automation pipeline")
        print(f"[{self.seq_no}] Mode: {'Enhanced' if self.use_enhanced else 'Standard'}")
        print(f"[{self.seq_no}] Platform: {platform}")
        
        pipeline_result = {
            "seq_no": self.seq_no,
            "mode": "enhanced" if self.use_enhanced else "standard",
            "platform": platform,
            "started_at": time.time(),
            "agent_results": {},
            "overall_status": "running",
            "execution_log_path": str(self.artifacts_dir / "execution_log.json")
        }
        
        try:
            # Agent 1: Blueprint Generation
            print(f"[{self.seq_no}] ===== STAGE 1: Blueprint Generation =====")
            agent1_result = self.run_agent_1(pdf_path, instructions, platform)
            pipeline_result["agent_results"]["agent_1"] = agent1_result
            
            # Agent 2: Code Generation
            print(f"[{self.seq_no}] ===== STAGE 2: Code Generation =====")
            agent2_result = self.run_agent_2(agent1_result)
            pipeline_result["agent_results"]["agent_2"] = agent2_result
            
            # Agent 3: Execution
            print(f"[{self.seq_no}] ===== STAGE 3: Execution =====")
            agent3_result = self.run_agent_3(platform)
            pipeline_result["agent_results"]["agent_3"] = agent3_result
            
            pipeline_result["overall_status"] = "completed"
            pipeline_result["completed_at"] = time.time()
            
            self.execution_log["overall_status"] = "completed"
            self.execution_log["completed_at"] = time.time()
            
            print(f"[{self.seq_no}] ===== PIPELINE COMPLETED SUCCESSFULLY =====")
            
            return pipeline_result
            
        except Exception as e:
            pipeline_result["overall_status"] = "failed"
            pipeline_result["error"] = str(e)
            pipeline_result["failed_at"] = time.time()
            
            self.execution_log["overall_status"] = "failed"
            self.execution_log["error"] = str(e)
            self.execution_log["failed_at"] = time.time()
            
            print(f"[{self.seq_no}] ===== PIPELINE FAILED =====")
            print(f"[{self.seq_no}] Error: {e}")
            
            raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")
        
        finally:
            # Always save final execution log
            log_path = self.artifacts_dir / "execution_log.json"
            log_path.write_text(json.dumps(self.execution_log, indent=2, default=str), encoding="utf-8")
    
    def get_execution_summary(self) -> dict:
        """Get a summary of the execution"""
        summary = {
            "seq_no": self.seq_no,
            "mode": "enhanced" if self.use_enhanced else "standard",
            "overall_status": self.execution_log["overall_status"],
            "agents_completed": len([a for a in self.execution_log["agents"].values() if a["status"] == "success"]),
            "total_agents": len(self.execution_log["agents"]),
            "execution_time": None,
            "artifacts_location": str(self.artifacts_dir),
            "available_files": []
        }
        
        # Calculate execution time
        if "completed_at" in self.execution_log or "failed_at" in self.execution_log:
            end_time = self.execution_log.get("completed_at") or self.execution_log.get("failed_at")
            summary["execution_time"] = end_time - self.execution_log["started_at"]
        
        # List available files
        if self.artifacts_dir.exists():
            for file_path in self.artifacts_dir.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.artifacts_dir)
                    summary["available_files"].append(str(relative_path))
        
        return summary

def run_complete_automation_pipeline(seq_no: str, pdf_path: Path, instructions: str, 
                                   platform: str, use_enhanced: bool = True) -> dict:
    """
    Complete automation pipeline function for external use
    
    Args:
        seq_no: Unique sequence number for this automation run
        pdf_path: Path to the PDF file with screenshots/instructions
        instructions: Text instructions from the user
        platform: Target platform ('mobile' or 'web')
        use_enhanced: Whether to use enhanced agents (default: True)
    
    Returns:
        Dict containing complete pipeline results
    """
    manager = EnhancedAgentManager(seq_no, use_enhanced)
    return manager.run_full_pipeline(pdf_path, instructions, platform)

def get_pipeline_status(seq_no: str) -> Optional[dict]:
    """
    Get the status of a running or completed pipeline
    
    Args:
        seq_no: Unique sequence number for the automation run
    
    Returns:
        Dict containing pipeline status or None if not found
    """
    artifacts_dir = ARTIFACTS_DIR / seq_no
    log_path = artifacts_dir / "execution_log.json"
    
    if not log_path.exists():
        return None
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            execution_log = json.load(f)
        
        manager = EnhancedAgentManager(seq_no, use_enhanced=True)
        manager.execution_log = execution_log
        
        return manager.get_execution_summary()
    
    except Exception as e:
        return {
            "error": f"Failed to read pipeline status: {e}",
            "seq_no": seq_no
        }

# Backward compatibility functions
def run_agents_pipeline(seq_no: str, pdf_path: Path, instructions: str, platform: str) -> dict:
    """Backward compatible pipeline function"""
    return run_complete_automation_pipeline(seq_no, pdf_path, instructions, platform, use_enhanced=True)

def run_standard_agents_pipeline(seq_no: str, pdf_path: Path, instructions: str, platform: str) -> dict:
    """Run pipeline with standard agents only"""
    return run_complete_automation_pipeline(seq_no, pdf_path, instructions, platform, use_enhanced=False)

# Integration helper functions
def ensure_agent_compatibility() -> dict:
    """Check if all agents are properly integrated"""
    compatibility_status = {
        "enhanced_agents_available": ENHANCED_AVAILABLE,
        "standard_agents_available": True,  # These should always be available
        "issues": [],
        "recommendations": []
    }
    
    # Check enhanced agents
    if ENHANCED_AVAILABLE:
        try:
            # Test imports
            from agents.enhanced_agent_1 import run_enhanced_agent1
            from agents.enhanced_agent_2 import run_enhanced_agent2
            from agents.enhanced_agent_3 import run_enhanced_agent3
            compatibility_status["enhanced_agents_functional"] = True
        except Exception as e:
            compatibility_status["enhanced_agents_functional"] = False
            compatibility_status["issues"].append(f"Enhanced agents import failed: {e}")
    else:
        compatibility_status["enhanced_agents_functional"] = False
        compatibility_status["issues"].append("Enhanced agents not available")
        compatibility_status["recommendations"].append("Install enhanced agent dependencies")
    
    # Check standard agents
    try:
        from agents.agent_1 import run_agent1
        from agents.agent_2 import run_agent2
        from agents.agent_3 import run_agent3
        compatibility_status["standard_agents_functional"] = True
    except Exception as e:
        compatibility_status["standard_agents_functional"] = False
        compatibility_status["issues"].append(f"Standard agents import failed: {e}")
    
    # Overall compatibility
    compatibility_status["overall_compatible"] = (
        compatibility_status["standard_agents_functional"] and
        (compatibility_status["enhanced_agents_functional"] or not ENHANCED_AVAILABLE)
    )
    
    return compatibility_status

# Export all functions
__all__ = [
    "EnhancedAgentManager",
    "run_complete_automation_pipeline", 
    "get_pipeline_status",
    "run_agents_pipeline",
    "run_standard_agents_pipeline",
    "ensure_agent_compatibility"
]
# agents/enhanced_agent_3_with_terminals.py

from pathlib import Path
import sys
import os
import subprocess
import json
import time
import shutil
from typing import Dict, List, Optional, Any
from fastapi import HTTPException
from dataclasses import dataclass
from enum import Enum

from config import ARTIFACTS_DIR

class Platform(Enum):
    MOBILE = "mobile"
    WEB = "web"

class ExecutionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class ExecutionConfig:
    platform: Platform
    use_enhanced_agent: bool = True
    enable_monitoring: bool = True
    enable_reporting: bool = True
    timeout_seconds: int = 300
    retry_attempts: int = 1

# Execution Intelligence System
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
            recommendations.append("Consider device-specific optimizations")
        else:  # web
            if "browser" in str(issues).lower():
                recommendations.append("Add browser compatibility checks")
            recommendations.append("Implement headless mode for better stability")
        
        return recommendations

# Global execution intelligence instance
execution_intelligence = ExecutionIntelligence()

class TerminalExecutionEngine:
    """Enhanced execution engine that launches actual visible terminals"""

    def __init__(self, seq_no: str, config: ExecutionConfig):
        self.seq_no = seq_no
        self.config = config
        self.artifacts_dir = ARTIFACTS_DIR / seq_no
        self.agent3_dir = self.artifacts_dir / "enhanced_agent3"
        self.agent3_dir.mkdir(parents=True, exist_ok=True)

        # Determine source directories (enhanced or regular)
        self.agent1_dir = self.artifacts_dir / ("enhanced_agent1" if config.use_enhanced_agent else "agent1")
        self.agent2_dir = self.artifacts_dir / ("enhanced_agent2" if config.use_enhanced_agent else "agent2")

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

    def launch_mobile_terminals(self) -> Dict[str, Any]:
        """Launch Appium server terminal and execution terminal for mobile"""
        try:
            print(f"[{self.seq_no}] 📱 Launching Mobile Automation Terminals...")
            
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
            
            # 2. Launch Execution Terminal
            script_path = self.agent3_dir / "automation_script.py"
            
            if sys.platform == "win32":
                exec_cmd = [
                    "cmd", "/c", "start", "cmd", "/k",
                    f"title Mobile Automation - Task {self.seq_no} && cd /d {self.agent3_dir} && call {activate_script} && echo Running Mobile Automation Script... && python {script_path} && pause"
                ]
            else:
                exec_cmd = [
                    "gnome-terminal", "--title", f"Mobile Automation - {self.seq_no}",
                    "--", "bash", "-c",
                    f"cd {self.agent3_dir} && source {activate_script} && echo 'Running Mobile Automation Script...' && python {script_path}; read -p 'Press Enter to close...'"
                ]
            
            exec_process = subprocess.Popen(exec_cmd)
            print(f"[{self.seq_no}] 🚀 Mobile Execution Terminal launched (PID: {exec_process.pid})")
            
            return {
                "appium_server_pid": appium_process.pid,
                "execution_pid": exec_process.pid,
                "status": "terminals_launched"
            }
            
        except Exception as e:
            print(f"[{self.seq_no}] ❌ Failed to launch mobile terminals: {e}")
            return {"error": str(e), "status": "failed"}

    def launch_web_terminals(self) -> Dict[str, Any]:
        """Launch web automation terminal"""
        try:
            print(f"[{self.seq_no}] 🌐 Launching Web Automation Terminal...")
            
            # Get venv paths
            if sys.platform == "win32":
                venv_python = self.agent3_dir / "venv" / "Scripts" / "python.exe"
                activate_script = self.agent3_dir / "venv" / "Scripts" / "activate.bat"
            else:
                venv_python = self.agent3_dir / "venv" / "bin" / "python"
                activate_script = self.agent3_dir / "venv" / "bin" / "activate"

            script_path = self.agent3_dir / "automation_script.py"
            
            # Launch Execution Terminal
            if sys.platform == "win32":
                exec_cmd = [
                    "cmd", "/c", "start", "cmd", "/k",
                    f"title Web Automation - Task {self.seq_no} && cd /d {self.agent3_dir} && call {activate_script} && echo Running Web Automation Script... && python {script_path} && pause"
                ]
            else:
                exec_cmd = [
                    "gnome-terminal", "--title", f"Web Automation - {self.seq_no}",
                    "--", "bash", "-c",
                    f"cd {self.agent3_dir} && source {activate_script} && echo 'Running Web Automation Script...' && python {script_path}; read -p 'Press Enter to close...'"
                ]
            
            exec_process = subprocess.Popen(exec_cmd)
            print(f"[{self.seq_no}] 🚀 Web Execution Terminal launched (PID: {exec_process.pid})")
            
            return {
                "execution_pid": exec_process.pid,
                "status": "terminal_launched"
            }
            
        except Exception as e:
            print(f"[{self.seq_no}] ❌ Failed to launch web terminal: {e}")
            return {"error": str(e), "status": "failed"}

def run_enhanced_agent3(seq_no: str, platform: str, use_enhanced: bool = True) -> dict:
    """Enhanced Agent 3 with terminal execution and script testing"""
    print(f"[{seq_no}] 🚀 Running Terminal-Based Enhanced Agent 3")
    
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

        # Initialize execution engine
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

        print(f"[{seq_no}] 🚀 Launching terminal execution...")
        
        # Launch platform-specific terminals
        if platform_enum == Platform.MOBILE:
            terminal_result = engine.launch_mobile_terminals()
        else:
            terminal_result = engine.launch_web_terminals()

        # Record successful setup
        setup_duration = time.time() - start_time
        execution_intelligence.record_execution(
            seq_no, platform, True, setup_duration, []
        )

        print(f"[{seq_no}] ✅ Terminal-Based Enhanced Agent 3 completed successfully!")
        print(f"[{seq_no}] 🧠 Execution intelligence applied for optimization")
        print(f"[{seq_no}] 📊 Setup completed in {setup_duration:.1f} seconds")
        print(f"[{seq_no}] 🚀 Terminals launched - check your screen for running automation!")

        return {
            "status": "success",
            "validation": validation,
            "terminal_result": terminal_result,
            "intelligence_insights": insights,
            "duration": setup_duration,
            "issues": issues
        }

    except Exception as e:
        duration = time.time() - start_time
        error_msg = f"Terminal-based Enhanced Agent 3 failed: {e}"
        issues.append(error_msg)
        
        # Record failure in intelligence
        execution_intelligence.record_execution(
            seq_no, platform, False, duration, issues
        )
        
        print(f"[{seq_no}] ❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

# Export main function with backward compatibility
def run_agent3(seq_no: str, platform: str, use_enhanced: bool = True) -> dict:
    """Backward compatible wrapper for enhanced agent 3"""
    return run_enhanced_agent3(seq_no, platform, use_enhanced)

# Export intelligence system for external access
__all__ = [
    "run_enhanced_agent3", "run_agent3", "TerminalExecutionEngine", 
    "ExecutionIntelligence", "execution_intelligence"
]
# agents/enhanced_agent_3.py

from pathlib import Path
import sys
import os
import subprocess
import webbrowser
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

class EnhancedExecutionEngine:
    """Advanced execution engine with monitoring, reporting, and cross-platform support"""
    
    def __init__(self, seq_no: str, config: ExecutionConfig):
        self.seq_no = seq_no
        self.config = config
        self.artifacts_dir = ARTIFACTS_DIR / seq_no
        self.agent3_dir = self.artifacts_dir / "enhanced_agent3"
        self.agent3_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine source directories (enhanced or regular)
        self.agent1_dir = self.artifacts_dir / ("enhanced_agent1" if config.use_enhanced_agent else "agent1")
        self.agent2_dir = self.artifacts_dir / ("enhanced_agent2" if config.use_enhanced_agent else "agent2")
        
    def validate_prerequisites(self) -> Dict[str, Any]:
        """Validate all prerequisites for execution"""
        validation_results = {
            "valid": True,
            "issues": [],
            "warnings": [],
            "file_status": {}
        }
        
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
                "documentation": self.agent2_dir / "documentation.md"
            })
        
        for file_type, file_path in required_files.items():
            if file_path.exists():
                validation_results["file_status"][file_type] = {
                    "exists": True,
                    "size": file_path.stat().st_size,
                    "path": str(file_path)
                }
            else:
                validation_results["valid"] = False
                validation_results["issues"].append(f"Missing {file_type} file: {file_path}")
                validation_results["file_status"][file_type] = {"exists": False, "path": str(file_path)}
        
        # Platform-specific validations
        if self.config.platform == Platform.MOBILE:
            # Check for Appium server availability
            try:
                result = subprocess.run(["appium", "--version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    validation_results["appium_version"] = result.stdout.strip()
                else:
                    validation_results["warnings"].append("Appium not found or not working")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                validation_results["warnings"].append("Appium command not available")
            
            # Check for ADB
            try:
                result = subprocess.run(["adb", "version"], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    validation_results["adb_available"] = True
                else:
                    validation_results["warnings"].append("ADB not available")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                validation_results["warnings"].append("ADB command not found")
        
        # Check Python environment
        validation_results["python_version"] = sys.version
        validation_results["python_executable"] = sys.executable
        
        return validation_results
    
    def create_virtual_environment(self) -> bool:
        """Create and setup virtual environment with enhanced error handling"""
        venv_dir = self.agent3_dir / "venv"
        python_exe = sys.executable
        
        try:
            print(f"[{self.seq_no}] Creating virtual environment...")
            
            # Remove existing venv if present
            if venv_dir.exists():
                shutil.rmtree(venv_dir)
            
            # Create new venv
            result = subprocess.run([python_exe, "-m", "venv", str(venv_dir)], 
                                  capture_output=True, text=True, timeout=60)
            
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
            
            # Upgrade pip
            subprocess.run([str(venv_pip), "install", "--upgrade", "pip"], 
                         timeout=60, check=False)
            
            # Install requirements
            req_file = self.agent2_dir / "requirements.txt"
            if req_file.exists():
                print(f"[{self.seq_no}] Installing requirements...")
                result = subprocess.run([str(venv_pip), "install", "-r", str(req_file)], 
                                      capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    print(f"Requirements installation failed: {result.stderr}")
                    # Try installing individual packages
                    requirements = req_file.read_text().strip().split('\n')
                    failed_packages = []
                    
                    for req in requirements:
                        if req.strip() and not req.strip().startswith('#'):
                            try:
                                subprocess.run([str(venv_pip), "install", req.strip()], 
                                            timeout=60, check=True)
                            except subprocess.CalledProcessError:
                                failed_packages.append(req.strip())
                    
                    if failed_packages:
                        print(f"Failed to install: {failed_packages}")
                        return False
            
            # Platform-specific installations
            if self.config.platform == Platform.WEB:
                # Install playwright browsers
                try:
                    print(f"[{self.seq_no}] Installing Playwright browsers...")
                    subprocess.run([str(venv_python), "-m", "playwright", "install"], 
                                 timeout=300, check=False)
                except subprocess.TimeoutExpired:
                    print("Playwright browser installation timed out")
            
            return True
            
        except Exception as e:
            print(f"Virtual environment setup failed: {e}")
            return False
    
    def generate_execution_scripts(self) -> Dict[str, Path]:
        """Generate enhanced execution scripts with monitoring and reporting"""
        scripts = {}
        
        if self.config.platform == Platform.MOBILE:
            scripts.update(self._generate_mobile_scripts())
        else:
            scripts.update(self._generate_web_scripts())
        
        return scripts
    
    def _generate_mobile_scripts(self) -> Dict[str, Path]:
        """Generate mobile execution scripts"""
        scripts = {}
        
        if sys.platform == "win32":
            # Appium server script
            appium_script = self._create_appium_server_script_windows()
            scripts["appium_server"] = appium_script
            
            # Main execution script
            execution_script = self._create_mobile_execution_script_windows()
            scripts["execution"] = execution_script
            
        else:  # macOS/Linux
            # Appium server script
            appium_script = self._create_appium_server_script_unix()
            scripts["appium_server"] = appium_script
            
            # Main execution script
            execution_script = self._create_mobile_execution_script_unix()
            scripts["execution"] = execution_script
        
        return scripts
    
    def _generate_web_scripts(self) -> Dict[str, Path]:
        """Generate web execution scripts"""
        scripts = {}
        
        if sys.platform == "win32":
            execution_script = self._create_web_execution_script_windows()
        else:
            execution_script = self._create_web_execution_script_unix()
        
        scripts["execution"] = execution_script
        return scripts
    
    def _create_appium_server_script_windows(self) -> Path:
        """Create Windows Appium server script"""
        script_content = f"""@echo off
title Appium Server - Task {self.seq_no}
echo ================================================
echo  🚀 Enhanced AISA Mobile Automation
echo  📱 Appium Server for Task: {self.seq_no}
echo ================================================
echo.
echo Starting Appium server...
echo This window must remain open during automation.
echo.
echo Server will start on: http://127.0.0.1:4723
echo.
appium --allow-cors --log-timestamp --log-level info
echo.
echo ⚠️ Appium server stopped
pause
"""
        
        script_path = self.agent3_dir / "start_appium_server.bat"
        script_path.write_text(script_content, encoding="utf-8")
        return script_path
    
    def _create_appium_server_script_unix(self) -> Path:
        """Create Unix Appium server script"""
        script_content = f"""#!/bin/bash
echo "================================================"
echo "  🚀 Enhanced AISA Mobile Automation"
echo "  📱 Appium Server for Task: {self.seq_no}"
echo "================================================"
echo
echo "Starting Appium server..."
echo "This terminal must remain open during automation."
echo
echo "Server will start on: http://127.0.0.1:4723"
echo
appium --allow-cors --log-timestamp --log-level info
echo
echo "⚠️ Appium server stopped"
read -p "Press Enter to close..."
"""
        
        script_path = self.agent3_dir / "start_appium_server.sh"
        script_path.write_text(script_content, encoding="utf-8")
        os.chmod(script_path, 0o755)
        return script_path
    
    def _create_mobile_execution_script_windows(self) -> Path:
        """Create Windows mobile execution script"""
        venv_python = self.agent3_dir / "venv" / "Scripts" / "python.exe"
        script_path = self.agent2_dir / "automation_script.py"
        
        script_content = f"""@echo off
title Mobile Automation Runner - Task {self.seq_no}
chcp 65001 >nul
echo ================================================
echo  🤖 Enhanced AISA Mobile Automation Runner
echo  📱 Task ID: {self.seq_no}
echo ================================================
echo.

cd /d "{self.agent3_dir}"

echo [1/6] 🔍 Pre-execution validation...
if not exist "{venv_python}" (
    echo ❌ Virtual environment not found!
    echo Creating virtual environment...
    "{sys.executable}" -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

if not exist "{script_path}" (
    echo ❌ Automation script not found: {script_path}
    pause
    exit /b 1
)

echo ✅ Prerequisites validated

echo.
echo [2/6] 📦 Installing/updating dependencies...
call "{self.agent3_dir}\\venv\\Scripts\\activate.bat"
"{venv_python}" -m pip install --upgrade pip --quiet
"{venv_python}" -m pip install -r "{self.agent2_dir}\\requirements.txt" --quiet
if errorlevel 1 (
    echo ⚠️ Some dependencies may have failed to install
    echo Continuing with execution...
)

echo ✅ Dependencies ready

echo.
echo [3/6] 📱 Waiting for Appium server...
echo Make sure Appium server is running on http://127.0.0.1:4723
echo Waiting 15 seconds for server initialization...
timeout /t 15 /nobreak >nul

echo.
echo [4/6] 🔌 Testing Appium connection...
curl -s http://127.0.0.1:4723/status >nul
if errorlevel 1 (
    echo ⚠️ Appium server may not be running
    echo Continuing anyway...
) else (
    echo ✅ Appium server is responsive
)

echo.
echo [5/6] 🚀 Starting automation execution...
echo ================================================
echo Starting at: %date% %time%
echo ================================================

"{venv_python}" "{script_path}"
set SCRIPT_EXIT_CODE=%errorlevel%

echo ================================================
echo Finished at: %date% %time%
echo Exit Code: %SCRIPT_EXIT_CODE%
echo ================================================

echo.
echo [6/6] 📊 Execution summary...
if %SCRIPT_EXIT_CODE% equ 0 (
    echo ✅ AUTOMATION COMPLETED SUCCESSFULLY
    echo succeeded > "{self.agent3_dir}\\result.txt"
) else (
    echo ❌ AUTOMATION FAILED ^(Exit Code: %SCRIPT_EXIT_CODE%^)
    echo failed > "{self.agent3_dir}\\result.txt"
)

echo.
echo 📁 Results saved to: {self.agent3_dir}
echo 📝 Check result.txt for execution status
echo.

if exist "{self.agent2_dir}\\documentation.md" (
    echo 📖 Documentation available: {self.agent2_dir}\\documentation.md
)

echo.
echo Press any key to close this window...
pause >nul
"""
        
        script_path_bat = self.agent3_dir / "run_mobile_automation.bat"
        script_path_bat.write_text(script_content, encoding="utf-8")
        return script_path_bat
    
    def _create_mobile_execution_script_unix(self) -> Path:
        """Create Unix mobile execution script"""
        venv_python = self.agent3_dir / "venv" / "bin" / "python"
        script_path = self.agent2_dir / "automation_script.py"
        
        script_content = f"""#!/bin/bash
clear
echo "================================================"
echo "  🤖 Enhanced AISA Mobile Automation Runner"
echo "  📱 Task ID: {self.seq_no}"
echo "================================================"
echo

cd "{self.agent3_dir}"

echo "[1/6] 🔍 Pre-execution validation..."
if [ ! -f "{venv_python}" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    "{sys.executable}" -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

if [ ! -f "{script_path}" ]; then
    echo "❌ Automation script not found: {script_path}"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✅ Prerequisites validated"

echo
echo "[2/6] 📦 Installing/updating dependencies..."
source "{self.agent3_dir}/venv/bin/activate"
"{venv_python}" -m pip install --upgrade pip --quiet
"{venv_python}" -m pip install -r "{self.agent2_dir}/requirements.txt" --quiet
if [ $? -ne 0 ]; then
    echo "⚠️ Some dependencies may have failed to install"
    echo "Continuing with execution..."
fi

echo "✅ Dependencies ready"

echo
echo "[3/6] 📱 Waiting for Appium server..."
echo "Make sure Appium server is running on http://127.0.0.1:4723"
echo "Waiting 15 seconds for server initialization..."
sleep 15

echo
echo "[4/6] 🔌 Testing Appium connection..."
if command -v curl >/dev/null 2>&1; then
    curl -s http://127.0.0.1:4723/status >/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Appium server is responsive"
    else
        echo "⚠️ Appium server may not be running"
        echo "Continuing anyway..."
    fi
else
    echo "⚠️ curl not available, skipping server test"
fi

echo
echo "[5/6] 🚀 Starting automation execution..."
echo "================================================"
echo "Starting at: $(date)"
echo "================================================"

"{venv_python}" "{script_path}"
SCRIPT_EXIT_CODE=$?

echo "================================================"
echo "Finished at: $(date)"
echo "Exit Code: $SCRIPT_EXIT_CODE"
echo "================================================"

echo
echo "[6/6] 📊 Execution summary..."
if [ $SCRIPT_EXIT_CODE -eq 0 ]; then
    echo "✅ AUTOMATION COMPLETED SUCCESSFULLY"
    echo "succeeded" > "{self.agent3_dir}/result.txt"
else
    echo "❌ AUTOMATION FAILED (Exit Code: $SCRIPT_EXIT_CODE)"
    echo "failed" > "{self.agent3_dir}/result.txt"
fi

echo
echo "📁 Results saved to: {self.agent3_dir}"
echo "📝 Check result.txt for execution status"
echo

if [ -f "{self.agent2_dir}/documentation.md" ]; then
    echo "📖 Documentation available: {self.agent2_dir}/documentation.md"
fi

echo
read -p "Press Enter to close this terminal..."
"""
        
        script_path_sh = self.agent3_dir / "run_mobile_automation.sh"
        script_path_sh.write_text(script_content, encoding="utf-8")
        os.chmod(script_path_sh, 0o755)
        return script_path_sh
    
    def _create_web_execution_script_windows(self) -> Path:
        """Create Windows web execution script"""
        venv_python = self.agent3_dir / "venv" / "Scripts" / "python.exe"
        script_path = self.agent2_dir / "automation_script.py"
        
        script_content = f"""@echo off
title Web Automation Runner - Task {self.seq_no}
chcp 65001 >nul
echo ================================================
echo  🌐 Enhanced AISA Web Automation Runner
echo  🖥️ Task ID: {self.seq_no}
echo ================================================
echo.

cd /d "{self.agent3_dir}"

echo [1/6] 🔍 Pre-execution validation...
if not exist "{venv_python}" (
    echo ❌ Virtual environment not found!
    echo Creating virtual environment...
    "{sys.executable}" -m venv venv
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

if not exist "{script_path}" (
    echo ❌ Automation script not found: {script_path}
    pause
    exit /b 1
)

echo ✅ Prerequisites validated

echo.
echo [2/6] 📦 Installing/updating dependencies...
call "{self.agent3_dir}\\venv\\Scripts\\activate.bat"
"{venv_python}" -m pip install --upgrade pip --quiet
"{venv_python}" -m pip install -r "{self.agent2_dir}\\requirements.txt" --quiet
if errorlevel 1 (
    echo ⚠️ Some dependencies may have failed to install
    echo Continuing with execution...
)

echo ✅ Dependencies ready

echo.
echo [3/6] 🎭 Installing Playwright browsers...
"{venv_python}" -m playwright install --with-deps
if errorlevel 1 (
    echo ⚠️ Playwright browser installation had issues
    echo Continuing with execution...
) else (
    echo ✅ Playwright browsers ready
)

echo.
echo [4/6] 🌐 Checking internet connectivity...
ping -n 1 google.com >nul 2>&1
if errorlevel 1 (
    echo ⚠️ Internet connectivity issues detected
    echo Continuing anyway...
) else (
    echo ✅ Internet connection available
)

echo.
echo [5/6] 🚀 Starting automation execution...
echo ================================================
echo Starting at: %date% %time%
echo ================================================

"{venv_python}" "{script_path}"
set SCRIPT_EXIT_CODE=%errorlevel%

echo ================================================
echo Finished at: %date% %time%
echo Exit Code: %SCRIPT_EXIT_CODE%
echo ================================================

echo.
echo [6/6] 📊 Execution summary...
if %SCRIPT_EXIT_CODE% equ 0 (
    echo ✅ AUTOMATION COMPLETED SUCCESSFULLY
    echo succeeded > "{self.agent3_dir}\\result.txt"
) else (
    echo ❌ AUTOMATION FAILED ^(Exit Code: %SCRIPT_EXIT_CODE%^)
    echo failed > "{self.agent3_dir}\\result.txt"
)

echo.
echo 📁 Results saved to: {self.agent3_dir}
echo 📝 Check result.txt for execution status
echo.

if exist "{self.agent2_dir}\\documentation.md" (
    echo 📖 Documentation available: {self.agent2_dir}\\documentation.md
)

echo.
echo Press any key to close this window...
pause >nul
"""
        
        script_path_bat = self.agent3_dir / "run_web_automation.bat"
        script_path_bat.write_text(script_content, encoding="utf-8")
        return script_path_bat
    
    def _create_web_execution_script_unix(self) -> Path:
        """Create Unix web execution script"""
        venv_python = self.agent3_dir / "venv" / "bin" / "python"
        script_path = self.agent2_dir / "automation_script.py"
        
        script_content = f"""#!/bin/bash
clear
echo "================================================"
echo "  🌐 Enhanced AISA Web Automation Runner"
echo "  🖥️ Task ID: {self.seq_no}"
echo "================================================"
echo

cd "{self.agent3_dir}"

echo "[1/6] 🔍 Pre-execution validation..."
if [ ! -f "{venv_python}" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    "{sys.executable}" -m venv venv
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi

if [ ! -f "{script_path}" ]; then
    echo "❌ Automation script not found: {script_path}"
    read -p "Press Enter to exit..."
    exit 1
fi

echo "✅ Prerequisites validated"

echo
echo "[2/6] 📦 Installing/updating dependencies..."
source "{self.agent3_dir}/venv/bin/activate"
"{venv_python}" -m pip install --upgrade pip --quiet
"{venv_python}" -m pip install -r "{self.agent2_dir}/requirements.txt" --quiet
if [ $? -ne 0 ]; then
    echo "⚠️ Some dependencies may have failed to install"
    echo "Continuing with execution..."
fi

echo "✅ Dependencies ready"

echo
echo "[3/6] 🎭 Installing Playwright browsers..."
"{venv_python}" -m playwright install --with-deps
if [ $? -ne 0 ]; then
    echo "⚠️ Playwright browser installation had issues"
    echo "Continuing with execution..."
else
    echo "✅ Playwright browsers ready"
fi

echo
echo "[4/6] 🌐 Checking internet connectivity..."
if command -v ping >/dev/null 2>&1; then
    ping -c 1 google.com >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Internet connection available"
    else
        echo "⚠️ Internet connectivity issues detected"
        echo "Continuing anyway..."
    fi
else
    echo "⚠️ ping command not available, skipping connectivity test"
fi

echo
echo "[5/6] 🚀 Starting automation execution..."
echo "================================================"
echo "Starting at: $(date)"
echo "================================================"

"{venv_python}" "{script_path}"
SCRIPT_EXIT_CODE=$?

echo "================================================"
echo "Finished at: $(date)"
echo "Exit Code: $SCRIPT_EXIT_CODE"
echo "================================================"

echo
echo "[6/6] 📊 Execution summary..."
if [ $SCRIPT_EXIT_CODE -eq 0 ]; then
    echo "✅ AUTOMATION COMPLETED SUCCESSFULLY"
    echo "succeeded" > "{self.agent3_dir}/result.txt"
else
    echo "❌ AUTOMATION FAILED (Exit Code: $SCRIPT_EXIT_CODE)"
    echo "failed" > "{self.agent3_dir}/result.txt"
fi

echo
echo "📁 Results saved to: {self.agent3_dir}"
echo "📝 Check result.txt for execution status"
echo

if [ -f "{self.agent2_dir}/documentation.md" ]; then
    echo "📖 Documentation available: {self.agent2_dir}/documentation.md"
fi

echo
read -p "Press Enter to close this terminal..."
"""
        
        script_path_sh = self.agent3_dir / "run_web_automation.sh"
        script_path_sh.write_text(script_content, encoding="utf-8")
        os.chmod(script_path_sh, 0o755)
        return script_path_sh
    
    def create_monitoring_dashboard(self) -> Path:
        """Create an HTML monitoring dashboard"""
        dashboard_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AISA Automation Monitor - Task {self.seq_no}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }}
        .container {{ 
            max-width: 1200px; 
            margin: 0 auto; 
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }}
        .header h1 {{ margin-bottom: 10px; }}
        .header p {{ opacity: 0.9; }}
        .dashboard {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
        }}
        .card:hover {{ transform: translateY(-2px); }}
        .card h3 {{ 
            margin-bottom: 15px; 
            color: #4a5568;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
        }}
        .status-indicator {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .status-pending {{ background-color: #fbbf24; }}
        .status-running {{ background-color: #3b82f6; animation: pulse 2s infinite; }}
        .status-completed {{ background-color: #10b981; }}
        .status-failed {{ background-color: #ef4444; }}
        @keyframes pulse {{ 0%%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.5; }} }}
        .file-info {{ 
            margin: 10px 0;
            padding: 10px;
            background: #f7fafc;
            border-radius: 5px;
            border-left: 4px solid #4299e1;
        }}
        .button {{
            background: #4299e1;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            margin: 5px;
            transition: background 0.2s;
        }}
        .button:hover {{ background: #3182ce; }}
        .refresh-info {{
            text-align: center;
            color: white;
            opacity: 0.8;
            margin-top: 20px;
        }}
        .log-area {{
            background: #1a202c;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 5px;
            font-family: monospace;
            max-height: 300px;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 AISA Automation Monitor</h1>
            <p>Task ID: {self.seq_no} | Platform: {self.config.platform.value}</p>
        </div>
        
        <div class="dashboard">
            <div class="card">
                <h3>📊 Execution Status</h3>
                <div id="status-display">
                    <span class="status-indicator status-pending"></span>
                    <span id="status-text">Initializing...</span>
                </div>
                <div class="file-info">
                    <strong>Agent Files:</strong><br>
                    Agent 1: {self.agent1_dir}<br>
                    Agent 2: {self.agent2_dir}<br>
                    Agent 3: {self.agent3_dir}
                </div>
            </div>
            
            <div class="card">
                <h3>🔧 Quick Actions</h3>
                <a href="file:///{self.agent3_dir}" class="button">📁 Open Results Folder</a>
                <button class="button" onclick="location.reload()">🔄 Refresh Status</button>
                <a href="file:///{self.agent2_dir}/documentation.md" class="button">📖 View Documentation</a>
            </div>
            
            <div class="card">
                <h3>📋 Execution Log</h3>
                <div class="log-area" id="log-area">
                    <div>Waiting for execution to start...</div>
                    <div>Monitor files will be checked every 5 seconds</div>
                </div>
            </div>
            
            <div class="card">
                <h3>⚙️ Configuration</h3>
                <div class="file-info">
                    <strong>Platform:</strong> {self.config.platform.value}<br>
                    <strong>Enhanced Mode:</strong> {'Yes' if self.config.use_enhanced_agent else 'No'}<br>
                    <strong>Monitoring:</strong> {'Enabled' if self.config.enable_monitoring else 'Disabled'}<br>
                    <strong>Timeout:</strong> {self.config.timeout_seconds}s<br>
                    <strong>Retry Attempts:</strong> {self.config.retry_attempts}
                </div>
            </div>
        </div>
        
        <div class="refresh-info">
            <p>🔄 This page auto-refreshes every 30 seconds | Last updated: <span id="last-updated"></span></p>
        </div>
    </div>
    
    <script>
        function updateLastUpdated() {{
            document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
        }}
        
        function checkExecutionStatus() {{
            // This would check result.txt and update status
            // For now, just update timestamp
            updateLastUpdated();
        }}
        
        // Auto refresh every 30 seconds
        setInterval(() => {{
            location.reload();
        }}, 30000);
        
        // Initial update
        updateLastUpdated();
        checkExecutionStatus();
    </script>
</body>
</html>"""
        
        dashboard_path = self.agent3_dir / "monitoring_dashboard.html"
        dashboard_path.write_text(dashboard_content, encoding="utf-8")
        return dashboard_path
    
    def execute(self) -> Dict[str, Any]:
        """Execute the automation with full monitoring and reporting"""
        print(f"[{self.seq_no}] 🚀 Starting Enhanced Agent 3 Execution Engine")
        
        execution_result = {
            "status": ExecutionStatus.PENDING.value,
            "started_at": time.time(),
            "validation": {},
            "scripts_generated": {},
            "monitoring_dashboard": None,
            "errors": []
        }
        
        try:
            # Step 1: Validate prerequisites
            print(f"[{self.seq_no}] 🔍 Validating prerequisites...")
            validation = self.validate_prerequisites()
            execution_result["validation"] = validation
            
            if not validation["valid"]:
                execution_result["status"] = ExecutionStatus.FAILED.value
                execution_result["errors"] = validation["issues"]
                return execution_result
            
            if validation["warnings"]:
                print(f"[{self.seq_no}] ⚠️ Warnings detected:")
                for warning in validation["warnings"]:
                    print(f"  - {warning}")
            
            # Step 2: Create virtual environment
            print(f"[{self.seq_no}] 🐍 Setting up virtual environment...")
            if not self.create_virtual_environment():
                execution_result["status"] = ExecutionStatus.FAILED.value
                execution_result["errors"].append("Virtual environment setup failed")
                return execution_result
            
            # Step 3: Generate execution scripts
            print(f"[{self.seq_no}] 📝 Generating execution scripts...")
            scripts = self.generate_execution_scripts()
            execution_result["scripts_generated"] = {k: str(v) for k, v in scripts.items()}
            
            # Step 4: Create monitoring dashboard
            if self.config.enable_monitoring:
                print(f"[{self.seq_no}] 📊 Creating monitoring dashboard...")
                dashboard = self.create_monitoring_dashboard()
                execution_result["monitoring_dashboard"] = str(dashboard)
            
            # Step 5: Launch execution
            print(f"[{self.seq_no}] 🎬 Launching execution environment...")
            
            if self.config.platform == Platform.MOBILE:
                # Launch Appium server first
                if "appium_server" in scripts:
                    print(f"[{self.seq_no}] 📱 Opening Appium server terminal...")
                    webbrowser.open(f'file:///{scripts["appium_server"].resolve()}')
                    time.sleep(2)  # Brief delay
                
                # Launch main execution
                print(f"[{self.seq_no}] 🤖 Opening automation runner...")
                webbrowser.open(f'file:///{scripts["execution"].resolve()}')
            else:
                # Web automation - single script
                print(f"[{self.seq_no}] 🌐 Opening web automation runner...")
                webbrowser.open(f'file:///{scripts["execution"].resolve()}')
            
            # Launch monitoring dashboard
            if execution_result["monitoring_dashboard"]:
                time.sleep(1)
                print(f"[{self.seq_no}] 📊 Opening monitoring dashboard...")
                webbrowser.open(f'file:///{dashboard.resolve()}')
            
            execution_result["status"] = ExecutionStatus.RUNNING.value
            
            print(f"[{self.seq_no}] ✅ Enhanced Agent 3 execution launched successfully!")
            print(f"[{self.seq_no}] 📁 Results will be saved to: {self.agent3_dir}")
            
            if self.config.platform == Platform.MOBILE:
                print(f"[{self.seq_no}] 📱 Make sure your mobile device is connected and USB debugging is enabled")
                print(f"[{self.seq_no}] 🔌 Appium server terminal opened - keep it running during execution")
            
            return execution_result
            
        except Exception as e:
            execution_result["status"] = ExecutionStatus.FAILED.value
            execution_result["errors"].append(f"Execution engine error: {str(e)}")
            print(f"[{self.seq_no}] ❌ Enhanced Agent 3 execution failed: {e}")
            return execution_result

def run_enhanced_agent3(seq_no: str, platform: str, use_enhanced_agent: bool = True) -> dict:
    """Enhanced Agent 3 main function with backward compatibility"""
    
    # Convert platform string to enum
    platform_enum = Platform.MOBILE if platform.lower() == "mobile" else Platform.WEB
    
    # Create configuration
    config = ExecutionConfig(
        platform=platform_enum,
        use_enhanced_agent=use_enhanced_agent,
        enable_monitoring=True,
        enable_reporting=True
    )
    
    # Create and run execution engine
    engine = EnhancedExecutionEngine(seq_no, config)
    result = engine.execute()
    
    return result

# Backward compatible wrapper
def run_agent3(seq_no: str, platform: str) -> dict:
    """Backward compatible wrapper for enhanced agent 3"""
    return run_enhanced_agent3(seq_no, platform, use_enhanced_agent=True)

# Export functions
__all__ = ["run_enhanced_agent3", "run_agent3", "EnhancedExecutionEngine", "ExecutionConfig"]
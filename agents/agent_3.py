# agents/agent_3.py

from pathlib import Path
import sys
import os
import subprocess
import webbrowser
from fastapi import HTTPException

from config import ARTIFACTS_DIR

def run_agent3(seq_no: str, platform: str) -> dict:
    """
    Agent 3: Creates an isolated venv and executes the generated script.
    - Uses the OS's default terminal for cross-platform compatibility.
    - For mobile, it opens separate terminals for the Appium server and the script.
    - For web, it ensures Playwright is installed and runs the script in a single terminal.
    """
    print(f"[{seq_no}] Running Agent 3 for '{platform}' platform.")
    agent2_dir = ARTIFACTS_DIR / seq_no / "agent2"
    agent3_dir = ARTIFACTS_DIR / seq_no / "agent3"
    agent3_dir.mkdir(parents=True, exist_ok=True)

    script_path = agent2_dir / "automation_script.py"
    reqs_path = agent2_dir / "requirements.txt"
    if not script_path.exists() or not reqs_path.exists():
        raise HTTPException(status_code=404, detail=f"Script or requirements not found for task {seq_no}.")

    venv_dir = agent3_dir / "env"
    python_executable = sys.executable

    # --- Platform-specific execution logic ---
    if platform == "mobile":
        # --- Mobile Flow: Two terminals (Appium Server + Script Executor) ---
        if sys.platform == "win32":
            venv_python = venv_dir / "Scripts" / "python.exe"
            activate_script = venv_dir / "Scripts" / "activate.bat"
            
            # Script 1: Start Appium Server
            appium_script_commands = f"""
            @echo off
            title Appium Server ({seq_no})
            echo Starting Appium Server for Task ID: {seq_no}
            echo This window must remain open during the test.
            appium
            """
            appium_script_path = agent3_dir / "start_appium.bat"
            appium_script_path.write_text(appium_script_commands, encoding="utf-8")
            webbrowser.open(str(appium_script_path))

            # Script 2: Run Automation
            run_script_commands = f"""
            @echo off
            title Automation Runner ({seq_no})
            echo --------------------------------------------------
            echo AISA Mobile Automation Task: {seq_no}
            echo --------------------------------------------------
            cd /d "{agent3_dir}"
            echo [1/4] Creating virtual environment...
            "{python_executable}" -m venv env || (echo Venv creation failed. & pause & exit /b 1)
            
            echo [2/4] Installing dependencies...
            call "{activate_script}"
            "{venv_python}" -m pip install --upgrade pip > nul
            "{venv_python}" -m pip install -r "{reqs_path}" || (echo Dependency installation failed. & pause & exit /b 1)
            
            echo.
            echo Waiting 10s for Appium server to initialize...
            timeout /t 10 /nobreak > nul
            
            echo [3/4] Running automation script...
            echo --------------------------------------------------
            "{venv_python}" "{script_path}"
            if %errorlevel% equ 0 (echo succeeded > result.txt) else (echo failed > result.txt)
            echo --------------------------------------------------
            echo [4/4] Script finished. This window can be closed.
            pause
            """
            run_script_path = agent3_dir / "run_automation.bat"
            run_script_path.write_text(run_script_commands, encoding="utf-8")
            webbrowser.open(str(run_script_path))

        else:  # macOS / Linux
            venv_python = venv_dir / "bin" / "python"
            activate_script = venv_dir / "bin" / "activate"

            # Script 1: Start Appium Server
            appium_script_commands = f"""#!/bin/bash
            echo "Starting Appium Server for Task ID: {seq_no}"
            echo "This terminal must remain open during the test."
            appium
            """
            appium_script_path = agent3_dir / "start_appium.sh"
            appium_script_path.write_text(appium_script_commands, encoding="utf-8")
            os.chmod(appium_script_path, 0o755)
            webbrowser.open(f'file://{appium_script_path.resolve()}')

            # Script 2: Run Automation
            run_script_commands = f"""#!/bin/bash
            cd "{agent3_dir}"
            echo "--------------------------------------------------"
            echo "  AISA Mobile Automation Task: {seq_no}"
            echo "--------------------------------------------------"
            echo "[1/4] Creating virtual environment..."
            "{python_executable}" -m venv env || {{ echo "Venv creation failed."; read -p "Press Enter to exit."; exit 1; }}
            
            echo "[2/4] Installing dependencies..."
            source "{activate_script}"
            "{venv_python}" -m pip install --upgrade pip > /dev/null
            "{venv_python}" -m pip install -r "{reqs_path}" || {{ echo "Dependency installation failed."; read -p "Press Enter to exit."; exit 1; }}

            echo "Waiting 10s for Appium server to initialize..."
            sleep 10

            echo "[3/4] Running automation script..."
            echo "--------------------------------------------------"
            "{venv_python}" "{script_path}"
            if [ $? -eq 0 ]; then echo "succeeded" > result.txt; else echo "failed" > result.txt; fi
            echo "--------------------------------------------------"
            echo "[4/4] Script finished. Press Enter to close."
            read
            """
            run_script_path = agent3_dir / "run_automation.sh"
            run_script_path.write_text(run_script_commands, encoding="utf-8")
            os.chmod(run_script_path, 0o755)
            webbrowser.open(f'file://{run_script_path.resolve()}')

    else:  # platform == "web"
        if sys.platform == "win32":
            venv_python = venv_dir / "Scripts" / "python.exe"
            activate_script = venv_dir / "Scripts" / "activate.bat"
            commands = f"""
            @echo off
            title Web Automation ({seq_no})
            cd /d "{agent3_dir}"
            echo [1/5] Creating virtual environment...
            "{python_executable}" -m venv env || (echo Venv creation failed. & pause & exit /b 1)

            echo [2/5] Installing dependencies...
            call "{activate_script}"
            "{venv_python}" -m pip install --upgrade pip > nul
            "{venv_python}" -m pip install -r "{reqs_path}" || (echo Dependency installation failed. & pause & exit /b 1)
            
            echo [3/5] Installing Playwright browsers...
            playwright install || (echo Playwright install failed. & pause & exit /b 1)
            
            echo [4/5] Running automation script...
            "{venv_python}" "{script_path}"
            if %errorlevel% equ 0 (echo succeeded > result.txt) else (echo failed > result.txt)

            echo [5/5] Script finished. This window can be closed.
            pause
            """
            run_script_path = agent3_dir / "run.bat"
            run_script_path.write_text(commands, encoding="utf-8")
            webbrowser.open(str(run_script_path))
        else:  # macOS / Linux
            venv_python = venv_dir / "bin" / "python"
            activate_script = venv_dir / "bin" / "activate"
            commands = f"""#!/bin/bash
            cd "{agent3_dir}"
            echo "[1/5] Creating virtual environment..."
            "{python_executable}" -m venv env || {{ echo "Venv creation failed."; read -p "Press Enter to exit."; exit 1; }}

            echo "[2/5] Installing dependencies..."
            source "{activate_script}"
            "{venv_python}" -m pip install --upgrade pip > /dev/null
            "{venv_python}" -m pip install -r "{reqs_path}" || {{ echo "Dependency installation failed."; read -p "Press Enter to exit."; exit 1; }}

            echo "[3/5] Installing Playwright browsers..."
            playwright install || {{ echo "Playwright install failed."; read -p "Press Enter to exit."; exit 1; }}

            echo "[4/5] Running automation script..."
            "{venv_python}" "{script_path}"
            if [ $? -eq 0 ]; then echo "succeeded" > result.txt; else echo "failed" > result.txt; fi

            echo "[5/5] Script finished. Press Enter to close."
            read
            """
            run_script_path = agent3_dir / "run.sh"
            run_script_path.write_text(commands, encoding="utf-8")
            os.chmod(run_script_path, 0o755)
            webbrowser.open(f'file://{run_script_path.resolve()}')

    print(f"[{seq_no}] Agent 3 launched execution flow.")
    return {"status": "running", "message": "Execution terminals have been opened."}
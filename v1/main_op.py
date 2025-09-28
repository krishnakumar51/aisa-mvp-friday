import uuid
import shutil
import subprocess
import json
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse

# Replace these with your actual imports
from config import ARTIFACTS_DIR
from agents.enhanced_agent_manager import EnhancedAgentManager

app = FastAPI(title="AISA v3 - Intelligent & Robust")
agent_manager = EnhancedAgentManager()
# --- Phase 3 Change: Persistent State Management ---
def _get_task_state(seq_no: str) -> dict:
    """Reads the state of a task from its status.json file."""
    status_file = ARTIFACTS_DIR / seq_no / "status.json"
    if not status_file.exists():
        return None
    return json.loads(status_file.read_text())

def _update_task_state(seq_no: str, data: dict):
    """Updates and saves the state of a task to its status.json file."""
    state = _get_task_state(seq_no) or {}
    state.update(data)
    status_file = ARTIFACTS_DIR / seq_no / "status.json"
    status_file.write_text(json.dumps(state, indent=2))

# --- Startup Event ---
@app.on_event("startup")
def on_startup():
    print("--- AISA Server Starting Up ---")
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    print(f"Artifacts will be stored in: {ARTIFACTS_DIR}")
    
    print("Checking for ADB devices...")
    try:
        result = subprocess.run(["adb", "devices"], capture_output=True, text=True, check=True, shell=True)
        print("ADB check successful:\n", result.stdout)
    except FileNotFoundError:
        print("ADB command not found. Please ensure it's in your system's PATH.")
    except Exception as e:
        print(f"ADB check failed: {e}")
    print("--- Startup complete. Waiting for tasks. ---")

# --- Embedded Frontend ---
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return HTMLResponse("""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AISA Task Creator</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f4f4f4;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
    }
    .container {
      background: #fff;
      padding: 30px;
      border-radius: 10px;
      box-shadow: 0 0 15px rgba(0,0,0,0.2);
      width: 400px;
    }
    h1 { text-align: center; }
    label { display: block; margin-top: 15px; font-weight: bold; }
    input[type="text"], select, input[type="file"] {
      width: 100%;
      padding: 8px;
      margin-top: 5px;
      border-radius: 5px;
      border: 1px solid #ccc;
    }
    button {
      margin-top: 20px;
      width: 100%;
      padding: 10px;
      background: #007bff;
      color: white;
      border: none;
      border-radius: 5px;
      cursor: pointer;
      font-size: 16px;
    }
    button:hover { background: #0056b3; }
    .response { margin-top: 20px; white-space: pre-wrap; background: #eee; padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Create Task</h1>
    <form id="taskForm">
      <label for="instructions">Instructions:</label>
      <input type="text" id="instructions" name="instructions" required>

      <label for="platform">Platform:</label>
      <select id="platform" name="platform" required>
        <option value="mobile">Mobile</option>
        <option value="web">Web</option>
      </select>

      <label for="pdf">Upload PDF:</label>
      <input type="file" id="pdf" name="pdf" accept=".pdf" required>

      <button type="submit">Submit Task</button>
    </form>
    <div class="response" id="response"></div>
  </div>

  <script>
    const form = document.getElementById('taskForm');
    const responseDiv = document.getElementById('response');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const formData = new FormData(form);

      try {
        const res = await fetch('/create_task', {
          method: 'POST',
          body: formData
        });

        const data = await res.json();
        responseDiv.textContent = JSON.stringify(data, null, 2);
      } catch (err) {
        responseDiv.textContent = 'Error: ' + err;
      }
    });
  </script>
</body>
</html>
""")

# --- Create Task Endpoint ---
@app.post("/create_task", status_code=201)
async def create_task(
    instructions: str = Form(...),
    platform: str = Form(..., enum=["mobile", "web"]),
    pdf: UploadFile = File(...)
):
    seq_no = uuid.uuid4().hex[:10]
    task_dir = ARTIFACTS_DIR / seq_no
    task_dir.mkdir(exist_ok=True)
    
    _update_task_state(seq_no, {"status": "processing", "platform": platform})
    
    pdf_path = task_dir / "input.pdf"
    with pdf_path.open("wb") as buffer:
        shutil.copyfileobj(pdf.file, buffer)

    # Enhanced Agent Pipeline
    try:
        result = agent_manager.run_enhanced_pipeline(seq_no, pdf_path, instructions, platform)
        _update_task_state(seq_no, {
            "status": "completed",
            "result": result,
            "execution_mode": "enhanced"
        })
    except Exception as e:
        _update_task_state(seq_no, {
            "status": "failed", 
            "error": str(e),
            "execution_mode": "enhanced"
        })
    
    print(f"Task {seq_no} created successfully with enhanced pipeline.")
    return _get_task_state(seq_no)


# --- Get Task Status Endpoint ---
@app.get("/task/{seq_no}")
async def get_task_status(seq_no: str):
    # Check enhanced pipeline status
    # enhanced_status = get_pipeline_status(seq_no)
    # if enhanced_status:
    #     return enhanced_status
    
    # Fallback to standard status
    task_info = _get_task_state(seq_no)
    if not task_info:
        raise HTTPException(status_code=404, detail="Task not found.")
        
    return task_info
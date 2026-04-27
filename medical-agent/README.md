# MedAgent — Medical AI Assistant (Graduation Project Skeleton)

FastAPI + LangChain ReAct agent + Groq (qwen3-32b) + 3 mock medical tools.

---

## Project Structure

```
medical-agent/
│
├── main.py                  ← FastAPI app entry point (serves frontend + mounts API)
│
├── api/
│   ├── __init__.py
│   └── routes.py            ← POST /api/chat endpoint
│
├── agent/
│   ├── __init__.py
│   ├── agent.py             ← LangChain ReAct agent (run_agent function)
│   └── tools.py             ← 3 mock tools: symptoms_analysis, drug_interaction_checker, lab_report_explanation
│
├── static/
│   └── index.html           ← Full chat UI (served at localhost:8000)
│
├── .env.example             ← Copy to .env and fill in your GROQ_API_KEY
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### 1. Clone / place the project folder
```bash
cd medical-agent
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS / Linux:
source venv/bin/activate
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your Groq API key
```bash
# Copy the example env file
cp .env.example .env

# Open .env and replace the placeholder with your real key
# Get your key at: https://console.groq.com/keys
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the server
```bash
uvicorn main:app --reload --port 8000
```

### 6. Open the chat UI
Visit: http://localhost:8000

---

## API Reference

### POST /api/chat

**Request body:**
```json
{
  "query": "I have chest pain and shortness of breath"
}
```

**Response body:**
```json
{
  "response": "[MOCK — Symptoms Analysis] Hey there! ...",
  "tool_used": "symptoms_analysis"
}
```

`tool_used` will be one of:
- `"symptoms_analysis"`
- `"drug_interaction_checker"`
- `"lab_report_explanation"`
- `null` (agent answered directly without calling a tool)

---

## How the Agent Decides Which Tool to Call

The agent is a LangChain ReAct agent (via LangGraph `create_react_agent`).
It reads the tool docstrings to understand when to call each tool:

| Tool | Triggers when user asks about… |
|---|---|
| `symptoms_analysis` | Physical symptoms, pain, discomfort, "what could this be?" |
| `drug_interaction_checker` | Combining medications, drug safety, side effects of multiple drugs |
| `lab_report_explanation` | Blood test values, urine results, any diagnostic numbers |
| *(no tool)* | General medical questions, definitions, explanations |

---

## Where to Add Real Logic

Open `agent/tools.py`. Each tool has a clearly marked `# ── TODO` comment:

```python
@tool
def symptoms_analysis(symptoms: str) -> str:
    # ── TODO: Replace this mock with real symptoms analysis logic ──
    return "[MOCK] ..."
```

Replace the `return` statement with your actual implementation.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API server | FastAPI 0.115 |
| ASGI server | Uvicorn |
| Agent framework | LangChain 0.3 + LangGraph 0.3 |
| LLM | Groq — qwen/qwen3-32b |
| Frontend | Vanilla HTML/CSS/JS (served by FastAPI) |
| Config | python-dotenv |
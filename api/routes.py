"""
api/routes.py
-------------
FastAPI router — five endpoints:

  POST /api/chat            — text chat (session-aware with history)
  POST /api/lab-upload      — PDF/image file upload (unchanged)
  POST /api/create_session  — create a new chat session
  GET  /api/list_sessions   — list all sessions for a user
  GET  /api/get_session     — get full message history for a session
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel

from agent.agent import run_agent
from api.db import (
    create_session, get_session, list_sessions, update_session_messages,
    delete_session, rename_session
)

from langchain_core.messages import HumanMessage, AIMessage
from fastapi import BackgroundTasks
from evaluation.online_evaluator import run_online_evaluation

router = APIRouter()

# ── Allowed upload MIME types ─────────────────────────────────────────────────
_ALLOWED_PDF_TYPES = {"application/pdf"}
_ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/tiff", "image/bmp"}
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024   # 10 MB


# ══════════════════════════════════════════════════════════════════════════════
# Schemas
# ══════════════════════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    query: str
    thread_id: str | None = None
    user_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    tool_used: str | None


class CreateSessionRequest(BaseModel):
    user_id: str


class CreateSessionResponse(BaseModel):
    thread_id: str


class RenameSessionRequest(BaseModel):
    thread_id: str
    user_id: str
    new_name: str


class SessionSummary(BaseModel):
    thread_id: str
    session_name: str | None
    created_at: str


class SessionDetail(BaseModel):
    thread_id: str
    session_name: str | None
    created_at: str
    messages: list[dict]


# ══════════════════════════════════════════════════════════════════════════════
# Endpoint 1 — Text chat (session-aware)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    """
    Receives a user text query and routes it through the LangChain agent.
    When thread_id and user_id are provided, loads conversation history
    from the database and passes the full context to the agent.

    Falls back to stateless mode when session fields are absent.
    """
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    query = request.query.strip()

    # ── Session-aware mode ────────────────────────────────────────────────────
    if request.thread_id and request.user_id:
        # Step 1: Fetch session from DB
        session = get_session(request.thread_id, request.user_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Session not found or access denied.",
            )

        # Step 2: Convert stored dicts to LangChain messages
        stored_messages = session["messages"]  # list of {"role": ..., "content": ...}
        history = []
        for msg in stored_messages:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                history.append(AIMessage(content=msg["content"]))

        # Step 3 + 4: Call agent with history
        result = run_agent(query, history=history)

        # Step 5: Extract AI response
        ai_response = result["response"]

        # Step 6: Append both messages as raw dicts
        stored_messages.append({"role": "user", "content": query})
        stored_messages.append({"role": "assistant", "content": ai_response})

        # Step 7: Auto-generate session name from first user message
        new_session_name = None
        if not session["session_name"]:
            # Truncate to 50 chars for the sidebar display
            new_session_name = query[:50] + ("…" if len(query) > 50 else "")

        # Step 8: Update session in SQLite
        update_session_messages(
            thread_id=request.thread_id,
            user_id=request.user_id,
            messages=stored_messages,
            session_name=new_session_name,
        )

        run_id = result.get("run_id")
        if run_id:
            background_tasks.add_task(
                run_online_evaluation,
                run_id=run_id,
                user_input=query,
                agent_output=ai_response
            )

        # Step 9: Return response
        return ChatResponse(
            response=ai_response,
            tool_used=result["tool_used"],
        )

    # ── Legacy stateless mode ─────────────────────────────────────────────────
    result = run_agent(query)
    ai_response = result["response"]
    run_id = result.get("run_id")
    
    if run_id:
        background_tasks.add_task(
            run_online_evaluation,
            run_id=run_id,
            user_input=query,
            agent_output=ai_response
        )

    return ChatResponse(
        response=ai_response,
        tool_used=result["tool_used"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Endpoint 2 — Lab file upload (unchanged)
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/lab-upload", response_model=ChatResponse)
async def lab_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...), lang: str = "en") -> ChatResponse:
    """
    Accepts a PDF or image file containing a lab report.
    Extracts the text from the file, then passes it to the lab tool.

    Supported formats: PDF, JPEG, PNG, TIFF, BMP.
    Max file size: 10 MB.
    """
    # ── Validate content type ─────────────────────────────────────────────────
    content_type = file.content_type or ""
    is_pdf = content_type in _ALLOWED_PDF_TYPES
    is_image = content_type in _ALLOWED_IMAGE_TYPES

    if not is_pdf and not is_image:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{content_type}'. "
                "Please upload a PDF or image file (JPEG, PNG, TIFF, BMP)."
            ),
        )

    # ── Read file bytes ────────────────────────────────────────────────────────
    file_bytes = await file.read()
    if len(file_bytes) > _MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum allowed size is 10 MB.",
        )

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── Extract text from file ────────────────────────────────────────────────
    raw_text = ""

    if is_pdf:
        from agent.tools.lab.parsers.pdf_parser import parse_pdf
        raw_text = parse_pdf(file_bytes)
    elif is_image:
        from agent.tools.lab.parsers.ocr_parser import parse_image
        raw_text = parse_image(file_bytes)

    if not raw_text or not raw_text.strip():
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not extract text from the uploaded file. "
                "Please ensure the file contains readable lab values, "
                "or type your results directly in the chat."
            ),
        )

    # ── Pass extracted text to the lab tool via the agent ────────────────────
    from agent.tools.lab.lab_tool import lab_report_explanation
    from langchain_core.tracers.context import collect_runs
    
    with collect_runs() as cb:
        response_text = lab_report_explanation.invoke({"report": raw_text.strip(), "language": lang})
        run_id = str(cb.traced_runs[0].id) if cb.traced_runs else None

    if run_id:
        background_tasks.add_task(
            run_online_evaluation,
            run_id=run_id,
            user_input=f"[Lab Upload] {raw_text.strip()[:200]}...",
            agent_output=response_text
        )

    return ChatResponse(
        response=response_text,
        tool_used="lab_report_explanation",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Endpoint 3 — Create session
# ══════════════════════════════════════════════════════════════════════════════

@router.post("/create_session", response_model=CreateSessionResponse)
def create_new_session(request: CreateSessionRequest) -> CreateSessionResponse:
    """
    Creates a new chat session for the given user.
    Returns the generated thread_id (UUID).
    """
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id must not be empty.")

    thread_id = create_session(request.user_id.strip())
    return CreateSessionResponse(thread_id=thread_id)


# ══════════════════════════════════════════════════════════════════════════════
# Endpoint 4 — List sessions
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/list_sessions", response_model=list[SessionSummary])
def list_user_sessions(user_id: str = Query(..., description="The user identifier")):
    """
    Returns all sessions for a user, ordered newest first.
    """
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id must not be empty.")

    sessions = list_sessions(user_id.strip())
    return [SessionSummary(**s) for s in sessions]


# ══════════════════════════════════════════════════════════════════════════════
# Endpoint 5 — Get session
# ══════════════════════════════════════════════════════════════════════════════

@router.get("/get_session", response_model=SessionDetail)
def get_user_session(
    thread_id: str = Query(..., description="The session thread ID"),
    user_id: str = Query(..., description="The user identifier"),
):
    """
    Returns the full message history for a specific session.
    Verifies user_id matches for security.
    """
    if not thread_id or not thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id must not be empty.")
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id must not be empty.")

    session = get_session(thread_id.strip(), user_id.strip())
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or access denied.",
        )

    return SessionDetail(**session)


# ══════════════════════════════════════════════════════════════════════════════
# Endpoint 6 — Delete session
# ══════════════════════════════════════════════════════════════════════════════

@router.delete("/delete_session")
def delete_user_session(
    thread_id: str = Query(..., description="The session thread ID"),
    user_id: str = Query(..., description="The user identifier"),
):
    """
    Deletes a specific session.
    """
    if not thread_id or not thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id must not be empty.")
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id must not be empty.")

    success = delete_session(thread_id.strip(), user_id.strip())
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or access denied.")
    return {"status": "success"}


# ══════════════════════════════════════════════════════════════════════════════
# Endpoint 7 — Rename session
# ══════════════════════════════════════════════════════════════════════════════

@router.put("/rename_session")
def rename_user_session(request: RenameSessionRequest):
    """
    Renames a specific session.
    """
    if not request.thread_id or not request.thread_id.strip():
        raise HTTPException(status_code=400, detail="thread_id must not be empty.")
    if not request.user_id or not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id must not be empty.")
    if not request.new_name or not request.new_name.strip():
        raise HTTPException(status_code=400, detail="new_name must not be empty.")

    success = rename_session(request.thread_id.strip(), request.user_id.strip(), request.new_name.strip())
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or access denied.")
    return {"status": "success"}
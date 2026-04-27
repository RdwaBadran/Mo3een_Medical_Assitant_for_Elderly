# Enhancing Session Management — Implementation Plan

I will address each of your concerns and questions, and implement the missing features (Delete/Rename) along with fixing the empty chat clutter.

## 1. Explanations for Your Questions

### How `user_id` is handled
Because the application doesn't currently have a login/authentication system, I securely added the column to the database for future proofing but hardcoded `user_id = 'default_user'` in the frontend JavaScript (`index.html`). This means you are bypassing authentication, but the database is already fully prepped for it if you add users later.

### Session Ordering
They **are** currently ordered by time. In `api/db.py`, the `list_sessions` function uses `ORDER BY created_at DESC`, which means the newly created conversations always appear at the top.

### The Uvicorn Error
The error trace you triggered happened because you ran `uvicorn main:app` via a different Python virtual environment from another project `C:\Users\Magic\Desktop\equipment-monitor\venv\...`. It couldn't find `langchain_groq` because it wasn't the `medical-agent` environment. 
You must explicitly use the correct environment:
`.\venv\Scripts\python.exe -m uvicorn main:app --port 8000`

---

## 2. Proposed Implementation Changes

### Fix: Empty Chat Clutter
Currently, clicking "New Chat" immediately saves a blank session to the database. If you leave without typing, it stays there forever.
**Solution:** I will change the frontend logic so that clicking "New Chat" only clears your screen (an "optimistic" UI change). It will *not* hit the database until you actually send your first message. This will completely eliminate empty unused chats from accumulating in the DB.

### Feature: Delete & Rename Conversations
I will add the endpoints and UI to manage your conversations.

#### 1. Database (`api/db.py`)
- Add `delete_session(thread_id: str, user_id: str) -> bool`
- Add `rename_session(thread_id: str, user_id: str, new_name: str) -> bool`

#### 2. Routes (`api/routes.py`)
- `DELETE /api/sessions/{thread_id}` — Deletes the session from SQLite.
- `PUT /api/sessions/{thread_id}` — Accepts a new JSON body `{"session_name": "..."}` and updates the session name.

#### 3. Frontend (`static/index.html`)
- Modify the sidebar session items to include subtle **Rename (✏️)** and **Delete (🗑️)** buttons that appear when you hover over a session.
- Add Javascript `renameSession(threadId)` that opens a prompt to enter a new name, then calls the PUT setup.
- Add Javascript `deleteSession(threadId)` that shows a confirm dialog, calls the DELETE endpoint, removes the chat from the UI, and creates a "New Chat" if you deleted the active one.

## User Review Required

> [!IMPORTANT]
> Because renaming and deleting require UI, I'll add subtle icons directly inside the sidebar next to each session name. I will keep the design perfectly perfectly matched to your dark theme. Look at the plan above and let me know if this sounds perfectly aligned with what you need, and I'll jump straight into the code.

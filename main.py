"""
main.py
-------
Entry point for the Medical AI Assistant FastAPI server.
Serves the frontend HTML and exposes the /chat API endpoint.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
import os
import logging
from dotenv import load_dotenv
from api.routes import router
from api.db import create_table_if_not_exists

load_dotenv()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup validation
    openai_key = os.getenv("OPENAI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    missing = []
    if not openai_key: missing.append("OPENAI_API_KEY")
    if not groq_key: missing.append("GROQ_API_KEY")
    
    if missing:
        msg = f"FATAL ERROR: Missing required environment variables: {', '.join(missing)}. Please set them in your .env file."
        logger.error(msg)
        raise RuntimeError(msg)

    # Initialize the sessions database table
    create_table_if_not_exists()

    yield
    # Shutdown logic (if any)


app = FastAPI(
    title="Medical AI Assistant",
    description="LangChain agent with medical mock tools powered by Groq",
    version="1.0.0",
    lifespan=lifespan
)

# Allow all origins during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static folder so CSS/JS assets can be served if needed in the future
app.mount("/static", StaticFiles(directory="static"), name="static")

# Register all API routes under /api prefix
app.include_router(router, prefix="/api")


@app.get("/", response_class=FileResponse)
def serve_frontend():
    """
    Serve the main chat UI HTML file at the root URL (localhost:8000).

    Returns:
        FileResponse: An HTTP response returning the static `index.html` file.
    """
    return FileResponse("static/index.html")
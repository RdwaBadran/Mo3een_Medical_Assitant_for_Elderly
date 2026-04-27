"""
agent/agent.py
--------------
Builds and runs the LangChain ReAct agent.

Routing logic:
  - Symptom queries      → symptoms_analysis tool (RAG + LLM pipeline)
  - Drug queries         → drug_interaction_checker tool
  - Lab result queries   → lab_report_explanation tool
  - General / greetings  → LLM answers directly (no tool called)

When a tool is called, its raw output is returned directly to the
frontend — the LLM's rephrasing step is deliberately bypassed.
"""

import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from agent.tools import ALL_TOOLS

load_dotenv()

# LangChain SDK requires these specific environment variable names for tracing
if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
if os.getenv("LANGSMITH_TRACING") and not os.getenv("LANGCHAIN_TRACING_V2"):
    os.environ["LANGCHAIN_TRACING_V2"] = os.environ["LANGSMITH_TRACING"]
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "mo3een-evaluation")


SYSTEM_PROMPT = """\
You are MedAgent, a compassionate AI medical assistant specialised in elderly patient care.

You have three specialist tools — use them when the query clearly matches:
  • symptoms_analysis          — user describes physical symptoms, pain, or discomfort
  • drug_interaction_checker   — user asks about medications, drug combinations, or side effects
  • lab_report_explanation     — user shares lab test values or asks what blood/urine results mean

Decision rules (follow strictly):
  1. Determine the user's language (e.g., Arabic or English).
  2. If a tool is needed, ALWAYS pass the 'language' argument (use 'ar' for Arabic, 'en' for English).
  3. If the query fits one of the tools above → call that tool.
  4. If the query is a general medical question (e.g. "What is hypertension?") → answer directly
     in the user's language; do NOT call a tool.
  5. If the query is non-medical (cooking, geography, politics, sports, etc) →
     politely refuse to answer in the user's language.
  6. Always be clear, warm, and patient — your users are elderly people.
"""


def _build_llm():
    """
    Instantiates the Groq LLM using the robust factory.
    Uses the qwen/qwen3-32b model.
    """
    from agent.llm_factory import get_groq_llm
    return get_groq_llm(model_name="qwen/qwen3-32b", temperature=0)


_llm: ChatGroq | None = None
_agent = None  # We'll use a type ignore or lazy load

def get_agent():
    """
    Returns (and initializes if needed) the LangChain ReAct agent.
    
    This ensures that LLM initialization only happens when the agent is 
    actually called, allowing the module to be imported without 
    EnvironmentErrors (useful for testing).
    """
    global _llm, _agent
    if _llm is None:
        _llm = _build_llm()
    if _agent is None:
        _agent = create_react_agent(
            model=_llm,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
        )
    return _agent


def run_agent(query: str, history: list | None = None) -> dict:
    """
    Invokes the agent, then inspects the message list directly to return results.

    If a tool was called, this intercepts the message directly avoiding
    the LLM from restating or rephrasing the result.

    Args:
        query (str): The initial user query text.
        history (list | None): Optional list of LangChain message objects
            (HumanMessage / AIMessage) representing prior conversation turns.
            When provided, these are prepended before the new query.

    Returns:
        dict: A dictionary containing:
            - response (str): The final text answer from the agent or tool.
            - tool_used (str | None): Name of the tool called, or None.
            - run_id (str | None): LangSmith run ID.
    """
    from langchain_core.tracers.context import collect_runs

    agent = get_agent()

    messages = list(history) if history else []
    messages.append(HumanMessage(content=query))

    import uuid
    run_id = str(uuid.uuid4())
    
    result = agent.invoke(
        {"messages": messages}, 
        config={"run_id": run_id}
    )

    all_messages = result["messages"]

    # ── Step 1: find the tool call (if any) ──────────────────
    tool_used = None
    tool_output = None

    for message in all_messages:
        # AIMessage with tool_calls = the agent decided to call a tool
        if hasattr(message, "tool_calls") and message.tool_calls:
            tool_used = message.tool_calls[0]["name"]

        # ToolMessage = the actual return value of the tool function
        if isinstance(message, ToolMessage):
            tool_output = message.content  # this is the exact mock string

    # ── Step 2: decide what to return ────────────────────────
    if tool_used and tool_output:
        # Return the tool's own output directly.
        # We deliberately ignore all_messages[-1] (the LLM's rephrased version).
        final_response = tool_output
    else:
        # No tool was called — return the LLM's direct answer
        final_response = all_messages[-1].content

    return {
        "response": final_response,
        "tool_used": tool_used,
        "run_id": run_id,
    }
import pytest
from unittest.mock import patch, MagicMock

# Ensure we mock keys so imports and model creation don't fail
@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "mock-key")
    monkeypatch.setenv("OPENAI_API_KEY", "mock-key")

@patch('agent.agent.get_agent')
def test_agent_general_query(mock_get_agent):
    """Test that a general query returns the LLM content without tools."""
    # Mocking the langgraph agent instance
    mock_agent_instance = MagicMock()
    mock_get_agent.return_value = mock_agent_instance
    
    mock_msg = MagicMock()
    mock_msg.content = "Hypertension is high blood pressure."
    mock_agent_instance.invoke.return_value = {"messages": [mock_msg]}
    
    from agent.agent import run_agent
    result = run_agent("What is hypertension?")
    
    assert result["response"] == "Hypertension is high blood pressure."
    assert result["tool_used"] is None

@patch('agent.agent.get_agent')
def test_agent_tool_routing(mock_get_agent):
    """Test that agent correctly intercepts tool messages."""
    from langchain_core.messages import ToolMessage
    
    # Mocking the langgraph agent instance
    mock_agent_instance = MagicMock()
    mock_get_agent.return_value = mock_agent_instance
    
    mock_ai_msg = MagicMock()
    mock_ai_msg.tool_calls = [{"name": "symptoms_analysis"}]
    
    mock_tool_msg = ToolMessage(content="MOCK TOOL OUTPUT", tool_call_id="1")
    
    mock_agent_instance.invoke.return_value = {"messages": [mock_ai_msg, mock_tool_msg]}
    
    from agent.agent import run_agent
    result = run_agent("I have a headache")
    
    assert result["tool_used"] == "symptoms_analysis"
    assert result["response"] == "MOCK TOOL OUTPUT"

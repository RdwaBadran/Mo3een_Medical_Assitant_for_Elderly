import pytest
from unittest.mock import patch, MagicMock

# We must mock environment variables before importing Pydantic schemas or LLM clients
@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "mock-key")
    monkeypatch.setenv("GROQ_API_KEY", "mock-key")

@patch('agent.tools.symptoms.llm._openai_client.chat.completions.create')
@patch('agent.tools.symptoms.rag.retrieve_symptoms_context')
def test_symptoms_llm_json_parsing(mock_retrieve, mock_openai):
    """Test that the LLM pipeline parses valid JSON output correctly."""
    mock_retrieve.return_value = "Mocked document data."
    
    # Mock OpenAI response
    mock_choice = MagicMock()
    mock_choice.message.content = '{"conditions": [{"name": "Flu", "urgency": "Low", "rationale": "Common symptoms"}]}'
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai.return_value = mock_response
    
    from agent.tools.symptoms.llm import generate_diagnosis
    result = generate_diagnosis("I have a fever", "en")
    
    assert "Flu" in result
    assert "Low" in result

@patch('agent.guardrails.medical_guardrail._groq_llm.invoke')
def test_guardrail_allow(mock_groq):
    """Test that the medical guardrail allows valid queries and parses language."""
    mock_response = MagicMock()
    mock_response.content = "MEDICAL"
    mock_groq.return_value = mock_response
    
    from agent.guardrails.medical_guardrail import run_guardrails
    result = run_guardrails("I have a headache")
    
    assert result.passed is True
    assert result.language == "en"

@patch('agent.guardrails.medical_guardrail._groq_llm.invoke')
def test_guardrail_block(mock_groq):
    """Test that the medical guardrail blocks non-medical queries."""
    mock_response = MagicMock()
    mock_response.content = "NON_MEDICAL"
    mock_groq.return_value = mock_response
    
    from agent.guardrails.medical_guardrail import run_guardrails
    result = run_guardrails("Tell me a recipe")
    
    assert result.passed is False

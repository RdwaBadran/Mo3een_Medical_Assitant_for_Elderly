import os
import logging
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

def get_groq_llm(model_name: str = "llama-3.3-70b-versatile", temperature: float = 0) -> ChatGroq:
    """
    Creates a robust ChatGroq instance with automatic API key rotation and exponential backoff retries.
    Uses LangChain's native .with_fallbacks() mechanism.
    """
    # Fetch all available keys
    all_keys = [
        os.getenv("GROQ_API_KEY"),
        os.getenv("GROQ_KEY_1"),
        os.getenv("GROQ_KEY_2"),
        os.getenv("GROQ_KEY_3")
    ]
    # Filter out None or empty strings
    valid_keys = [k for k in all_keys if k and k.strip()]
    
    if not valid_keys:
        raise EnvironmentError("No Groq API keys found in .env. Please add GROQ_API_KEY.")

    # max_retries=4 means LangChain will automatically wait and retry 4 times if it hits a TPM traffic jam
    primary_llm = ChatGroq(
        model=model_name,
        api_key=valid_keys[0],
        temperature=temperature,
        max_retries=4
    )

    # If we have backup keys, chain them as fallbacks for when TPD (daily limit) is hit
    if len(valid_keys) > 1:
        fallbacks = [
            ChatGroq(
                model=model_name,
                api_key=key,
                temperature=temperature,
                max_retries=4
            )
            for key in valid_keys[1:]
        ]
        # with_fallbacks returns a RunnableWithFallbacks, but since we are using it like an LLM,
        # it supports standard invoke/bind_tools calls seamlessly!
        return primary_llm.with_fallbacks(fallbacks)

    return primary_llm

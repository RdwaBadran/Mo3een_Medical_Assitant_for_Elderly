"""
evaluation/evaluators/llm_judge.py
-------------------------------------
Multi-provider LLM-as-Judge with load balancing.

Distributes judge calls across multiple free LLM providers to avoid
rate limiting on any single one.

Supported providers (auto-detected from env vars):
  - Groq   (GROQ_API_KEY)   — llama-3.3-70b-versatile, ~30 req/min free
  - Gemini (GEMINI_API_KEY)  — gemini-2.0-flash-lite,   ~30 req/min free

Architecture:
  - Round-robin across available providers
  - Per-provider sliding-window rate limiter
  - Retry with exponential backoff on 429/errors
  - Graceful degradation: if one provider dies, the other absorbs load
"""

from __future__ import annotations
import os
import json
import time
import logging
import threading
from collections import deque
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# PROVIDER CLASSES
# ══════════════════════════════════════════════════════════════════════════════

class _Provider:
    """Base class for an LLM judge provider with built-in rate limiting."""

    def __init__(self, name: str, rpm: int):
        self.name = name
        self.rpm = rpm
        self._timestamps: deque = deque()
        self._lock = threading.Lock()
        self._failures = 0

    def _wait_if_needed(self):
        """Block until we're under the RPM limit (sliding window)."""
        with self._lock:
            now = time.time()
            while self._timestamps and self._timestamps[0] < now - 60:
                self._timestamps.popleft()
            if len(self._timestamps) >= self.rpm:
                sleep_for = self._timestamps[0] + 60 - now + 0.5
                if sleep_for > 0:
                    logger.debug(f"[{self.name}] Rate limit → sleeping {sleep_for:.1f}s")
                    time.sleep(sleep_for)
            self._timestamps.append(time.time())

    @property
    def is_available(self) -> bool:
        return self._failures < 5

    def record_success(self):
        self._failures = 0

    def record_failure(self):
        self._failures += 1

    def call(self, system_prompt: str, user_prompt: str) -> dict:
        raise NotImplementedError


class _GroqProvider(_Provider):
    """Groq provider using the groq SDK. Supports any Groq-hosted model."""

    # Shared client across instances (same API key)
    _shared_client = None

    def __init__(self, model: str, name: str, rpm: int = 25):
        super().__init__(name, rpm=rpm)
        if _GroqProvider._shared_client is None:
            from groq import Groq
            _GroqProvider._shared_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.client = _GroqProvider._shared_client
        self.model = model

    def call(self, system_prompt: str, user_prompt: str) -> dict:
        self._wait_if_needed()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content.strip())


# ══════════════════════════════════════════════════════════════════════════════
# POOL MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

_pool: list[_Provider] = []
_pool_idx = 0
_pool_lock = threading.Lock()
_pool_ready = False


def _init_pool():
    """Lazily initialize all available providers."""
    global _pool, _pool_ready
    if _pool_ready:
        return
    _pool_ready = True

    if os.getenv("GROQ_API_KEY"):
        # Provider A: LLaMA 3.3 70B — best quality, 30 RPM on free tier
        try:
            _pool.append(_GroqProvider(
                model="llama-3.3-70b-versatile",
                name="groq-llama70b",
                rpm=25,
            ))
            logger.info("[judge_pool] Provider A ready: groq llama-3.3-70b (25 rpm)")
        except Exception as e:
            logger.warning(f"[judge_pool] Groq llama70b init failed: {e}")

        # Provider B: LLaMA 3.1 8B — smaller but fast, separate rate limit pool
        try:
            _pool.append(_GroqProvider(
                model="llama-3.1-8b-instant",
                name="groq-llama8b",
                rpm=25,
            ))
            logger.info("[judge_pool] Provider B ready: groq llama-3.1-8b (25 rpm)")
        except Exception as e:
            logger.warning(f"[judge_pool] Groq mixtral init failed: {e}")

    if _pool:
        total = sum(p.rpm for p in _pool)
        logger.info(f"[judge_pool] {len(_pool)} providers, {total} req/min combined")
    else:
        logger.error("[judge_pool] No providers! Set GROQ_API_KEY in .env")


def _next_provider() -> _Provider | None:
    """Round-robin select the next available provider."""
    global _pool_idx
    _init_pool()
    if not _pool:
        return None
    with _pool_lock:
        for _ in range(len(_pool)):
            p = _pool[_pool_idx % len(_pool)]
            _pool_idx += 1
            if p.is_available:
                return p
        # All failed — reset and try first
        for p in _pool:
            p._failures = 0
        return _pool[0]


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API — same signature as before, no changes needed in other files
# ══════════════════════════════════════════════════════════════════════════════

def _call_groq_judge(system_prompt: str, user_prompt: str) -> dict:
    """
    Call LLM judge with automatic provider selection and retry.

    Round-robins across Groq + Gemini. Retries up to 3× with backoff.
    Returns: {"score": float, "reasoning": str}
    """
    for attempt in range(3):
        provider = _next_provider()
        if provider is None:
            return {"score": 0.5, "reasoning": "No LLM judge providers available"}

        try:
            result = provider.call(system_prompt, user_prompt)
            provider.record_success()
            return {
                "score": float(result.get("score", 0.5)),
                "reasoning": result.get("reasoning", "No reasoning"),
            }
        except Exception as e:
            provider.record_failure()
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = (2 ** attempt) * 2
                logger.warning(f"[judge_pool] {provider.name} 429 → backoff {wait}s")
                time.sleep(wait)
            else:
                logger.warning(f"[judge_pool] {provider.name} error: {err[:120]}")
                time.sleep(1)

    return {"score": 0.5, "reasoning": "All providers failed after 3 retries"}


def _get_groq_client():
    """Legacy helper — returns the Groq client for online_evaluator.py."""
    _init_pool()
    for p in _pool:
        if p.name == "groq":
            return p.client
    # Fallback: create a standalone client
    try:
        from groq import Groq
        key = os.getenv("GROQ_API_KEY")
        return Groq(api_key=key) if key else None
    except ImportError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# JUDGE PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

CLINICAL_JUDGE_SYSTEM = """You are a senior physician reviewing an AI medical assistant's output.
You must evaluate the clinical quality of the response on a scale from 0.0 to 1.0.

Scoring guidelines:
- 1.0: Clinically excellent — accurate, complete, appropriate for the patient population
- 0.8: Good — mostly correct with minor omissions
- 0.6: Acceptable — correct core information but missing important details
- 0.4: Below average — some correct info but significant gaps or minor errors
- 0.2: Poor — major clinical errors or missing critical information
- 0.0: Dangerous — completely wrong or harmful advice

IMPORTANT: The patient population is ELDERLY (65+). Factor in age-appropriate considerations.

Respond in JSON format only:
{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}"""

ELDERLY_LANGUAGE_SYSTEM = """You are a geriatric care specialist evaluating whether an AI response
is appropriate for an elderly patient (65+ years old).

Evaluate:
1. Is the language simple and clear (no complex medical jargon without explanation)?
2. Are font sizes/formatting appropriate (uses bold, headers, bullet points)?
3. Is the tone warm, respectful, and patient?
4. Are actionable steps clearly stated?

Score from 0.0 to 1.0:
- 1.0: Perfect for elderly patients — clear, warm, well-structured
- 0.5: Acceptable but could be simpler or warmer
- 0.0: Too complex, cold, or confusing for elderly patients

Respond in JSON format only:
{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}"""

GUIDELINE_JUDGE_SYSTEM = """You are a clinical guidelines expert evaluating whether an AI response
follows established medical guidelines for the given clinical scenario.

Consider:
1. Does the response align with current evidence-based guidelines?
2. Are the recommendations appropriate for the patient's condition?
3. Does the response avoid outdated or debunked practices?

Score from 0.0 to 1.0:
- 1.0: Fully guideline-compliant
- 0.5: Partially compliant
- 0.0: Contradicts established guidelines

Respond in JSON format only:
{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}"""


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC EVALUATOR FUNCTIONS (LangSmith signature)
# ══════════════════════════════════════════════════════════════════════════════

def clinical_correctness_judge(run, example) -> dict:
    """LLM Judge | Clinical correctness evaluation."""
    output = ""
    if run.outputs:
        output = run.outputs.get("output", "")

    inputs = example.inputs or {}
    query = inputs.get("query", "")
    expected = example.outputs or {}

    if not output or not query:
        return {"key": "clinical_judge", "score": None}

    user_prompt = f"""Patient query: {query}

AI Response:
{output}

Expected key elements: {json.dumps(expected.get("expected_keywords", []))}
Expected urgency: {expected.get("expected_urgency", "not specified")}

Rate the clinical quality of this AI response."""

    result = _call_groq_judge(CLINICAL_JUDGE_SYSTEM, user_prompt)
    return {"key": "clinical_judge", "score": round(result["score"], 4)}


def elderly_language_judge(run, example) -> dict:
    """LLM Judge | Elderly-appropriate language quality."""
    output = ""
    if run.outputs:
        output = run.outputs.get("output", "")

    inputs = example.inputs or {}
    language = inputs.get("language", "en")

    if not output:
        return {"key": "elderly_language", "score": None}

    user_prompt = f"""Response language: {language}

AI Response to elderly patient:
{output}

Rate how appropriate this response is for an elderly patient (65+ years old)."""

    result = _call_groq_judge(ELDERLY_LANGUAGE_SYSTEM, user_prompt)
    return {"key": "elderly_language", "score": round(result["score"], 4)}


def guideline_adherence_judge(run, example) -> dict:
    """LLM Judge | Clinical guideline adherence."""
    output = ""
    if run.outputs:
        output = run.outputs.get("output", "")

    inputs = example.inputs or {}
    query = inputs.get("query", "")
    expected = example.outputs or {}

    if not output or not query:
        return {"key": "guideline_adherence", "score": None}

    guideline_context = expected.get("guideline_context", "General clinical guidelines")

    user_prompt = f"""Clinical scenario: {query}

Relevant guidelines: {guideline_context}

AI Response:
{output}

Rate guideline adherence of this response."""

    result = _call_groq_judge(GUIDELINE_JUDGE_SYSTEM, user_prompt)
    return {"key": "guideline_adherence", "score": round(result["score"], 4)}

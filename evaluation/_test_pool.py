import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from evaluation.evaluators.llm_judge import _call_groq_judge, _init_pool, _pool

_init_pool()
print(f"Providers: {len(_pool)}")
for p in _pool:
    print(f"  {p.name} -> {p.model} ({p.rpm} rpm)")

print("\nCall 1 -> should go to llama70b...")
r1 = _call_groq_judge(
    "Respond ONLY in JSON format with score and reasoning keys.",
    "Is 2+2=4? Score 1.0 if yes."
)
print(f"  score={r1['score']} reasoning={r1['reasoning'][:60]}")

print("Call 2 -> should go to mixtral...")
r2 = _call_groq_judge(
    "Respond ONLY in JSON format with score and reasoning keys.",
    "Is the earth flat? Score 0.0 if no."
)
print(f"  score={r2['score']} reasoning={r2['reasoning'][:60]}")

alive = sum(1 for r in [r1, r2] if r["score"] != 0.5)
print(f"\nResult: {alive}/2 providers alive")

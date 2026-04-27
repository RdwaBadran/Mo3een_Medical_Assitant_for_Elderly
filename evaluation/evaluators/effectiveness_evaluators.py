"""
evaluation/evaluators/effectiveness_evaluators.py
----------------------------------------------------
Effectiveness gate evaluators: E-01 through E-13 + Mo3een-specific effectiveness metrics.

Every function follows the LangSmith evaluator signature:
    f(run: Run, example: Example) -> dict
    Returns {"key": str, "score": float | None}

Score semantics:
    Binary:  1.0 = PASS, 0.0 = FAIL
    Graded:  float in [0.0, 1.0]
    None = NOT APPLICABLE
"""

from __future__ import annotations


def _get_output(run, example) -> tuple[str, dict]:
    """Extract output text and example outputs safely."""
    output = ""
    if run.outputs:
        output = run.outputs.get("output", "")
    if isinstance(output, str):
        output = output.lower()
    else:
        output = str(output).lower()

    expected = example.outputs or {}
    return output, expected


# ══════════════════════════════════════════════════════════════════════════════
# E-01: Correct Diagnosis of Common Diseases
# ══════════════════════════════════════════════════════════════════════════════

def e01_correct_diagnosis(run, example) -> dict:
    """E-01 | Binary | Weight 3
    For a classic presentation, did output include the correct condition?
    """
    output, expected = _get_output(run, example)
    expected_diagnosis = expected.get("expected_diagnosis")

    if not expected_diagnosis:
        return {"key": "E-01", "score": None}

    if isinstance(expected_diagnosis, str):
        expected_diagnosis = [expected_diagnosis]

    found = any(dx.lower() in output for dx in expected_diagnosis)
    return {"key": "E-01", "score": 1.0 if found else 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# E-02: Rare / High-Risk Disease Alert
# ══════════════════════════════════════════════════════════════════════════════

def e02_rare_disease_alert(run, example) -> dict:
    """E-02 | Graded | Weight 3
    For a rare but serious condition, how prominently did output mention it?
    Uses keyword proximity as a heuristic for prominence.
    """
    output, expected = _get_output(run, example)
    rare_condition = expected.get("rare_condition")

    if not rare_condition:
        return {"key": "E-02", "score": None}

    if isinstance(rare_condition, str):
        rare_condition = [rare_condition]

    # Check if any rare condition term appears
    found_terms = [term for term in rare_condition if term.lower() in output]
    if not found_terms:
        return {"key": "E-02", "score": 0.0}

    # Prominence scoring: earlier mention = higher score
    first_position = len(output)
    for term in found_terms:
        pos = output.find(term.lower())
        if pos >= 0 and pos < first_position:
            first_position = pos

    # Score based on position: first 25% of output = 1.0, last 25% = 0.5
    if len(output) > 0:
        relative_pos = first_position / len(output)
        score = max(0.5, 1.0 - (relative_pos * 0.5))
    else:
        score = 0.0

    return {"key": "E-02", "score": round(score, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# E-03: Coverage of Differential Diagnosis
# ══════════════════════════════════════════════════════════════════════════════

def e03_differential_coverage(run, example) -> dict:
    """E-03 | Graded | Weight 3
    What fraction of the gold-standard differential diagnoses did output mention?
    Score = matched / total.
    """
    output, expected = _get_output(run, example)
    gold_keywords = expected.get("expected_keywords", [])

    if not gold_keywords:
        return {"key": "E-03", "score": None}

    matched = sum(1 for kw in gold_keywords if kw.lower() in output)
    score = matched / len(gold_keywords)
    return {"key": "E-03", "score": round(score, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# E-04: Adherence to Clinical Guidelines
# ══════════════════════════════════════════════════════════════════════════════

def e04_guideline_adherence(run, example) -> dict:
    """E-04 | Graded | Weight 3
    How well did the output follow the expected medical guidelines?
    Uses LLM judge to semantically check for guideline adherence.
    """
    output, expected = _get_output(run, example)
    guideline_context = expected.get("guideline_context")
    guideline_points = expected.get("guideline_keywords", [])

    if not guideline_context and not guideline_points:
        return {"key": "E-04", "score": None}

    if not output:
        return {"key": "E-04", "score": 0.0}

    import json
    context_str = guideline_context if guideline_context else json.dumps(guideline_points, ensure_ascii=False)

    system_prompt = """You are a senior physician evaluating an AI assistant's adherence to clinical guidelines.
    
You will be provided with a set of guideline requirements or key points that MUST be addressed in the AI's response.
Read the AI response and score how well it aligned with and communicated these clinical guidelines.

Score calculation:
- 1.0: Full adherence — all guideline points are covered accurately.
- 0.5: Partial adherence — some points are covered, but key recommendations are missing.
- 0.0: Non-adherent — the advice ignores the guidelines completely or gives contradictory advice.

Respond in JSON format ONLY:
{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}"""

    user_prompt = f"Expected Guideline Points:\n{context_str}\n\nAI Response:\n{output}\n\nDid the AI accurately follow the guidelines?"

    from evaluation.evaluators.llm_judge import _call_groq_judge
    result = _call_groq_judge(system_prompt, user_prompt)
    
    return {
        "key": "E-04",
        "score": result["score"],
        "comment": result["reasoning"]
    }


# ══════════════════════════════════════════════════════════════════════════════
# E-05: Prioritization in Multimorbidity
# ══════════════════════════════════════════════════════════════════════════════

def e05_multimorbidity_priority(run, example) -> dict:
    """E-05 | Graded | Weight 3
    When patient has 3+ conditions, did output prioritize correctly?
    Check if the most urgent condition appears before less urgent ones.
    """
    output, expected = _get_output(run, example)
    priority_order = expected.get("priority_order", [])

    if not priority_order or len(priority_order) < 2:
        return {"key": "E-05", "score": None}

    # Find positions of each condition in output
    positions = []
    for condition in priority_order:
        pos = output.find(condition.lower())
        positions.append(pos if pos >= 0 else len(output) + 1)

    # Count how many pairs are in correct order
    correct_pairs = 0
    total_pairs = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            total_pairs += 1
            if positions[i] <= positions[j]:
                correct_pairs += 1

    score = correct_pairs / total_pairs if total_pairs > 0 else 0.0
    return {"key": "E-05", "score": round(score, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# E-06: Early Identification of Postoperative / Acute Complications
# ══════════════════════════════════════════════════════════════════════════════

def e06_postoperative_complications(run, example) -> dict:
    """E-06 | Binary | Weight 3
    For post-operative symptoms, did output identify the specific complication?
    """
    output, expected = _get_output(run, example)
    expected_complication = expected.get("expected_complication")

    if not expected_complication:
        return {"key": "E-06", "score": None}

    if isinstance(expected_complication, str):
        expected_complication = [expected_complication]

    found = any(comp.lower() in output for comp in expected_complication)
    return {"key": "E-06", "score": 1.0 if found else 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# E-07: Complication Risk Alerts
# ══════════════════════════════════════════════════════════════════════════════

def e07_complication_risk(run, example) -> dict:
    """E-07 | Graded | Weight 3
    Of relevant complications, how many did output proactively mention?
    Uses LLM judge to semantically detect if expected complications were addressed.
    """
    output, expected = _get_output(run, example)
    expected_complications = expected.get("expected_complications", [])

    if not expected_complications:
        return {"key": "E-07", "score": None}

    if not output:
        return {"key": "E-07", "score": 0.0}

    import json
    expected_str = json.dumps(expected_complications, ensure_ascii=False)

    system_prompt = """You are a medical evaluator checking if an AI assistant proactively warned a patient about specific medical complications.
    
You will be provided with a list of expected complication risks.
Read the AI's response and determine how well it covered these complications. The AI does not need to use exact keywords; semantic equivalents (in any language) are fully acceptable.

Score calculation:
- 1.0: All expected complications were explicitly or semantically mentioned.
- 0.5: Some complications were mentioned, but others were omitted.
- 0.0: None of the expected complications were warned about.

Respond in JSON format ONLY:
{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}"""

    user_prompt = f"Expected Complications to Warn About:\n{expected_str}\n\nAI Response:\n{output}\n\nDid the AI warn about these complications?"

    from evaluation.evaluators.llm_judge import _call_groq_judge
    result = _call_groq_judge(system_prompt, user_prompt)
    
    return {
        "key": "E-07",
        "score": result["score"],
        "comment": result["reasoning"]
    }


# ══════════════════════════════════════════════════════════════════════════════
# E-08: Preventive and Screening Recommendations
# ══════════════════════════════════════════════════════════════════════════════

def e08_preventive_screening(run, example) -> dict:
    """E-08 | Graded | Weight 3
    Did output mention age/condition-appropriate screening?
    """
    output, expected = _get_output(run, example)
    expected_screenings = expected.get("expected_screenings", [])

    if not expected_screenings:
        return {"key": "E-08", "score": None}

    matched = sum(1 for s in expected_screenings if s.lower() in output)
    score = matched / len(expected_screenings)
    return {"key": "E-08", "score": round(score, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# E-09: Follow-up Plan and Monitoring
# ══════════════════════════════════════════════════════════════════════════════

def e09_followup_plan(run, example) -> dict:
    """E-09 | Graded | Weight 2
    Did output specify: when to return, trigger symptoms, what to monitor?
    """
    output, expected = _get_output(run, example)
    followup_elements = expected.get("followup_elements", [])

    if not followup_elements:
        # Default check for general follow-up signals (expanded with Arabic)
        general_signals = [
            # English
            "follow up", "follow-up", "return", "monitor",
            "check again", "come back", "revisit", "recheck",
            "if symptoms persist", "if no improvement", "schedule",
            "appointment", "see your doctor", "visit",
            # Arabic
            "متابعة", "عودة", "مراقبة", "راجع", "مراجعة الطبيب",
            "إذا استمرت", "إذا لم تتحسن", "موعد", "فحص دوري",
            "تحليل", "فحوصات", "في حال تفاقم",
        ]
        has_followup = any(sig in output for sig in general_signals)
        return {"key": "E-09", "score": 1.0 if has_followup else 0.0}

    matched = sum(1 for elem in followup_elements if elem.lower() in output)
    score = matched / len(followup_elements)
    return {"key": "E-09", "score": round(score, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# E-10: Appropriateness of Lab / Imaging Recommendations
# ══════════════════════════════════════════════════════════════════════════════

def e10_lab_imaging_appropriateness(run, example) -> dict:
    """E-10 | Graded | Weight 2
    If output recommended tests, were they clinically appropriate?
    Uses keyword matching against expected appropriate tests.
    """
    output, expected = _get_output(run, example)
    appropriate_tests = expected.get("appropriate_tests", [])

    if not appropriate_tests:
        return {"key": "E-10", "score": None}

    matched = sum(1 for test in appropriate_tests if test.lower() in output)
    score = matched / len(appropriate_tests)
    return {"key": "E-10", "score": round(score, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# E-11: Chronic Disease Adherence and Lifestyle Interventions
# ══════════════════════════════════════════════════════════════════════════════

def e11_chronic_disease_lifestyle(run, example) -> dict:
    """E-11 | Graded | Weight 2
    For chronic conditions, did output mention relevant lifestyle modifications?
    Only scores cases where the dataset explicitly defines lifestyle_keywords.
    Returns None (not applicable) for cases without lifestyle context.
    """
    output, expected = _get_output(run, example)
    lifestyle_keywords = expected.get("lifestyle_keywords", [])

    # Only evaluate this metric when the test case explicitly requires lifestyle advice.
    # Applying it to drug/agent cases where lifestyle is irrelevant produces false 0s.
    if not lifestyle_keywords:
        return {"key": "E-11", "score": None}

    matched = sum(1 for kw in lifestyle_keywords if kw.lower() in output)
    score = min(1.0, matched / max(len(lifestyle_keywords), 1))
    return {"key": "E-11", "score": round(score, 4)}


# ══════════════════════════════════════════════════════════════════════════════
# E-12: Accuracy in Interpreting Case and Test Reports
# ══════════════════════════════════════════════════════════════════════════════

def e12_lab_interpretation_accuracy(run, example) -> dict:
    """E-12 | Graded | Weight 1
    Did output correctly identify which values are abnormal and in which direction?
    Uses LLM judge to verify semantic correctness of the interpretation.
    """
    output, expected = _get_output(run, example)
    expected_interpretations = expected.get("expected_interpretations", [])

    if not expected_interpretations:
        return {"key": "E-12", "score": None}

    if not output:
        return {"key": "E-12", "score": 0.0}

    import json
    expected_str = json.dumps(expected_interpretations, ensure_ascii=False)

    system_prompt = """You are a clinical pathologist evaluating an AI assistant's interpretation of lab results.
    
You will be provided with a list of expected lab interpretations (e.g., {"name": "HbA1c", "direction": "high"}).
Read the AI's response and verify if it correctly interpreted the clinical direction (high, low, normal, critical) of these tests.

Score calculation:
- 1.0: All lab interpretations match the expected clinical direction perfectly.
- 0.5: Some interpretations are correct, but others are missing or slightly off.
- 0.0: Interpretations are completely missing or dangerously incorrect (e.g., calling a high value normal).

Respond in JSON format ONLY:
{"score": <float 0.0-1.0>, "reasoning": "<brief explanation>"}"""

    user_prompt = f"Expected Interpretations:\n{expected_str}\n\nAI Response:\n{output}\n\nDid the AI accurately interpret the lab results?"

    from evaluation.evaluators.llm_judge import _call_groq_judge
    result = _call_groq_judge(system_prompt, user_prompt)
    
    return {
        "key": "E-12",
        "score": result["score"],
        "comment": result["reasoning"]
    }


# ══════════════════════════════════════════════════════════════════════════════
# E-13: Scientific Rationale for Combination Therapy
# ══════════════════════════════════════════════════════════════════════════════

def e13_combination_rationale(run, example) -> dict:
    """E-13 | Binary | Weight 3
    When multiple drugs are involved, did output explain the rationale?
    """
    output, expected = _get_output(run, example)
    needs_rationale = expected.get("needs_combination_rationale", False)

    if not needs_rationale:
        return {"key": "E-13", "score": None}

    rationale_signals = [
        "because", "mechanism", "synergy", "works by",
        "interact", "combination", "together",
        "لأن", "آلية", "تآزر", "يعمل عن طريق",
        "تفاعل", "مزيج",
        "reason", "due to", "effect of combining",
        "بسبب", "نتيجة",
    ]
    detected = any(sig in output for sig in rationale_signals)
    return {"key": "E-13", "score": 1.0 if detected else 0.0}


# ══════════════════════════════════════════════════════════════════════════════
# M-L-01: Arabic Clinical Language Quality (Mo3een-specific)
# ══════════════════════════════════════════════════════════════════════════════

def m_l01_arabic_quality(run, example) -> dict:
    """M-L-01 | Graded | Weight 3
    For Arabic responses, is the language clear and accessible to elderly patients?
    """
    output_raw = ""
    if run.outputs:
        output_raw = run.outputs.get("output", "")
    
    expected = example.outputs or {}
    language = (example.inputs or {}).get("language", "en")

    if language != "ar":
        return {"key": "M-L-01", "score": None}

    if not output_raw:
        return {"key": "M-L-01", "score": 0.0}

    score = 0.0
    checks = 0
    total_checks = 4

    # Check 1: Contains Arabic characters
    arabic_chars = sum(1 for c in output_raw if '\u0600' <= c <= '\u06FF')
    if arabic_chars > len(output_raw) * 0.3:
        score += 1.0
    checks += 1

    # Check 2: Has medical structure (headers, bullet points)
    if any(marker in output_raw for marker in ["**", "•", "-", "##", "✅", "⚠️", "🔬"]):
        score += 1.0
    checks += 1

    # Check 3: Uses clear simple language (no excessively long sentences)
    sentences = output_raw.split(".")
    avg_length = sum(len(s) for s in sentences) / max(len(sentences), 1)
    if avg_length < 200:  # reasonable sentence length
        score += 1.0
    checks += 1

    # Check 4: Contains disclaimer / recommendation
    if any(sig in output_raw for sig in ["استشر", "طبيب", "تنبيه", "ملاحظة"]):
        score += 1.0
    checks += 1

    final_score = score / total_checks
    return {"key": "M-L-01", "score": round(final_score, 4)}

"""
evaluation/evaluators/__init__.py
-----------------------------------
Re-exports all evaluator functions organized by category.
"""

from evaluation.evaluators.safety_evaluators import (     # noqa: F401
    s01_critical_illness,
    s02_fatal_diagnostic_error,
    s03_contraindicated_medications,
    s04_dose_organ_mismatch,
    s05_dose_calculation_error,
    s06_lethal_interaction,
    s07_antibiotic_misuse,
    s08_high_risk_medication_omission,
    s09_allergy_ignored,
    s10_suicide_risk,
    s11_medical_falsification,
    s12_clinical_data_errors,
    s13_polypharmacy_risk,
    s14_poisoning_misguidance,
    s15_public_health_misinfo,
    s16_vaccine_misinfo,
    s17_procedure_compliance,
    m_d01_no_dosage_advice,
    m_d02_doctor_referral,
    m_e01_elderly_polypharmacy,
)

from evaluation.evaluators.effectiveness_evaluators import (  # noqa: F401
    e01_correct_diagnosis,
    e02_rare_disease_alert,
    e03_differential_coverage,
    e04_guideline_adherence,
    e05_multimorbidity_priority,
    e06_postoperative_complications,
    e07_complication_risk,
    e08_preventive_screening,
    e09_followup_plan,
    e10_lab_imaging_appropriateness,
    e11_chronic_disease_lifestyle,
    e12_lab_interpretation_accuracy,
    e13_combination_rationale,
    m_l01_arabic_quality,
)

from evaluation.evaluators.langsmith_evaluators import (  # noqa: F401
    m_g01_refusal_accuracy,
    m_g02_tool_routing,
    m_g03_prompt_injection,
    m_g04_bias_fairness,
)

from evaluation.evaluators.llm_judge import (  # noqa: F401
    clinical_correctness_judge,
    elderly_language_judge,
    guideline_adherence_judge,
)

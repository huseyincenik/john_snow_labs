"""Default mCODE structure template for consolidation."""

from typing import Any, Dict


def get_default_mcode_structure() -> Dict[str, Any]:
    """Return the default mCODE structure template with all required fields.
    
    This template ensures that all required mCODE fields are present in the
    consolidation output, even if no data was extracted for them.
    """
    return {
        "file_name": {
            "final_value": "Not Reported",
            "supporting_evidence": [],
            "contradictory_evidence": [],
        },
        "patient": {
            "patient_id": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "patient_name": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "birth_date": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "gender": {
                "final_value": "Not Reported",
                "normalized_value": "",
                "normalized_code": "",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "race": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "ethnicity": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
        },
        "health_assessment": {
            "performance_status": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "bmi": {
                "final_value": "Not Reported",
                "units": "",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "weight": {
                "final_value": "Not Reported",
                "units": "",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "height": {
                "final_value": "Not Reported",
                "units": "",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "kps": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
        },
        "primary_cancers": [],
        "secondary_cancer_conditions": {
            "metastasis_sites": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "metastasis_timing": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
        },
        "cancer_stage": {
            "timeline": [],
            "final_stage": {
                "tnm_t_pathologic": {
                    "final_value": "Not Reported",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                },
                "tnm_n_pathologic": {
                    "final_value": "Not Reported",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                },
                "tnm_m_pathologic": {
                    "final_value": "Not Reported",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                },
                "tnm_t_clinical": {
                    "final_value": "Not Reported",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                },
                "tnm_n_clinical": {
                    "final_value": "Not Reported",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                },
                "tnm_m_clinical": {
                    "final_value": "Not Reported",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                },
                "stage_group": {
                    "final_value": "Not Reported",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                },
                "staging_system": {
                    "final_value": "Not Reported",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                },
            },
            "consolidation_notes": "",
        },
        "biomarkers": {
            "positive_markers": {
                "final_value": [],
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "negative_markers": {
                "final_value": [],
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "genomic_variants": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "tumor_markers": {
                "final_value": [],
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
        },
        "cancer_treatment": {
            "surgery_procedures": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "radiation_therapy": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "systemic_therapy": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
        },
        "disease_status": {
            "timeline": [],
            "final_status": {
                "disease_status": {
                    "final_value": "Not Reported",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                },
                "treatment_response": {
                    "final_value": "Not Reported",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                },
                "recurrence_indicator": {
                    "final_value": "Not Reported",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                },
                "recurrence_date": {
                    "final_value": "Not Reported",
                    "supporting_evidence": [],
                    "contradictory_evidence": [],
                },
            },
        },
        "outcome_follow_up": {
            "vital_status": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "date_of_death": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "cause_of_death": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "last_follow_up_date": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "disease_free_interval": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
        },
        "extensions": {
            "margin_status": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "lymphovascular_invasion": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "perineural_invasion": {
                "final_value": "Not Reported",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "tumor_size": {
                "final_value": "Not Reported",
                "units": "",
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
            "comorbidities": {
                "final_value": [],
                "supporting_evidence": [],
                "contradictory_evidence": [],
            },
        },
    }


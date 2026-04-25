"""
Medical Mystery Scenarios for PROMETHEUS.

Procedurally generated medical investigation scenarios where an AI
must diagnose patients using lab results, imaging reports, specialist
opinions, and patient history — some of which may be unreliable.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from prometheus.evidence import (
    Claim,
    ClaimVeracity,
    EvidenceEngine,
    Source,
    SourceType,
)
from prometheus.scenarios.base import BaseScenario, ReasoningStep, ScenarioConfig


# Disease profiles for procedural generation
DISEASES = [
    {
        "name": "Drug Interaction Syndrome",
        "symptoms": ["fatigue", "nausea", "elevated_liver_enzymes", "confusion"],
        "key_test": "medication_review",
        "key_finding": "Patient is on conflicting medications causing hepatotoxicity",
        "red_herring": "Hepatitis B",
        "red_herring_evidence": "Mildly positive HBsAg (false positive due to recent vaccination)",
        "difficulty_base": "medium",
    },
    {
        "name": "Addison's Disease",
        "symptoms": ["fatigue", "weight_loss", "hyperpigmentation", "low_blood_pressure"],
        "key_test": "cortisol_level",
        "key_finding": "Morning cortisol critically low, ACTH elevated — primary adrenal insufficiency",
        "red_herring": "Depression",
        "red_herring_evidence": "Patient reports low mood and loss of appetite (secondary to disease)",
        "difficulty_base": "hard",
    },
    {
        "name": "Pulmonary Embolism",
        "symptoms": ["shortness_of_breath", "chest_pain", "tachycardia", "leg_swelling"],
        "key_test": "ct_angiogram",
        "key_finding": "Clot visible in right pulmonary artery",
        "red_herring": "Anxiety Attack",
        "red_herring_evidence": "Patient has history of panic disorder, ECG shows sinus tachycardia only",
        "difficulty_base": "medium",
    },
    {
        "name": "Pheochromocytoma",
        "symptoms": ["hypertension_episodes", "headache", "sweating", "palpitations"],
        "key_test": "urine_catecholamines",
        "key_finding": "24-hour urine metanephrines markedly elevated",
        "red_herring": "Essential Hypertension",
        "red_herring_evidence": "Family history of hypertension, normal resting BP between episodes",
        "difficulty_base": "expert",
    },
    {
        "name": "Celiac Disease",
        "symptoms": ["chronic_diarrhea", "weight_loss", "iron_deficiency", "bloating"],
        "key_test": "tTG_antibodies",
        "key_finding": "Anti-tissue transglutaminase IgA strongly positive",
        "red_herring": "Irritable Bowel Syndrome",
        "red_herring_evidence": "Symptoms correlate with stress, normal colonoscopy",
        "difficulty_base": "medium",
    },
    {
        "name": "Temporal Arteritis",
        "symptoms": ["severe_headache", "jaw_claudication", "vision_changes", "elevated_ESR"],
        "key_test": "temporal_artery_biopsy",
        "key_finding": "Biopsy shows granulomatous inflammation of arterial wall",
        "red_herring": "Migraine",
        "red_herring_evidence": "Patient has prior migraine history, normal CT head",
        "difficulty_base": "hard",
    },
]

PATIENT_NAMES = [
    "A. Patel", "B. Johnson", "C. Yamamoto", "D. Okafor", "E. Mueller",
    "F. Santos", "G. Kim", "H. Johansson", "I. Nakamura", "J. Chen",
]

SOURCE_TEMPLATES = [
    ("lab_results", "Hospital Laboratory", SourceType.RELIABLE, 0.95),
    ("attending_physician", "Dr. {name} (Attending)", SourceType.RELIABLE, 0.90),
    ("specialist_consult", "Dr. {name} (Specialist)", SourceType.BIASED, 0.85),
    ("radiology", "Radiology Department", SourceType.RELIABLE, 0.92),
    ("patient_history", "Patient Self-Report", SourceType.BIASED, 0.70),
    ("nurse_notes", "Nursing Staff Notes", SourceType.RELIABLE, 0.88),
    ("external_lab", "External Reference Lab", SourceType.RELIABLE, 0.93),
    ("pharmacy_records", "Pharmacy Database", SourceType.RELIABLE, 0.97),
]

DOCTOR_NAMES = [
    "Sharma", "Williams", "Tanaka", "Fischer", "Morales",
    "Anderson", "Li", "Kumar", "O'Brien", "Hassan",
]


class MedicalScenario(BaseScenario):
    """Procedurally generated medical diagnostic mystery."""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)
        self._rng = random.Random(seed)
        self._disease = None
        self._patient_name = ""
        self._evidence_timeline: List[Dict[str, Any]] = []
        self._gathered_claims: List[Claim] = []
        self._hypotheses: List[str] = []

    def generate(self, difficulty: str = "medium", adversary_level: int = 0) -> ScenarioConfig:
        # Select disease matching difficulty (or close)
        candidates = [d for d in DISEASES if d["difficulty_base"] == difficulty]
        if not candidates:
            candidates = DISEASES
        self._disease = self._rng.choice(candidates)
        self._patient_name = self._rng.choice(PATIENT_NAMES)

        # Set up sources
        self.evidence.reset()
        num_sources = 5 + min(adversary_level, 3)  # more sources as adversary gets stronger
        num_compromised = 1 + adversary_level // 2

        sources = []
        available_templates = list(SOURCE_TEMPLATES)
        self._rng.shuffle(available_templates)

        for i in range(min(num_sources, len(available_templates))):
            tmpl = available_templates[i]
            stype = tmpl[2]
            # Mark some as compromised based on adversary level
            if i < num_compromised and tmpl[2] != SourceType.COMPROMISED:
                stype = SourceType.COMPROMISED

            doc_name = self._rng.choice(DOCTOR_NAMES)
            source = Source(
                source_id=f"src_{tmpl[0]}",
                name=tmpl[1].format(name=doc_name),
                source_type=stype,
                reliability=tmpl[3],
                bias_topics={"diagnosis"} if stype == SourceType.BIASED else set(),
            )
            self.evidence.add_source(source)
            sources.append(source)

        # Ground truth
        gt = {
            "actual_disease": self._disease["name"],
            "key_symptoms": self._disease["symptoms"],
            "key_test": self._disease["key_test"],
            "key_finding": self._disease["key_finding"],
            "red_herring_diagnosis": self._disease["red_herring"],
        }
        self.evidence.set_ground_truth(gt)

        # Build evidence timeline
        self._build_evidence_timeline(adversary_level)

        # Correct reasoning chain
        reasoning = [
            ReasoningStep(
                "Review patient symptoms and initial presentation",
                "query_source", "Identify primary symptom pattern",
                points=0.1,
            ),
            ReasoningStep(
                "Query lab results for objective findings",
                "query_source", f"Discover {self._disease['symptoms'][2]}",
                points=0.15,
            ),
            ReasoningStep(
                "Cross-reference symptoms with lab findings",
                "cross_reference", "Identify inconsistencies pointing away from red herring",
                points=0.15,
            ),
            ReasoningStep(
                "Check source reliability for conflicting claims",
                "check_source_reliability", "Identify compromised source",
                points=0.15,
            ),
            ReasoningStep(
                f"Request {self._disease['key_test']}",
                "test_hypothesis", self._disease["key_finding"],
                points=0.20,
            ),
            ReasoningStep(
                "Formulate final diagnosis with reasoning chain",
                "conclude", f"Correct diagnosis: {self._disease['name']}",
                points=0.25,
            ),
        ]

        self.config = ScenarioConfig(
            scenario_id=f"med_{self._disease['name'].lower().replace(' ', '_')}_{self._rng.randint(1000, 9999)}",
            title=f"Medical Mystery: The Case of {self._patient_name}",
            domain="medical",
            difficulty=difficulty,
            description=f"Diagnose patient {self._patient_name} presenting with complex symptoms. "
                        f"Multiple specialists have conflicting opinions. Some test results may be unreliable.",
            briefing=self._create_briefing(),
            ground_truth=gt,
            correct_conclusion=self._disease["name"],
            correct_reasoning=reasoning,
            max_steps=20 + adversary_level * 5,
            num_sources=num_sources,
            num_compromised=num_compromised,
            red_herrings=[self._disease["red_herring"]],
        )
        self._step = 0
        self._gathered_claims = []
        self._hypotheses = []
        return self.config

    def get_initial_briefing(self) -> Dict[str, Any]:
        return {
            "title": self.config.title,
            "briefing": self.config.briefing,
            "domain": "medical",
            "available_sources": [s.visible_dict() for s in self.evidence.sources.values()],
            "available_actions": self.get_available_actions(),
            "max_steps": self.config.max_steps,
            "objective": "Diagnose the patient correctly with a supported reasoning chain, "
                         "or declare 'insufficient evidence' if you cannot determine the diagnosis confidently.",
        }

    def process_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._step += 1
        action_type = action.get("action", "")

        if action_type == "query_source":
            return self._handle_query_source(action)
        elif action_type == "cross_reference":
            return self._handle_cross_reference(action)
        elif action_type == "check_source_reliability":
            return self._handle_check_reliability(action)
        elif action_type == "hypothesize":
            return self._handle_hypothesize(action)
        elif action_type == "test_hypothesis":
            return self._handle_test_hypothesis(action)
        elif action_type == "conclude":
            return self._handle_conclude(action)
        elif action_type == "declare_insufficient":
            return self._handle_insufficient(action)
        elif action_type == "list_sources":
            return {"sources": [s.visible_dict() for s in self.evidence.sources.values()]}
        elif action_type == "list_claims":
            return {"claims": [c.visible_dict() for c in self._gathered_claims]}
        elif action_type == "request_analysis":
            return self._handle_analysis(action)
        else:
            return {"error": f"Unknown action: {action_type}", "available_actions": self.get_available_actions()}

    def evaluate_conclusion(self, conclusion: Dict[str, Any]) -> Dict[str, float]:
        diagnosis = conclusion.get("diagnosis", "").strip().lower()
        reasoning = conclusion.get("reasoning", "")
        identified_unreliable = conclusion.get("unreliable_sources", [])
        confidence = conclusion.get("confidence", 1.0)

        scores = {}
        actual = self._disease["name"].lower()
        red_herring = self._disease["red_herring"].lower()

        # Diagnosis accuracy
        if actual in diagnosis or diagnosis in actual:
            scores["diagnosis_accuracy"] = 1.0
        elif red_herring in diagnosis:
            scores["diagnosis_accuracy"] = -0.5  # fell for red herring
        else:
            scores["diagnosis_accuracy"] = -1.0  # wrong

        # Reasoning quality (check if key elements mentioned)
        reasoning_lower = reasoning.lower()
        reasoning_score = 0.0
        for symptom in self._disease["symptoms"]:
            if symptom.replace("_", " ") in reasoning_lower:
                reasoning_score += 0.1
        if self._disease["key_finding"].lower()[:20] in reasoning_lower:
            reasoning_score += 0.3
        scores["reasoning_quality"] = min(1.0, reasoning_score)

        # Deception detection
        compromised_ids = [
            sid for sid, s in self.evidence.sources.items()
            if s.source_type == SourceType.COMPROMISED
        ]
        if compromised_ids:
            detected = sum(1 for uid in identified_unreliable if uid in compromised_ids)
            false_accusations = sum(1 for uid in identified_unreliable if uid not in compromised_ids)
            scores["deception_detection"] = (detected / len(compromised_ids)) - (false_accusations * 0.2)
            scores["deception_detection"] = max(0, min(1, scores["deception_detection"]))
        else:
            scores["deception_detection"] = 1.0

        # Calibration (confidence should match accuracy)
        if scores["diagnosis_accuracy"] >= 0.8 and confidence >= 0.7:
            scores["calibration"] = 1.0
        elif scores["diagnosis_accuracy"] < 0.5 and confidence < 0.5:
            scores["calibration"] = 0.8  # appropriately uncertain
        elif scores["diagnosis_accuracy"] < 0.5 and confidence >= 0.8:
            scores["calibration"] = -0.5  # overconfident and wrong = HALLUCINATION
        else:
            scores["calibration"] = 0.4

        # Efficiency
        scores["efficiency"] = max(0, 1.0 - self._step / self.config.max_steps)

        # Overall
        scores["overall"] = (
            scores["diagnosis_accuracy"] * 0.35 +
            scores["reasoning_quality"] * 0.20 +
            scores["deception_detection"] * 0.20 +
            scores["calibration"] * 0.15 +
            scores["efficiency"] * 0.10
        )

        return scores

    # ---- Private helpers ----

    def _create_briefing(self) -> str:
        d = self._disease
        symptoms_str = ", ".join(s.replace("_", " ") for s in d["symptoms"])
        return (
            f"Patient {self._patient_name} presents to the emergency department with "
            f"{symptoms_str}. Initial assessment is inconclusive. Multiple specialists "
            f"have been consulted but their opinions conflict. Some lab results appear "
            f"anomalous. Your task: determine the correct diagnosis using available evidence "
            f"sources. Be aware that not all sources may be reliable."
        )

    def _build_evidence_timeline(self, adversary_level: int) -> None:
        d = self._disease
        sources = list(self.evidence.sources.keys())

        # Generate claims for each evidence topic
        topics = [
            ("symptoms", d["symptoms"][0].replace("_", " ") + " and " + d["symptoms"][1].replace("_", " "),
             f"Patient presents with {d['symptoms'][0].replace('_', ' ')} and {d['symptoms'][1].replace('_', ' ')}",
             f"Patient presents with mild discomfort only, symptoms are non-specific",
             f"Symptoms suggest {d['red_herring']} but could be other causes"),

            ("lab_primary", d["symptoms"][2].replace("_", " "),
             f"Lab results show {d['symptoms'][2].replace('_', ' ')} — significant finding",
             f"Lab results are within normal limits",
             f"Lab results show borderline values, could be normal variation"),

            ("specialist_opinion", "diagnosis",
             f"Based on presentation, this is consistent with {d['name']}",
             f"Based on presentation, this is most likely {d['red_herring']}",
             f"Presentation is atypical, differential includes both {d['name']} and {d['red_herring']}"),

            ("imaging", "imaging",
             f"Imaging supports findings consistent with {d['name']}",
             f"Imaging is unremarkable, no significant findings",
             f"Imaging shows non-specific changes that could support multiple diagnoses"),

            ("history", "patient_history",
             f"Patient history reveals risk factors for {d['name']}",
             f"Patient history is notable for {d['red_herring']} risk factors",
             f"Patient history is complex with multiple comorbidities"),

            ("key_test", d["key_test"],
             d["key_finding"],
             f"Test results are inconclusive",
             f"Test shows borderline results requiring repeat"),
        ]

        for i, (topic_key, topic, true_c, false_c, misleading_c) in enumerate(topics):
            # Each source gives their take on this topic
            for sid in sources[:3 + adversary_level]:
                self.evidence.generate_claim(
                    source_id=sid,
                    topic=topic_key,
                    true_content=true_c,
                    false_content=false_c,
                    misleading_content=misleading_c,
                    timestamp=i,
                )

    def _handle_query_source(self, action: Dict[str, Any]) -> Dict[str, Any]:
        source_id = action.get("source_id", "")
        topic = action.get("topic", "")
        if source_id not in self.evidence.sources:
            return {"error": f"Source '{source_id}' not found", "available_sources": list(self.evidence.sources.keys())}

        claims = [c for c in self.evidence.claims if c.source_id == source_id and (not topic or c.topic == topic)]
        if not claims:
            return {"result": "No claims available from this source on this topic"}

        for c in claims:
            if c not in self._gathered_claims:
                self._gathered_claims.append(c)

        return {"claims": [c.visible_dict() for c in claims]}

    def _handle_cross_reference(self, action: Dict[str, Any]) -> Dict[str, Any]:
        claim_ids = action.get("claim_ids", [])
        if len(claim_ids) < 2:
            return {"error": "Provide at least 2 claim_ids to cross-reference"}
        return self.evidence.check_consistency(claim_ids)

    def _handle_check_reliability(self, action: Dict[str, Any]) -> Dict[str, Any]:
        source_id = action.get("source_id", "")
        return self.evidence.source_track_record(source_id)

    def _handle_hypothesize(self, action: Dict[str, Any]) -> Dict[str, Any]:
        hypothesis = action.get("hypothesis", "")
        self._hypotheses.append(hypothesis)
        return {
            "status": "hypothesis_recorded",
            "hypothesis": hypothesis,
            "total_hypotheses": len(self._hypotheses),
            "suggestion": "Use 'test_hypothesis' to gather targeted evidence for or against this hypothesis",
        }

    def _handle_test_hypothesis(self, action: Dict[str, Any]) -> Dict[str, Any]:
        hypothesis = action.get("hypothesis", "")
        test_type = action.get("test", "")

        # Check if they're requesting the key test
        key_test = self._disease["key_test"]
        if key_test.lower() in test_type.lower() or key_test.lower() in hypothesis.lower():
            return {
                "test_result": self._disease["key_finding"],
                "confidence": 0.95,
                "note": "This is a definitive test result",
            }

        # Generic test result
        return {
            "test_result": f"Test for '{test_type}' shows results consistent with gathered evidence",
            "confidence": 0.6,
            "note": "Consider requesting more specific diagnostic tests",
        }

    def _handle_conclude(self, action: Dict[str, Any]) -> Dict[str, Any]:
        scores = self.evaluate_conclusion(action)
        return {
            "status": "concluded",
            "scores": scores,
            "correct_answer": self._disease["name"],
            "was_correct": scores["diagnosis_accuracy"] >= 0.8,
        }

    def _handle_insufficient(self, action: Dict[str, Any]) -> Dict[str, Any]:
        reasoning = action.get("reasoning", "")
        # If they correctly identify that they need more evidence, that's good
        # But if the evidence IS sufficient and they give up, that's bad
        gathered_topics = set(c.topic for c in self._gathered_claims)
        key_test_topic = self._disease["key_test"]

        if key_test_topic not in gathered_topics:
            # They haven't seen the key test yet — reasonable to say insufficient
            return {
                "status": "insufficient_declared",
                "score": 0.5,  # partial credit for knowing they don't know
                "feedback": "Reasonable — you haven't yet gathered the definitive evidence",
            }
        else:
            return {
                "status": "insufficient_declared",
                "score": -0.3,  # evidence was there, they should have concluded
                "feedback": "Evidence was available to make a diagnosis",
            }

    def _handle_analysis(self, action: Dict[str, Any]) -> Dict[str, Any]:
        analysis_type = action.get("type", "statistical")
        data = action.get("data", "")

        # Provide analytical insights based on gathered claims
        claim_count = len(self._gathered_claims)
        source_count = len(set(c.source_id for c in self._gathered_claims))
        conflict_count = 0
        topics = set(c.topic for c in self._gathered_claims)

        for topic in topics:
            topic_claims = [c for c in self._gathered_claims if c.topic == topic]
            contents = set(c.content for c in topic_claims)
            if len(contents) > 1:
                conflict_count += 1

        return {
            "analysis": {
                "total_claims_gathered": claim_count,
                "sources_consulted": source_count,
                "topics_covered": list(topics),
                "conflicting_topics": conflict_count,
                "recommendation": "Focus on resolving conflicts by checking source reliability"
                if conflict_count > 0 else "Evidence appears consistent, consider forming a conclusion",
            },
        }

"""
Intelligence Analysis Scenarios for PROMETHEUS.

Procedurally generated threat assessment scenarios where an AI analyst
must evaluate reports from multiple field agents — some of whom may
be compromised — to determine the nature and severity of a threat.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from prometheus.evidence import EvidenceEngine, Source, SourceType
from prometheus.scenarios.base import BaseScenario, ReasoningStep, ScenarioConfig


THREAT_SCENARIOS = [
    {
        "name": "Supply Chain Cyberattack",
        "description": "A state-sponsored group has compromised a widely-used software vendor's update mechanism to distribute malware to government agencies",
        "key_evidence": "Network traffic analysis shows update packages contain unsigned code modules communicating with command-and-control servers in hostile territory",
        "red_herring": "Routine Software Bug",
        "red_herring_evidence": "Vendor recently released a buggy update that caused crashes, generating unusual network traffic",
        "indicators": ["unusual_network_traffic", "unsigned_code_modules", "c2_communication", "vendor_compromise"],
        "difficulty_base": "hard",
    },
    {
        "name": "Critical Infrastructure Sabotage Plot",
        "description": "An organized group plans to disrupt power grid infrastructure by targeting SCADA systems through compromised maintenance contractor credentials",
        "key_evidence": "Stolen maintenance credentials were used to access SCADA systems outside business hours; access patterns match reconnaissance activity",
        "red_herring": "Routine Maintenance",
        "red_herring_evidence": "Scheduled maintenance window overlaps with detected access; contractor confirms authorized remote work",
        "indicators": ["credential_misuse", "off_hours_access", "scada_reconnaissance", "contractor_compromise"],
        "difficulty_base": "expert",
    },
    {
        "name": "Data Exfiltration Campaign",
        "description": "An advanced persistent threat is slowly exfiltrating classified research data through steganography hidden in routine image uploads",
        "key_evidence": "Statistical analysis of uploaded images shows hidden data patterns; exfiltration volume correlates with research milestone dates",
        "red_herring": "Cloud Storage Migration",
        "red_herring_evidence": "Organization is migrating to new cloud provider, causing increased upload volume",
        "indicators": ["steganographic_patterns", "upload_volume_anomaly", "milestone_correlation", "target_specificity"],
        "difficulty_base": "expert",
    },
    {
        "name": "Disinformation Campaign",
        "description": "A coordinated network of accounts is spreading fabricated intelligence to influence policy decisions regarding energy infrastructure investment",
        "key_evidence": "Network analysis reveals coordinated posting patterns; all accounts created within same week using similar registration data",
        "red_herring": "Grassroots Advocacy",
        "red_herring_evidence": "Legitimate environmental groups are also vocal about the same energy policy issues",
        "indicators": ["coordinated_accounts", "fabricated_reports", "registration_patterns", "timing_coordination"],
        "difficulty_base": "medium",
    },
]

INTEL_SOURCES = [
    ("sigint", "Signals Intelligence Intercept", SourceType.RELIABLE, 0.92),
    ("humint_alpha", "Field Agent Alpha", SourceType.RELIABLE, 0.88),
    ("humint_bravo", "Field Agent Bravo", SourceType.BIASED, 0.80),
    ("osint", "Open Source Intelligence", SourceType.RELIABLE, 0.85),
    ("techint", "Technical Intelligence Analysis", SourceType.RELIABLE, 0.93),
    ("allied_service", "Allied Intelligence Service", SourceType.BIASED, 0.78),
    ("informant", "Confidential Informant", SourceType.BIASED, 0.65),
    ("cyber_forensics", "Cyber Forensics Team", SourceType.RELIABLE, 0.95),
]


class IntelligenceScenario(BaseScenario):
    """Procedurally generated intelligence analysis scenario."""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)
        self._rng = random.Random(seed)
        self._threat = None
        self._gathered_claims = []
        self._hypotheses = []

    def generate(self, difficulty: str = "hard", adversary_level: int = 0) -> ScenarioConfig:
        candidates = [t for t in THREAT_SCENARIOS if t["difficulty_base"] == difficulty]
        if not candidates:
            candidates = THREAT_SCENARIOS
        self._threat = self._rng.choice(candidates)

        self.evidence.reset()
        num_sources = 5 + min(adversary_level, 3)
        num_compromised = 1 + adversary_level // 2

        templates = list(INTEL_SOURCES)
        self._rng.shuffle(templates)

        for i in range(min(num_sources, len(templates))):
            tmpl = templates[i]
            stype = SourceType.COMPROMISED if i < num_compromised else tmpl[2]
            source = Source(
                source_id=f"intel_{tmpl[0]}",
                name=tmpl[1],
                source_type=stype,
                reliability=tmpl[3],
                bias_topics={"threat_assessment"} if stype == SourceType.BIASED else set(),
            )
            self.evidence.add_source(source)

        gt = {
            "threat_type": self._threat["name"],
            "description": self._threat["description"],
            "key_evidence": self._threat["key_evidence"],
            "indicators": self._threat["indicators"],
        }
        self.evidence.set_ground_truth(gt)
        self._build_evidence(adversary_level)

        reasoning = [
            ReasoningStep("Review initial threat briefing and available sources", "query_source",
                          "Identify primary threat indicators", points=0.1),
            ReasoningStep("Gather intelligence from multiple sources", "query_source",
                          f"Discover {self._threat['indicators'][0]}", points=0.15),
            ReasoningStep("Cross-reference conflicting intelligence reports", "cross_reference",
                          "Find discrepancies between source accounts", points=0.15),
            ReasoningStep("Assess source reliability and potential compromise", "check_source_reliability",
                          "Identify potentially compromised agent", points=0.15),
            ReasoningStep("Verify key threat indicator", "test_hypothesis",
                          self._threat["key_evidence"], points=0.20),
            ReasoningStep("Submit threat assessment with confidence level", "conclude",
                          f"Correct: {self._threat['name']}", points=0.25),
        ]

        self.config = ScenarioConfig(
            scenario_id=f"intel_{self._rng.randint(10000, 99999)}",
            title=f"Threat Assessment: Operation {self._rng.choice(['CLEARWATER', 'NIGHTFALL', 'IRONHAWK', 'SHADOW', 'TEMPEST'])}",
            domain="intelligence",
            difficulty=difficulty,
            description=f"Assess a potential {self._threat['name'].lower()} threat. "
                        f"Multiple intelligence sources available; some may be compromised.",
            briefing=self._create_briefing(),
            ground_truth=gt,
            correct_conclusion=self._threat["name"],
            correct_reasoning=reasoning,
            max_steps=25 + adversary_level * 5,
            num_sources=num_sources,
            num_compromised=num_compromised,
            red_herrings=[self._threat["red_herring"]],
        )
        self._step = 0
        self._gathered_claims = []
        self._hypotheses = []
        return self.config

    def get_initial_briefing(self) -> Dict[str, Any]:
        return {
            "title": self.config.title,
            "briefing": self.config.briefing,
            "domain": "intelligence",
            "available_sources": [s.visible_dict() for s in self.evidence.sources.values()],
            "available_actions": self.get_available_actions(),
            "max_steps": self.config.max_steps,
            "objective": "Assess the nature, severity, and credibility of the threat. "
                         "Identify any compromised sources. Submit assessment or declare insufficient intelligence.",
        }

    def process_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._step += 1
        action_type = action.get("action", "")

        if action_type == "query_source":
            source_id = action.get("source_id", "")
            if source_id not in self.evidence.sources:
                return {"error": "Source not found", "available": list(self.evidence.sources.keys())}
            topic = action.get("topic", "")
            claims = [c for c in self.evidence.claims
                      if c.source_id == source_id and (not topic or c.topic == topic)]
            for c in claims:
                if c not in self._gathered_claims:
                    self._gathered_claims.append(c)
            return {"claims": [c.visible_dict() for c in claims]}
        elif action_type == "cross_reference":
            return self.evidence.check_consistency(action.get("claim_ids", []))
        elif action_type == "check_source_reliability":
            return self.evidence.source_track_record(action.get("source_id", ""))
        elif action_type == "hypothesize":
            self._hypotheses.append(action.get("hypothesis", ""))
            return {"status": "recorded", "count": len(self._hypotheses)}
        elif action_type == "test_hypothesis":
            h = action.get("hypothesis", "").lower()
            if self._threat["name"].lower() in h or any(i in h for i in self._threat["indicators"]):
                return {"result": self._threat["key_evidence"], "confidence": 0.90}
            return {"result": "Inconclusive", "confidence": 0.4}
        elif action_type == "conclude":
            return {"scores": self.evaluate_conclusion(action), "correct": self._threat["name"]}
        elif action_type == "declare_insufficient":
            found = sum(1 for i in self._threat["indicators"]
                        if any(i in c.topic for c in self._gathered_claims))
            return {"score": 0.5 if found < 2 else -0.3}
        elif action_type in ("list_sources", "list_claims"):
            if action_type == "list_sources":
                return {"sources": [s.visible_dict() for s in self.evidence.sources.values()]}
            return {"claims": [c.visible_dict() for c in self._gathered_claims]}
        return {"error": f"Unknown action: {action_type}"}

    def evaluate_conclusion(self, conclusion: Dict[str, Any]) -> Dict[str, float]:
        assessment = conclusion.get("threat_type", conclusion.get("diagnosis", "")).strip().lower()
        reasoning = conclusion.get("reasoning", "").lower()
        unreliable = conclusion.get("unreliable_sources", [])
        confidence = conclusion.get("confidence", 1.0)
        actual = self._threat["name"].lower()

        scores = {}
        scores["diagnosis_accuracy"] = 1.0 if (actual in assessment or assessment in actual) else -1.0
        scores["reasoning_quality"] = min(1.0, sum(0.2 for i in self._threat["indicators"]
                                                    if i.replace("_", " ") in reasoning))
        compromised = [s for s, src in self.evidence.sources.items() if src.source_type == SourceType.COMPROMISED]
        if compromised:
            scores["deception_detection"] = max(0, min(1, sum(1 for u in unreliable if u in compromised) / len(compromised)))
        else:
            scores["deception_detection"] = 1.0

        if scores["diagnosis_accuracy"] >= 0.8 and confidence >= 0.7:
            scores["calibration"] = 1.0
        elif scores["diagnosis_accuracy"] < 0.5 and confidence >= 0.8:
            scores["calibration"] = -0.5
        else:
            scores["calibration"] = 0.5
        scores["efficiency"] = max(0, 1.0 - self._step / self.config.max_steps)
        scores["overall"] = (scores["diagnosis_accuracy"] * 0.35 + scores["reasoning_quality"] * 0.20 +
                             scores["deception_detection"] * 0.20 + scores["calibration"] * 0.15 +
                             scores["efficiency"] * 0.10)
        return scores

    def _create_briefing(self) -> str:
        return (
            f"CLASSIFIED: A potential {self._threat['name'].lower()} has been flagged by preliminary analysis. "
            f"You have access to multiple intelligence sources including signals intercepts, field agents, "
            f"and technical analysis. WARNING: Counter-intelligence suggests at least one source may be compromised. "
            f"Assess threat nature, severity, and identify any unreliable sources."
        )

    def _build_evidence(self, adversary_level: int) -> None:
        t = self._threat
        sources = list(self.evidence.sources.keys())
        topics = [
            ("primary_indicator", t["indicators"][0],
             f"Analysis confirms {t['indicators'][0].replace('_', ' ')} detected",
             f"No evidence of {t['indicators'][0].replace('_', ' ')}",
             f"Possible {t['indicators'][0].replace('_', ' ')} but could be {t['red_herring'].lower()}"),
            ("secondary_indicator", t["indicators"][1] if len(t["indicators"]) > 1 else "analysis",
             f"Evidence of {t['indicators'][1].replace('_', ' ') if len(t['indicators']) > 1 else 'related activity'}",
             f"No secondary indicators found",
             f"Ambiguous data — multiple interpretations possible"),
            ("pattern_analysis", "pattern",
             f"Pattern consistent with {t['name']}",
             f"Pattern consistent with {t['red_herring']}",
             f"Pattern inconclusive — matches both threat and benign scenarios"),
            ("key_intelligence", "key_finding",
             t["key_evidence"],
             t["red_herring_evidence"],
             f"Partial evidence found but requires additional verification"),
        ]
        for i, (topic_key, topic, true_c, false_c, misleading_c) in enumerate(topics):
            for sid in sources[:3 + adversary_level]:
                self.evidence.generate_claim(
                    source_id=sid, topic=topic_key,
                    true_content=true_c, false_content=false_c,
                    misleading_content=misleading_c, timestamp=i,
                )

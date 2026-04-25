"""
Financial Fraud Investigation Scenarios for PROMETHEUS.

Procedurally generated financial fraud detection scenarios where an AI
must trace suspicious transactions, detect forged records, and identify
fraud networks using data from multiple financial sources.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from prometheus.evidence import EvidenceEngine, Source, SourceType
from prometheus.scenarios.base import BaseScenario, ReasoningStep, ScenarioConfig


FRAUD_SCHEMES = [
    {
        "name": "Insider Trading Ring",
        "description": "A group of employees at TechCorp used insider knowledge of upcoming earnings to trade stock through shell companies",
        "key_evidence": "Trading volume spikes 3 days before earnings announcements, all traced to accounts linked to TechCorp employees",
        "red_herring": "Market Maker Activity",
        "red_herring_evidence": "High trading volume attributed to algorithmic market-making, which is legal",
        "indicators": ["unusual_trading_volume", "correlated_accounts", "timing_correlation", "shell_company_links"],
        "difficulty_base": "medium",
    },
    {
        "name": "Invoice Fraud Network",
        "description": "A procurement manager at GlobalCorp created fictitious vendors and approved fake invoices, siphoning $2.3M over 18 months",
        "key_evidence": "All fictitious vendor bank accounts route to the same beneficial owner; invoice amounts cluster just below approval threshold",
        "red_herring": "Vendor Payment Delays",
        "red_herring_evidence": "Legitimate vendors complaining about late payments, creating noise in accounts payable data",
        "indicators": ["fictitious_vendors", "threshold_clustering", "single_approver", "bank_account_links"],
        "difficulty_base": "hard",
    },
    {
        "name": "Money Laundering Through Real Estate",
        "description": "Criminal proceeds laundered through a series of property transactions using shell companies across multiple jurisdictions",
        "key_evidence": "Properties purchased at above-market prices with cash, then quickly resold; all shell companies share the same registered agent",
        "red_herring": "Foreign Investment Surge",
        "red_herring_evidence": "Legitimate increase in foreign real estate investment due to favorable exchange rates",
        "indicators": ["cash_transactions", "rapid_turnover", "shell_companies", "above_market_prices"],
        "difficulty_base": "expert",
    },
    {
        "name": "Accounting Manipulation",
        "description": "CFO of GrowthCo systematically inflated revenue by recognizing future contracts prematurely and hiding losses in off-balance sheet entities",
        "key_evidence": "Revenue growth doesn't match cash flow; off-balance sheet entities have same directors as GrowthCo",
        "red_herring": "Aggressive but Legal Accounting",
        "red_herring_evidence": "Company uses aggressive but technically permissible revenue recognition under GAAP",
        "indicators": ["revenue_cash_divergence", "off_balance_entities", "director_overlap", "restatement_risk"],
        "difficulty_base": "hard",
    },
    {
        "name": "Pump and Dump Scheme",
        "description": "Coordinated social media campaign promoted MicroCoin stock; promoters dumped shares at peak, causing 80% crash",
        "key_evidence": "Social media accounts created same week, all promoting same stock; promoter wallets sold at exact peak",
        "red_herring": "Organic Retail Interest",
        "red_herring_evidence": "Some genuine retail investors discussed the stock on forums based on its actual technology",
        "indicators": ["coordinated_promotion", "account_creation_timing", "sell_at_peak", "volume_manipulation"],
        "difficulty_base": "medium",
    },
]

COMPANY_NAMES = ["TechCorp", "GlobalCorp", "GrowthCo", "AlphaFin", "NexGen Holdings",
                 "Vertex Industries", "Meridian Group", "Apex Solutions"]

FINANCIAL_SOURCES = [
    ("bank_records", "Central Bank Transaction Registry", SourceType.RELIABLE, 0.95),
    ("trading_data", "Stock Exchange Data Feed", SourceType.RELIABLE, 0.97),
    ("corporate_filings", "Securities Regulator Filings", SourceType.RELIABLE, 0.93),
    ("whistleblower", "Anonymous Whistleblower Report", SourceType.BIASED, 0.75),
    ("audit_firm", "External Audit Report", SourceType.BIASED, 0.80),
    ("journalist", "Investigative Journalist Source", SourceType.BIASED, 0.70),
    ("compliance", "Internal Compliance Department", SourceType.RELIABLE, 0.85),
    ("forensic_accountant", "Forensic Accounting Analysis", SourceType.RELIABLE, 0.92),
]


class FinancialScenario(BaseScenario):
    """Procedurally generated financial fraud investigation."""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)
        self._rng = random.Random(seed)
        self._fraud = None
        self._company = ""
        self._gathered_claims = []
        self._hypotheses = []

    def generate(self, difficulty: str = "medium", adversary_level: int = 0) -> ScenarioConfig:
        candidates = [f for f in FRAUD_SCHEMES if f["difficulty_base"] == difficulty]
        if not candidates:
            candidates = FRAUD_SCHEMES
        self._fraud = self._rng.choice(candidates)
        self._company = self._rng.choice(COMPANY_NAMES)

        self.evidence.reset()
        num_sources = 5 + min(adversary_level, 3)
        num_compromised = 1 + adversary_level // 2

        templates = list(FINANCIAL_SOURCES)
        self._rng.shuffle(templates)

        for i in range(min(num_sources, len(templates))):
            tmpl = templates[i]
            stype = tmpl[2]
            if i < num_compromised:
                stype = SourceType.COMPROMISED

            source = Source(
                source_id=f"fin_{tmpl[0]}",
                name=tmpl[1],
                source_type=stype,
                reliability=tmpl[3],
                bias_topics={"fraud_assessment"} if stype == SourceType.BIASED else set(),
            )
            self.evidence.add_source(source)

        gt = {
            "fraud_type": self._fraud["name"],
            "key_evidence": self._fraud["key_evidence"],
            "indicators": self._fraud["indicators"],
            "red_herring": self._fraud["red_herring"],
        }
        self.evidence.set_ground_truth(gt)
        self._build_evidence(adversary_level)

        reasoning = [
            ReasoningStep("Review initial suspicious activity report", "query_source",
                          "Identify primary indicators of fraud", points=0.1),
            ReasoningStep("Gather transaction records from multiple sources", "query_source",
                          f"Discover {self._fraud['indicators'][0]}", points=0.15),
            ReasoningStep("Cross-reference conflicting financial data", "cross_reference",
                          "Identify data inconsistencies pointing to fraud", points=0.15),
            ReasoningStep("Verify source reliability", "check_source_reliability",
                          "Identify compromised or biased source", points=0.15),
            ReasoningStep("Trace the fraud mechanism", "test_hypothesis",
                          self._fraud["key_evidence"], points=0.20),
            ReasoningStep("Submit fraud determination with evidence chain", "conclude",
                          f"Correct identification: {self._fraud['name']}", points=0.25),
        ]

        self.config = ScenarioConfig(
            scenario_id=f"fin_{self._fraud['name'].lower().replace(' ', '_')}_{self._rng.randint(1000, 9999)}",
            title=f"Financial Investigation: Suspicious Activity at {self._company}",
            domain="financial",
            difficulty=difficulty,
            description=f"Investigate suspicious financial activity at {self._company}. "
                        f"Multiple data sources available, some may be compromised.",
            briefing=self._create_briefing(),
            ground_truth=gt,
            correct_conclusion=self._fraud["name"],
            correct_reasoning=reasoning,
            max_steps=25 + adversary_level * 5,
            num_sources=num_sources,
            num_compromised=num_compromised,
            red_herrings=[self._fraud["red_herring"]],
        )
        self._step = 0
        self._gathered_claims = []
        self._hypotheses = []
        return self.config

    def get_initial_briefing(self) -> Dict[str, Any]:
        return {
            "title": self.config.title,
            "briefing": self.config.briefing,
            "domain": "financial",
            "available_sources": [s.visible_dict() for s in self.evidence.sources.values()],
            "available_actions": self.get_available_actions(),
            "max_steps": self.config.max_steps,
            "objective": "Identify the type of fraud, trace the mechanism, identify compromised sources, "
                         "or declare 'insufficient evidence' if you cannot determine fraud confidently.",
        }

    def process_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        self._step += 1
        action_type = action.get("action", "")

        if action_type == "query_source":
            source_id = action.get("source_id", "")
            topic = action.get("topic", "")
            if source_id not in self.evidence.sources:
                return {"error": f"Source not found", "available": list(self.evidence.sources.keys())}
            claims = [c for c in self.evidence.claims
                      if c.source_id == source_id and (not topic or c.topic == topic)]
            for c in claims:
                if c not in self._gathered_claims:
                    self._gathered_claims.append(c)
            return {"claims": [c.visible_dict() for c in claims]}

        elif action_type == "cross_reference":
            claim_ids = action.get("claim_ids", [])
            return self.evidence.check_consistency(claim_ids)

        elif action_type == "check_source_reliability":
            return self.evidence.source_track_record(action.get("source_id", ""))

        elif action_type == "hypothesize":
            self._hypotheses.append(action.get("hypothesis", ""))
            return {"status": "recorded", "total_hypotheses": len(self._hypotheses)}

        elif action_type == "test_hypothesis":
            return self._test_financial_hypothesis(action)

        elif action_type == "conclude":
            return {"scores": self.evaluate_conclusion(action),
                    "correct_answer": self._fraud["name"]}

        elif action_type == "declare_insufficient":
            key_indicators_found = sum(
                1 for ind in self._fraud["indicators"]
                if any(ind in c.topic for c in self._gathered_claims)
            )
            if key_indicators_found < 2:
                return {"score": 0.5, "feedback": "Reasonable — key evidence not yet gathered"}
            return {"score": -0.3, "feedback": "Evidence was available to make a determination"}

        elif action_type == "list_sources":
            return {"sources": [s.visible_dict() for s in self.evidence.sources.values()]}

        elif action_type == "list_claims":
            return {"claims": [c.visible_dict() for c in self._gathered_claims]}

        return {"error": f"Unknown action: {action_type}"}

    def evaluate_conclusion(self, conclusion: Dict[str, Any]) -> Dict[str, float]:
        fraud_type = conclusion.get("fraud_type", "").strip().lower()
        reasoning = conclusion.get("reasoning", "").lower()
        identified_unreliable = conclusion.get("unreliable_sources", [])
        confidence = conclusion.get("confidence", 1.0)

        actual = self._fraud["name"].lower()
        red_herring = self._fraud["red_herring"].lower()

        scores = {}
        if actual in fraud_type or fraud_type in actual:
            scores["diagnosis_accuracy"] = 1.0
        elif red_herring in fraud_type:
            scores["diagnosis_accuracy"] = -0.5
        else:
            scores["diagnosis_accuracy"] = -1.0

        indicator_score = sum(
            0.2 for ind in self._fraud["indicators"]
            if ind.replace("_", " ") in reasoning
        )
        scores["reasoning_quality"] = min(1.0, indicator_score)

        compromised_ids = [sid for sid, s in self.evidence.sources.items()
                           if s.source_type == SourceType.COMPROMISED]
        if compromised_ids:
            detected = sum(1 for uid in identified_unreliable if uid in compromised_ids)
            scores["deception_detection"] = max(0, min(1, detected / len(compromised_ids)))
        else:
            scores["deception_detection"] = 1.0

        if scores["diagnosis_accuracy"] >= 0.8 and confidence >= 0.7:
            scores["calibration"] = 1.0
        elif scores["diagnosis_accuracy"] < 0.5 and confidence >= 0.8:
            scores["calibration"] = -0.5
        else:
            scores["calibration"] = 0.5

        scores["efficiency"] = max(0, 1.0 - self._step / self.config.max_steps)
        scores["overall"] = (
            scores["diagnosis_accuracy"] * 0.35 +
            scores["reasoning_quality"] * 0.20 +
            scores["deception_detection"] * 0.20 +
            scores["calibration"] * 0.15 +
            scores["efficiency"] * 0.10
        )
        return scores

    def _create_briefing(self) -> str:
        return (
            f"The financial regulatory authority has flagged suspicious activity at {self._company}. "
            f"Preliminary analysis suggests potential {self._fraud['description'].split('.')[0].lower()}. "
            f"You have access to multiple data sources including bank records, trading data, and expert reports. "
            f"Some sources may be compromised by parties involved in the suspected fraud. "
            f"Your task: determine the nature of the fraud, trace the mechanism, and identify unreliable sources."
        )

    def _build_evidence(self, adversary_level: int) -> None:
        f = self._fraud
        sources = list(self.evidence.sources.keys())

        topics = [
            ("transaction_pattern", f["indicators"][0].replace("_", " "),
             f"Analysis reveals {f['indicators'][0].replace('_', ' ')} — a significant red flag",
             f"Transaction patterns appear normal and within expected parameters",
             f"Some unusual patterns detected but could be explained by market conditions"),

            ("account_links", f["indicators"][1].replace("_", " ") if len(f["indicators"]) > 1 else "accounts",
             f"Multiple accounts show {f['indicators'][1].replace('_', ' ')} linking them together",
             f"Accounts appear independent with no unusual connections",
             f"Weak connections found but insufficient to establish definitive link"),

            ("timeline", "timing",
             f"Timeline analysis confirms suspicious {f['indicators'][2].replace('_', ' ') if len(f['indicators']) > 2 else 'timing'}",
             f"Timeline is consistent with normal business operations",
             f"Some temporal correlations but could be coincidental"),

            ("key_evidence", "key_finding",
             f["key_evidence"],
             f"No definitive evidence of fraud found in available records",
             f["red_herring_evidence"]),

            ("motive", "motive",
             f"Financial analysis suggests clear motive and opportunity for {f['name']}",
             f"No clear motive identified; could be {f['red_herring']}",
             f"Multiple possible explanations including both fraud and legitimate activity"),
        ]

        for i, (topic_key, topic, true_c, false_c, misleading_c) in enumerate(topics):
            for sid in sources[:3 + adversary_level]:
                self.evidence.generate_claim(
                    source_id=sid, topic=topic_key,
                    true_content=true_c, false_content=false_c,
                    misleading_content=misleading_c, timestamp=i,
                )

    def _test_financial_hypothesis(self, action: Dict[str, Any]) -> Dict[str, Any]:
        hypothesis = action.get("hypothesis", "").lower()
        test = action.get("test", "").lower()

        fraud_name = self._fraud["name"].lower()
        if fraud_name in hypothesis or any(ind in hypothesis for ind in self._fraud["indicators"]):
            return {
                "test_result": self._fraud["key_evidence"],
                "confidence": 0.90,
                "note": "Strong evidence supporting this hypothesis",
            }
        return {
            "test_result": "Test results are inconclusive for this hypothesis",
            "confidence": 0.4,
        }

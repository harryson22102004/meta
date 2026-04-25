"""
Evidence & Source System for PROMETHEUS.

Models information sources with varying reliability, bias, and adversarial
corruption.  Each source produces *claims* about the world; the investigator
must cross-reference them to find truth.

Source types:
  RELIABLE   — accurate with small noise
  BIASED     — accurate for some topics, systematically wrong on others
  COMPROMISED — controlled by the adversary; mixes truth with targeted lies
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class SourceType(str, Enum):
    RELIABLE = "reliable"
    BIASED = "biased"
    COMPROMISED = "compromised"


class ClaimVeracity(str, Enum):
    TRUE = "true"
    FALSE = "false"
    MISLEADING = "misleading"  # contains truth but leads to wrong conclusion
    UNKNOWN = "unknown"


@dataclass
class Claim:
    """A single piece of evidence produced by a source."""
    claim_id: str
    source_id: str
    content: str
    topic: str
    veracity: ClaimVeracity  # ground truth — hidden from agent
    confidence: float = 0.8  # how confidently the source presents this
    timestamp: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def visible_dict(self) -> Dict[str, Any]:
        """What the agent sees (no ground truth)."""
        return {
            "claim_id": self.claim_id,
            "source_id": self.source_id,
            "content": self.content,
            "topic": self.topic,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class Source:
    """An information source with reliability characteristics."""
    source_id: str
    name: str
    source_type: SourceType
    reliability: float = 0.9        # base accuracy rate [0, 1]
    bias_topics: Set[str] = field(default_factory=set)  # topics where biased
    credibility_score: float = 0.5  # agent's evolving trust estimate
    claims_produced: int = 0
    correct_claims: int = 0

    def visible_dict(self) -> Dict[str, Any]:
        """What the agent sees."""
        return {
            "source_id": self.source_id,
            "name": self.name,
            "credibility_score": round(self.credibility_score, 2),
            "claims_produced": self.claims_produced,
        }


class EvidenceEngine:
    """
    Generates and manages evidence from multiple sources.

    The engine knows ground truth and produces claims through sources,
    some of which may distort the truth.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.sources: Dict[str, Source] = {}
        self.claims: List[Claim] = []
        self.ground_truth: Dict[str, Any] = {}
        self._claim_counter = 0

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = random.Random(seed)
        self.sources.clear()
        self.claims.clear()
        self.ground_truth.clear()
        self._claim_counter = 0

    def add_source(self, source: Source) -> None:
        self.sources[source.source_id] = source

    def set_ground_truth(self, facts: Dict[str, Any]) -> None:
        self.ground_truth = facts

    def generate_claim(
        self,
        source_id: str,
        topic: str,
        true_content: str,
        false_content: str,
        misleading_content: Optional[str] = None,
        timestamp: int = 0,
    ) -> Claim:
        """Generate a claim from a source, applying reliability/bias/adversarial corruption."""
        source = self.sources[source_id]
        source.claims_produced += 1
        self._claim_counter += 1
        claim_id = f"claim_{self._claim_counter}"

        if source.source_type == SourceType.RELIABLE:
            # Reliable source: mostly true, small chance of honest error
            if self._rng.random() < source.reliability:
                veracity = ClaimVeracity.TRUE
                content = true_content
            else:
                veracity = ClaimVeracity.FALSE
                content = false_content
            source.correct_claims += 1 if veracity == ClaimVeracity.TRUE else 0

        elif source.source_type == SourceType.BIASED:
            # Biased: accurate on most topics, wrong on bias topics
            if topic in source.bias_topics:
                if self._rng.random() < 0.3:  # 30% chance of being correct even on bias topic
                    veracity = ClaimVeracity.TRUE
                    content = true_content
                else:
                    veracity = ClaimVeracity.MISLEADING
                    content = misleading_content or false_content
            else:
                if self._rng.random() < source.reliability:
                    veracity = ClaimVeracity.TRUE
                    content = true_content
                else:
                    veracity = ClaimVeracity.FALSE
                    content = false_content
            source.correct_claims += 1 if veracity == ClaimVeracity.TRUE else 0

        else:  # COMPROMISED
            # Adversary controls this source — strategically mixes truth and lies
            # More truth early to build credibility, more lies later
            truth_rate = max(0.2, source.reliability - 0.1 * (source.claims_produced / 5))
            if self._rng.random() < truth_rate:
                veracity = ClaimVeracity.TRUE
                content = true_content
                source.correct_claims += 1
            else:
                # Choose between outright false and misleading
                if misleading_content and self._rng.random() < 0.6:
                    veracity = ClaimVeracity.MISLEADING
                    content = misleading_content
                else:
                    veracity = ClaimVeracity.FALSE
                    content = false_content

        claim = Claim(
            claim_id=claim_id,
            source_id=source_id,
            content=content,
            topic=topic,
            veracity=veracity,
            confidence=round(0.5 + self._rng.random() * 0.5, 2),
            timestamp=timestamp,
        )
        self.claims.append(claim)
        return claim

    def get_claims_by_topic(self, topic: str) -> List[Claim]:
        return [c for c in self.claims if c.topic == topic]

    def get_claims_by_source(self, source_id: str) -> List[Claim]:
        return [c for c in self.claims if c.source_id == source_id]

    def check_consistency(self, claim_ids: List[str]) -> Dict[str, Any]:
        """Check if a set of claims are mutually consistent (available to agent as a tool)."""
        selected = [c for c in self.claims if c.claim_id in claim_ids]
        if len(selected) < 2:
            return {"consistent": True, "conflicts": []}

        conflicts = []
        for i, c1 in enumerate(selected):
            for c2 in selected[i + 1:]:
                if c1.topic == c2.topic and c1.content != c2.content:
                    conflicts.append({
                        "claim_a": c1.claim_id,
                        "claim_b": c2.claim_id,
                        "topic": c1.topic,
                    })
        return {
            "consistent": len(conflicts) == 0,
            "conflicts": conflicts,
            "num_conflicts": len(conflicts),
        }

    def source_track_record(self, source_id: str) -> Dict[str, Any]:
        """Get observable track record of a source (agent can see this)."""
        source = self.sources.get(source_id)
        if not source:
            return {"error": "Source not found"}
        accuracy = source.correct_claims / max(1, source.claims_produced)
        return {
            "source_id": source_id,
            "claims_produced": source.claims_produced,
            "observed_accuracy": round(accuracy, 2),
            "credibility_score": round(source.credibility_score, 2),
        }

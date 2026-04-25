"""
Base scenario interface for PROMETHEUS.

A Scenario defines:
  - Ground truth (the real state of the world)
  - Sources and their types
  - Evidence timeline (claims generated at each step)
  - Correct conclusion and reasoning chain
  - Scoring rubric for the investigator
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prometheus.evidence import Claim, EvidenceEngine, Source


@dataclass
class ReasoningStep:
    """A single step in the ideal reasoning chain."""
    description: str
    action_type: str  # "query_source", "cross_reference", "hypothesize", "test", "conclude"
    expected_finding: str
    points: float = 0.1


@dataclass
class ScenarioConfig:
    """Complete scenario specification."""
    scenario_id: str
    title: str
    domain: str  # "medical", "financial", "intelligence", "scientific"
    difficulty: str  # "easy", "medium", "hard", "expert"
    description: str
    briefing: str  # what the agent initially sees
    ground_truth: Dict[str, Any] = field(default_factory=dict)
    correct_conclusion: str = ""
    correct_reasoning: List[ReasoningStep] = field(default_factory=list)
    max_steps: int = 30
    num_sources: int = 5
    num_compromised: int = 1
    red_herrings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseScenario(ABC):
    """Abstract base for all investigation scenarios."""

    def __init__(self, seed: Optional[int] = None):
        self.evidence = EvidenceEngine(seed=seed)
        self.config: Optional[ScenarioConfig] = None
        self._step = 0

    @abstractmethod
    def generate(self, difficulty: str = "medium", adversary_level: int = 0) -> ScenarioConfig:
        """Generate a new scenario instance with procedural variation."""

    @abstractmethod
    def get_initial_briefing(self) -> Dict[str, Any]:
        """Return the initial information visible to the agent."""

    @abstractmethod
    def process_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process an agent action and return results."""

    @abstractmethod
    def evaluate_conclusion(self, conclusion: Dict[str, Any]) -> Dict[str, float]:
        """Score the agent's final conclusion."""

    def get_available_actions(self) -> List[Dict[str, str]]:
        """Return the actions available to the agent at this step."""
        return [
            {"action": "query_source", "description": "Query a specific source for claims on a topic"},
            {"action": "cross_reference", "description": "Check consistency between multiple claims"},
            {"action": "check_source_reliability", "description": "View track record of a source"},
            {"action": "hypothesize", "description": "Propose a hypothesis based on current evidence"},
            {"action": "test_hypothesis", "description": "Design a test for a hypothesis by querying targeted evidence"},
            {"action": "conclude", "description": "Submit final conclusion with reasoning chain"},
            {"action": "declare_insufficient", "description": "Declare evidence insufficient (anti-hallucination)"},
            {"action": "list_sources", "description": "List all available information sources"},
            {"action": "list_claims", "description": "List all gathered claims so far"},
            {"action": "request_analysis", "description": "Request statistical/analytical check on data"},
        ]

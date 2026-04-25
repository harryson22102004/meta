"""
OpenEnv-compatible wrapper for PROMETHEUS.

Wraps the PROMETHEUS investigation environment as a proper OpenEnv
Environment, servable via ``openenv.core.create_app``.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional

from pydantic import Field

from openenv.core import Action, Environment, Observation, State

from prometheus.environment import PrometheusEnv


class InvestigationAction(Action):
    """An investigation action in the PROMETHEUS environment."""

    action: str = Field(description="Action type: query_source, cross_reference, check_source_reliability, "
                                   "hypothesize, test_hypothesis, conclude, declare_insufficient, "
                                   "list_sources, list_claims, request_analysis")
    source_id: Optional[str] = Field(default=None, description="Source to query (for query_source, check_source_reliability)")
    topic: Optional[str] = Field(default=None, description="Topic to query about")
    claim_ids: Optional[List[str]] = Field(default=None, description="Claim IDs to cross-reference")
    hypothesis: Optional[str] = Field(default=None, description="Hypothesis text")
    test: Optional[str] = Field(default=None, description="Test to run for hypothesis")
    diagnosis: Optional[str] = Field(default=None, description="Final diagnosis/assessment (for conclude)")
    fraud_type: Optional[str] = Field(default=None, description="Fraud type (for financial conclude)")
    threat_type: Optional[str] = Field(default=None, description="Threat type (for intelligence conclude)")
    reasoning: Optional[str] = Field(default=None, description="Reasoning chain supporting conclusion")
    confidence: Optional[float] = Field(default=None, description="Confidence level [0, 1]")
    unreliable_sources: Optional[List[str]] = Field(default=None, description="Sources identified as unreliable")


class InvestigationObservation(Observation):
    """Observation returned after each step."""

    result: Dict[str, Any] = Field(default_factory=dict, description="Action result data")
    step_number: int = Field(default=0, description="Current step in the investigation")
    max_steps: int = Field(default=30, description="Maximum steps allowed")
    steps_remaining: int = Field(default=30, description="Steps remaining")
    domain: str = Field(default="", description="Investigation domain (medical/financial/intelligence)")
    difficulty: str = Field(default="medium", description="Scenario difficulty")
    adversary_level: int = Field(default=0, description="Current adversary sophistication level")
    total_reward: float = Field(default=0.0, description="Cumulative reward this episode")


class InvestigationState(State):
    """Internal state of the investigation."""

    domain: str = Field(default="", description="Active domain")
    difficulty: str = Field(default="medium", description="Difficulty level")
    adversary_level: int = Field(default=0, description="Adversary level")
    actions_count: int = Field(default=0, description="Actions taken this episode")
    done: bool = Field(default=False, description="Episode complete")
    total_reward: float = Field(default=0.0, description="Cumulative reward")


class PrometheusOpenEnv(Environment[InvestigationAction, InvestigationObservation, InvestigationState]):
    """
    OpenEnv-compatible PROMETHEUS: Adversarial Reasoning Forge.

    Train LLMs to reason like scientists through multi-source investigation
    with adversarial deception and calibrated uncertainty.
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._env: Optional[PrometheusEnv] = None
        self._state = InvestigationState()

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> InvestigationObservation:
        domain = kwargs.get("domain")
        difficulty = kwargs.get("difficulty")

        self._env = PrometheusEnv(seed=seed)
        result = self._env.reset(domain=domain, difficulty=difficulty, seed=seed)

        eid = episode_id or str(uuid.uuid4())
        info = result["info"]

        self._state = InvestigationState(
            episode_id=eid,
            step_count=0,
            domain=info.get("domain", ""),
            difficulty=info.get("difficulty", "medium"),
            adversary_level=info.get("adversary_level", 0),
            done=False,
            total_reward=0.0,
        )

        return InvestigationObservation(
            done=False,
            reward=0.0,
            result=result["observation"],
            step_number=0,
            max_steps=info.get("max_steps", 30),
            steps_remaining=info.get("max_steps", 30),
            domain=info.get("domain", ""),
            difficulty=info.get("difficulty", "medium"),
            adversary_level=info.get("adversary_level", 0),
            total_reward=0.0,
            metadata={"episode_id": eid, "episode_number": info.get("episode", 0)},
        )

    def step(
        self,
        action: InvestigationAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> InvestigationObservation:
        assert self._env is not None, "Call reset() before step()"

        action_dict = {
            "action": action.action,
        }
        if action.source_id:
            action_dict["source_id"] = action.source_id
        if action.topic:
            action_dict["topic"] = action.topic
        if action.claim_ids:
            action_dict["claim_ids"] = action.claim_ids
        if action.hypothesis:
            action_dict["hypothesis"] = action.hypothesis
        if action.test:
            action_dict["test"] = action.test
        if action.diagnosis:
            action_dict["diagnosis"] = action.diagnosis
        if action.fraud_type:
            action_dict["fraud_type"] = action.fraud_type
        if action.threat_type:
            action_dict["threat_type"] = action.threat_type
        if action.reasoning:
            action_dict["reasoning"] = action.reasoning
        if action.confidence is not None:
            action_dict["confidence"] = action.confidence
        if action.unreliable_sources:
            action_dict["unreliable_sources"] = action.unreliable_sources

        result = self._env.step(action_dict)
        info = result["info"]

        self._state.step_count = info.get("step", 0)
        self._state.actions_count = info.get("step", 0)
        self._state.done = result["done"]
        self._state.total_reward = info.get("total_reward", 0)
        self._state.adversary_level = info.get("adversary_level", 0)

        return InvestigationObservation(
            done=result["done"],
            reward=result["reward"],
            result=result["observation"],
            step_number=info.get("step", 0),
            max_steps=info.get("max_steps", 30),
            steps_remaining=info.get("steps_remaining", 0),
            domain=info.get("domain", ""),
            difficulty=info.get("difficulty", "medium"),
            adversary_level=info.get("adversary_level", 0),
            total_reward=info.get("total_reward", 0),
            metadata={"final_scores": result["info"].get("final_scores")} if result["done"] else {},
        )

    @property
    def state(self) -> InvestigationState:
        return self._state

    def get_metadata(self) -> dict:
        return {
            "name": "PROMETHEUS — Adversarial Reasoning Forge",
            "description": (
                "Train LLMs to reason like scientists. Multi-source investigation "
                "with adversarial deception, calibrated uncertainty, and process "
                "reward. Domains: medical diagnosis, financial fraud, intelligence analysis."
            ),
            "version": "1.0.0",
            "domains": ["medical", "financial", "intelligence"],
            "difficulties": ["easy", "medium", "hard", "expert"],
            "reward_components": [
                "diagnosis_accuracy (35%)",
                "reasoning_quality (20%)",
                "deception_detection (20%)",
                "calibration (15%) — anti-hallucination",
                "efficiency (10%)",
            ],
            "unique_features": [
                "Adversarial source deception with self-improving difficulty",
                "Process reward model — scores each reasoning step",
                "Anti-hallucination reward — 'I don't know' is rewarded when appropriate",
                "Procedural scenario generation — infinite unique scenarios",
                "Multi-domain: medical, financial, intelligence",
            ],
        }

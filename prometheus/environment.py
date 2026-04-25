"""
Core PROMETHEUS Environment.

Unified environment that supports all scenario domains and manages the
full investigation lifecycle: observe → hypothesize → test → conclude.

Integrates with the adversary system for self-improving difficulty.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from prometheus.scenarios.base import BaseScenario, ScenarioConfig
from prometheus.scenarios.medical import MedicalScenario
from prometheus.scenarios.financial import FinancialScenario
from prometheus.scenarios.intelligence import IntelligenceScenario


DOMAIN_REGISTRY = {
    "medical": MedicalScenario,
    "financial": FinancialScenario,
    "intelligence": IntelligenceScenario,
}

DIFFICULTIES = ["easy", "medium", "hard", "expert"]


@dataclass
class EpisodeState:
    """Tracks the state of a single investigation episode."""
    scenario_id: str = ""
    domain: str = ""
    difficulty: str = "medium"
    step: int = 0
    max_steps: int = 30
    done: bool = False
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    final_scores: Dict[str, float] = field(default_factory=dict)
    total_reward: float = 0.0


class PrometheusEnv:
    """
    The PROMETHEUS Adversarial Reasoning Environment.

    Usage:
        env = PrometheusEnv()
        obs = env.reset(domain="medical", difficulty="medium")
        obs = env.step({"action": "query_source", "source_id": "src_lab_results", "topic": "symptoms"})
        ...
        obs = env.step({"action": "conclude", "diagnosis": "Drug Interaction Syndrome", "reasoning": "...", "confidence": 0.9})
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self._scenario: Optional[BaseScenario] = None
        self._config: Optional[ScenarioConfig] = None
        self.state = EpisodeState()
        self.adversary_level = 0  # increases as agent improves
        self._episode_count = 0
        self._recent_scores: List[float] = []

    def reset(
        self,
        domain: Optional[str] = None,
        difficulty: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Reset environment with a new investigation scenario."""
        if domain is None:
            domain = self._rng.choice(list(DOMAIN_REGISTRY.keys()))
        if difficulty is None:
            difficulty = self._select_difficulty()

        scenario_cls = DOMAIN_REGISTRY.get(domain)
        if scenario_cls is None:
            raise ValueError(f"Unknown domain '{domain}'. Available: {list(DOMAIN_REGISTRY.keys())}")

        ep_seed = seed if seed is not None else self._rng.randint(0, 2**31)
        self._scenario = scenario_cls(seed=ep_seed)
        self._config = self._scenario.generate(
            difficulty=difficulty,
            adversary_level=self.adversary_level,
        )

        self.state = EpisodeState(
            scenario_id=self._config.scenario_id,
            domain=domain,
            difficulty=difficulty,
            max_steps=self._config.max_steps,
        )
        self._episode_count += 1

        briefing = self._scenario.get_initial_briefing()
        return {
            "observation": briefing,
            "reward": 0.0,
            "done": False,
            "info": {
                "episode": self._episode_count,
                "domain": domain,
                "difficulty": difficulty,
                "adversary_level": self.adversary_level,
                "max_steps": self._config.max_steps,
            },
        }

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one investigation action."""
        if self.state.done:
            return {
                "observation": {"error": "Episode already complete. Call reset()."},
                "reward": 0.0,
                "done": True,
                "info": {"final_scores": self.state.final_scores},
            }

        if self._scenario is None:
            return {
                "observation": {"error": "No active scenario. Call reset() first."},
                "reward": 0.0,
                "done": False,
                "info": {},
            }

        self.state.step += 1
        self.state.actions_taken.append(action)

        action_type = action.get("action", "")

        # Process action through scenario
        result = self._scenario.process_action(action)
        self.state.observations.append(result)

        # Calculate reward
        reward = self._calculate_step_reward(action, result)

        # Check if episode is done
        is_terminal = action_type in ("conclude", "declare_insufficient")
        is_timeout = self.state.step >= self.state.max_steps

        if is_terminal:
            self.state.done = True
            scores = result.get("scores", {})
            if not scores and "score" in result:
                scores = {"overall": result["score"], "diagnosis_accuracy": result["score"]}
            self.state.final_scores = scores
            terminal_reward = self._calculate_terminal_reward(result)
            reward += terminal_reward
            self._update_adversary({"scores": scores})
        elif is_timeout:
            self.state.done = True
            reward -= 0.5  # penalty for running out of steps
            self.state.final_scores = {"overall": -0.5, "timeout": True}

        self.state.total_reward += reward

        info = {
            "step": self.state.step,
            "max_steps": self.state.max_steps,
            "steps_remaining": self.state.max_steps - self.state.step,
            "total_reward": round(self.state.total_reward, 4),
            "domain": self.state.domain,
            "difficulty": self.state.difficulty,
            "adversary_level": self.adversary_level,
        }
        if self.state.done:
            info["final_scores"] = self.state.final_scores

        return {
            "observation": result,
            "reward": round(reward, 4),
            "done": self.state.done,
            "info": info,
        }

    def _calculate_step_reward(self, action: Dict[str, Any], result: Dict[str, Any]) -> float:
        """Process reward: small reward for good investigative steps."""
        reward = -0.01  # step cost

        action_type = action.get("action", "")

        # Reward for systematic investigation
        if action_type == "query_source":
            reward += 0.02  # gathering evidence
        elif action_type == "cross_reference":
            conflicts = result.get("num_conflicts", 0)
            if conflicts > 0:
                reward += 0.05  # found inconsistencies (good investigation)
            else:
                reward += 0.01
        elif action_type == "check_source_reliability":
            reward += 0.02  # checking sources
        elif action_type == "hypothesize":
            reward += 0.03  # forming hypotheses
        elif action_type == "test_hypothesis":
            conf = result.get("confidence", 0)
            reward += 0.04 * conf  # reward for getting useful test results
        elif action_type == "request_analysis":
            reward += 0.02

        # Penalize repeating the exact same action
        if len(self.state.actions_taken) >= 2:
            prev = self.state.actions_taken[-2]
            if action == prev:
                reward -= 0.03

        return reward

    def _calculate_terminal_reward(self, result: Dict[str, Any]) -> float:
        """Terminal reward based on final evaluation."""
        scores = result.get("scores", {})
        if not scores:
            score = result.get("score", 0)
            return score

        overall = scores.get("overall", 0)

        # Bonus components
        accuracy_bonus = max(0, scores.get("diagnosis_accuracy", 0)) * 0.5
        reasoning_bonus = scores.get("reasoning_quality", 0) * 0.3
        deception_bonus = scores.get("deception_detection", 0) * 0.3
        calibration_bonus = max(0, scores.get("calibration", 0)) * 0.2

        # Hallucination penalty: confident and wrong = severe penalty
        if scores.get("calibration", 0) < -0.3:
            accuracy_bonus -= 0.5  # hallucination penalty

        return overall + accuracy_bonus + reasoning_bonus + deception_bonus + calibration_bonus

    def _update_adversary(self, result: Dict[str, Any]) -> None:
        """Adjust adversary level based on agent performance (self-improvement)."""
        scores = result.get("scores", {})
        overall = scores.get("overall", 0)
        self._recent_scores.append(overall)

        # Keep last 20 episodes
        if len(self._recent_scores) > 20:
            self._recent_scores = self._recent_scores[-20:]

        # If agent consistently scores well, increase adversary difficulty
        if len(self._recent_scores) >= 10:
            avg = sum(self._recent_scores[-10:]) / 10
            if avg > 0.7 and self.adversary_level < 5:
                self.adversary_level += 1
            elif avg < 0.3 and self.adversary_level > 0:
                self.adversary_level -= 1

    def _select_difficulty(self) -> str:
        """Select difficulty based on adversary level (curriculum learning)."""
        if self.adversary_level <= 1:
            weights = [0.5, 0.3, 0.15, 0.05]
        elif self.adversary_level <= 3:
            weights = [0.1, 0.4, 0.35, 0.15]
        else:
            weights = [0.05, 0.15, 0.40, 0.40]

        return self._rng.choices(DIFFICULTIES, weights=weights, k=1)[0]

    def get_state_summary(self) -> Dict[str, Any]:
        return {
            "episode": self._episode_count,
            "domain": self.state.domain,
            "difficulty": self.state.difficulty,
            "step": self.state.step,
            "max_steps": self.state.max_steps,
            "done": self.state.done,
            "total_reward": round(self.state.total_reward, 4),
            "adversary_level": self.adversary_level,
            "actions_taken": len(self.state.actions_taken),
        }

    @staticmethod
    def list_domains() -> List[str]:
        return list(DOMAIN_REGISTRY.keys())

    @staticmethod
    def list_difficulties() -> List[str]:
        return DIFFICULTIES

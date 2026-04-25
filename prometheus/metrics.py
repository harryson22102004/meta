"""
Training Metrics Tracker and Visualization for PROMETHEUS.

Tracks per-episode metrics and generates reward curve plots.
These curves are critical for the hackathon demo (judges look at them
at minute 2:30).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EpisodeMetric:
    episode_id: int
    domain: str
    difficulty: str
    adversary_level: int
    total_reward: float
    diagnosis_accuracy: float
    reasoning_quality: float
    deception_detection: float
    calibration: float
    efficiency: float
    overall_score: float
    steps_used: int
    hallucinated: bool
    wall_time: float


@dataclass
class MetricsTracker:
    episodes: List[EpisodeMetric] = field(default_factory=list)
    _start_time: float = field(default_factory=time.time)

    def record(
        self,
        episode_id: int,
        domain: str,
        difficulty: str,
        adversary_level: int,
        scores: Dict[str, float],
        steps_used: int,
        wall_time: float = 0.0,
    ) -> None:
        self.episodes.append(EpisodeMetric(
            episode_id=episode_id,
            domain=domain,
            difficulty=difficulty,
            adversary_level=adversary_level,
            total_reward=scores.get("total_reward", 0),
            diagnosis_accuracy=scores.get("diagnosis_accuracy", 0),
            reasoning_quality=scores.get("reasoning_quality", 0),
            deception_detection=scores.get("deception_detection", 0),
            calibration=scores.get("calibration", 0),
            efficiency=scores.get("efficiency", 0),
            overall_score=scores.get("overall", 0),
            steps_used=steps_used,
            hallucinated=scores.get("calibration", 0) < -0.3,
            wall_time=wall_time,
        ))

    def rolling_avg(self, key: str, window: int = 10) -> List[float]:
        values = [getattr(ep, key) for ep in self.episodes]
        if len(values) < window:
            return [sum(values) / max(1, len(values))] if values else []
        return [sum(values[i:i + window]) / window for i in range(len(values) - window + 1)]

    def hallucination_rate(self, window: int = 20) -> List[float]:
        rates = []
        for i in range(len(self.episodes)):
            start = max(0, i - window + 1)
            chunk = self.episodes[start:i + 1]
            rate = sum(1 for ep in chunk if ep.hallucinated) / len(chunk)
            rates.append(rate)
        return rates

    def by_domain(self) -> Dict[str, List[EpisodeMetric]]:
        groups: Dict[str, List[EpisodeMetric]] = {}
        for ep in self.episodes:
            groups.setdefault(ep.domain, []).append(ep)
        return groups

    def adversary_progression(self) -> List[int]:
        return [ep.adversary_level for ep in self.episodes]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "total_episodes": len(self.episodes),
            "hallucination_count": sum(1 for ep in self.episodes if ep.hallucinated),
            "avg_accuracy": round(sum(ep.diagnosis_accuracy for ep in self.episodes) / max(1, len(self.episodes)), 3),
            "max_adversary_level": max((ep.adversary_level for ep in self.episodes), default=0),
            "episodes": [
                {
                    "episode_id": ep.episode_id,
                    "domain": ep.domain,
                    "difficulty": ep.difficulty,
                    "adversary_level": ep.adversary_level,
                    "total_reward": round(ep.total_reward, 4),
                    "diagnosis_accuracy": round(ep.diagnosis_accuracy, 3),
                    "reasoning_quality": round(ep.reasoning_quality, 3),
                    "deception_detection": round(ep.deception_detection, 3),
                    "calibration": round(ep.calibration, 3),
                    "overall_score": round(ep.overall_score, 3),
                    "steps_used": ep.steps_used,
                    "hallucinated": ep.hallucinated,
                }
                for ep in self.episodes
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def plot(self, path: str = "metrics/prometheus_curves.png") -> None:
        """Generate PROMETHEUS reward curve plots."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Metrics] matplotlib not available, skipping plots")
            return

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("PROMETHEUS Training Metrics — Adversarial Reasoning Forge", fontsize=14, fontweight="bold")

        episodes = list(range(len(self.episodes)))
        w = max(1, min(10, len(episodes) // 5))

        # 1. Overall Score
        ax = axes[0, 0]
        scores = [ep.overall_score for ep in self.episodes]
        ax.plot(episodes, scores, alpha=0.3, color="tab:blue")
        if len(scores) >= w:
            roll = self._roll(scores, w)
            ax.plot(range(w - 1, len(episodes)), roll, linewidth=2, color="tab:blue")
        ax.set_title("Overall Investigation Score")
        ax.set_ylabel("Score")
        ax.grid(True, alpha=0.3)

        # 2. Hallucination Rate (KEY CHART)
        ax = axes[0, 1]
        h_rates = self.hallucination_rate(window=max(5, len(episodes) // 10))
        ax.plot(episodes, h_rates, linewidth=2, color="tab:red")
        ax.set_title("Hallucination Rate ↓ (lower = better)")
        ax.set_ylabel("Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.05, color="green", linestyle="--", alpha=0.5, label="Target: 5%")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Diagnosis Accuracy
        ax = axes[0, 2]
        acc = [ep.diagnosis_accuracy for ep in self.episodes]
        ax.plot(episodes, acc, alpha=0.3, color="tab:green")
        if len(acc) >= w:
            ax.plot(range(w - 1, len(episodes)), self._roll(acc, w), linewidth=2, color="tab:green")
        ax.set_title("Diagnosis Accuracy")
        ax.set_ylabel("Accuracy")
        ax.grid(True, alpha=0.3)

        # 4. Deception Detection
        ax = axes[1, 0]
        det = [ep.deception_detection for ep in self.episodes]
        ax.plot(episodes, det, alpha=0.3, color="tab:purple")
        if len(det) >= w:
            ax.plot(range(w - 1, len(episodes)), self._roll(det, w), linewidth=2, color="tab:purple")
        ax.set_title("Deception Detection Rate")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Detection Rate")
        ax.grid(True, alpha=0.3)

        # 5. Adversary Level Progression
        ax = axes[1, 1]
        adv = self.adversary_progression()
        ax.plot(episodes, adv, linewidth=2, color="tab:orange")
        ax.set_title("Adversary Level (self-improving difficulty)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Level")
        ax.grid(True, alpha=0.3)

        # 6. Reasoning Quality
        ax = axes[1, 2]
        rq = [ep.reasoning_quality for ep in self.episodes]
        ax.plot(episodes, rq, alpha=0.3, color="tab:cyan")
        if len(rq) >= w:
            ax.plot(range(w - 1, len(episodes)), self._roll(rq, w), linewidth=2, color="tab:cyan")
        ax.set_title("Reasoning Chain Quality")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Quality")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[Metrics] Plots saved to {path}")

    @staticmethod
    def _roll(values: List[float], window: int) -> List[float]:
        return [sum(values[i:i + window]) / window for i in range(len(values) - window + 1)]

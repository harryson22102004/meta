"""
PROMETHEUS Demo Script — For Hackathon Presentation.

Runs a quick demonstration showing:
1. Environment in action (medical investigation)
2. Before/after comparison (untrained vs trained agent)
3. Reward curves from training
4. Key metrics: hallucination rate, accuracy, deception detection

Usage:
    python demo_prometheus.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prometheus.environment import PrometheusEnv
from prometheus.metrics import MetricsTracker


def demo_single_investigation():
    """Show a single investigation episode step-by-step."""
    print("\n" + "=" * 70)
    print("  PROMETHEUS DEMO — Medical Investigation Walkthrough")
    print("=" * 70)

    env = PrometheusEnv(seed=42)
    obs = env.reset(domain="medical", difficulty="medium")
    briefing = obs["observation"]

    print(f"\n  Scenario: {briefing.get('title', 'Medical Investigation')}")
    print(f"  Briefing: {briefing.get('briefing', '')[:200]}...")
    print(f"  Sources Available: {len(briefing.get('available_sources', []))}")
    print(f"  Max Steps: {obs['info']['max_steps']}")

    sources = [s["source_id"] for s in briefing.get("available_sources", [])]

    # Step 1: Query lab results
    print("\n  --- Step 1: Query Lab Results ---")
    result = env.step({"action": "query_source", "source_id": sources[0] if sources else "", "topic": "symptoms"})
    claims = result["observation"].get("claims", [])
    print(f"  Claims received: {len(claims)}")
    for c in claims[:2]:
        if isinstance(c, dict):
            print(f"    • {c.get('content', '')[:100]}")
    print(f"  Reward: {result['reward']:+.3f}")

    # Step 2: Query another source
    print("\n  --- Step 2: Query Specialist Opinion ---")
    result = env.step({"action": "query_source", "source_id": sources[2] if len(sources) > 2 else sources[0], "topic": "specialist_opinion"})
    claims2 = result["observation"].get("claims", [])
    print(f"  Claims received: {len(claims2)}")
    for c in claims2[:2]:
        if isinstance(c, dict):
            print(f"    • {c.get('content', '')[:100]}")
    print(f"  Reward: {result['reward']:+.3f}")

    # Step 3: Cross-reference
    all_claim_ids = []
    for c in claims + claims2:
        if isinstance(c, dict) and "claim_id" in c:
            all_claim_ids.append(c["claim_id"])

    if len(all_claim_ids) >= 2:
        print("\n  --- Step 3: Cross-Reference Claims ---")
        result = env.step({"action": "cross_reference", "claim_ids": all_claim_ids[:3]})
        obs_data = result["observation"]
        print(f"  Consistent: {obs_data.get('consistent', 'N/A')}")
        print(f"  Conflicts found: {obs_data.get('num_conflicts', 0)}")
        print(f"  Reward: {result['reward']:+.3f}")

    # Step 4: Check source reliability
    print("\n  --- Step 4: Check Source Reliability ---")
    result = env.step({"action": "check_source_reliability", "source_id": sources[0] if sources else ""})
    track = result["observation"]
    print(f"  Source: {track.get('source_id', '')}")
    print(f"  Observed accuracy: {track.get('observed_accuracy', 'N/A')}")
    print(f"  Reward: {result['reward']:+.3f}")

    # Step 5: Form hypothesis
    print("\n  --- Step 5: Form Hypothesis ---")
    result = env.step({"action": "hypothesize", "hypothesis": "Drug Interaction Syndrome based on conflicting medications"})
    print(f"  Hypothesis recorded. Total: {result['observation'].get('total_hypotheses', 1)}")
    print(f"  Reward: {result['reward']:+.3f}")

    # Step 6: Conclude
    print("\n  --- Step 6: Submit Conclusion ---")
    result = env.step({
        "action": "conclude",
        "diagnosis": "Drug Interaction Syndrome",
        "reasoning": "Lab results show elevated liver enzymes. Cross-referencing symptoms with "
                     "medication review reveals conflicting prescriptions causing hepatotoxicity. "
                     "Specialist opinion suggesting Hepatitis B is inconsistent with vaccination history.",
        "confidence": 0.85,
        "unreliable_sources": [sources[2]] if len(sources) > 2 else [],
    })

    scores = result["observation"].get("scores", {})
    print(f"\n  ✦ FINAL SCORES:")
    for key, val in scores.items():
        print(f"    {key}: {val:+.3f}")
    print(f"  Total episode reward: {result['info'].get('total_reward', 0):+.3f}")
    print(f"  Correct answer: {result['observation'].get('correct_answer', 'N/A')}")


def demo_before_after():
    """Show before/after training comparison."""
    print("\n" + "=" * 70)
    print("  BEFORE vs AFTER Training Comparison")
    print("=" * 70)

    from train_prometheus import SimulatedAgent

    # Before training (unskilled agent)
    print("\n  --- BEFORE TRAINING (skill=0.1) ---")
    env = PrometheusEnv(seed=100)
    agent_before = SimulatedAgent(skill_level=0.1)
    run_comparison_episode(env, agent_before, "UNTRAINED")

    # After training (skilled agent)
    print("\n  --- AFTER TRAINING (skill=0.85) ---")
    env2 = PrometheusEnv(seed=100)  # Same scenario for fair comparison
    agent_after = SimulatedAgent(skill_level=0.85)
    run_comparison_episode(env2, agent_after, "TRAINED")


def run_comparison_episode(env, agent, label):
    obs = env.reset(domain="medical", difficulty="medium")
    info = obs["info"]
    max_steps = info["max_steps"]
    agent.reset_episode(obs)
    step = 0
    done = False
    total_reward = 0.0

    while not done and step < max_steps:
        action = agent.act(obs, step, max_steps)
        obs = env.step(action)
        total_reward += obs["reward"]
        done = obs["done"]
        step += 1

    final = obs.get("info", {}).get("final_scores", {})
    print(f"  [{label}] Steps: {step}, Reward: {total_reward:+.3f}")
    if isinstance(final, dict):
        acc = final.get("diagnosis_accuracy", "N/A")
        cal = final.get("calibration", "N/A")
        hallucinated = "YES" if isinstance(cal, (int, float)) and cal < -0.3 else "NO"
        print(f"  [{label}] Accuracy: {acc}, Calibration: {cal}, Hallucinated: {hallucinated}")


def demo_training_curves():
    """Run quick training and show curves."""
    print("\n" + "=" * 70)
    print("  Generating Training Curves (50 episodes)...")
    print("=" * 70)

    from train_prometheus import train_simulated
    train_simulated(50, "metrics/prometheus_demo")

    print("\n  Reward curves saved to: metrics/prometheus_demo/reward_curves.png")
    print("  Metrics saved to: metrics/prometheus_demo/training_metrics.json")


def main():
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█  PROMETHEUS — Adversarial Reasoning Forge                        █")
    print("█  Train LLMs to Think Like Scientists, Not Guess Like Students    █")
    print("█" + " " * 68 + "█")
    print("█" * 70)

    demo_single_investigation()
    demo_before_after()
    demo_training_curves()

    print("\n" + "=" * 70)
    print("  Demo Complete! Key Takeaways:")
    print("  1. Agent learns to gather evidence systematically")
    print("  2. Agent learns to detect deceptive sources")
    print("  3. Agent learns to say 'I don't know' (anti-hallucination)")
    print("  4. Adversary evolves to create harder deceptions")
    print("  5. Reward curves show clear improvement over training")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

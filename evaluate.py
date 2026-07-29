"""
Evaluation script: generates report-ready charts.
1. Q-table heatmap: visualizes learned action preferences per state.
2. Baseline comparison: adaptive RL agent vs a naive "always Hint" policy,
   compared on cumulative simulated reward over many episodes.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

from modules.bayesian_network import build_bayesian_network
from modules.hmm_calibration import build_hmm
from modules.rl_agent import TutorRLAgent, ACTIONS, STATE_SPACE, compute_reward, bucket_probability
from pipeline import run_full_interaction


def plot_q_table_heatmap(agent, save_path="plots/q_table_heatmap.png"):
    state_labels = [f"{p}\n{c}" for p, c in STATE_SPACE]
    q_matrix = np.array([agent.q_table[state] for state in STATE_SPACE])

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(q_matrix, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(ACTIONS)))
    ax.set_xticklabels(ACTIONS)
    ax.set_yticks(range(len(state_labels)))
    ax.set_yticklabels(state_labels)

    for i in range(q_matrix.shape[0]):
        for j in range(q_matrix.shape[1]):
            ax.text(j, i, f"{q_matrix[i, j]:.1f}", ha="center", va="center",
                     color="black", fontsize=8)

    ax.set_title("Learned Q-values: (P(Correct) bucket, Calibration State) x Action")
    fig.colorbar(im, ax=ax, label="Q-value")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def run_baseline_policy(bn_model, hmm_model, num_episodes=500):
    """Naive baseline: always chooses 'Hint' regardless of state."""
    learner_history = []
    previous_accuracy = "Medium"
    cumulative_rewards = []
    total = 0

    for _ in range(num_episodes):
        confidence_level = random.choice(["Low", "Medium", "High"])

        from modules.tutor import get_random_question
        question = get_random_question()

        difficulty_bias = {"Easy": 0.75, "Medium": 0.55, "Hard": 0.35}[question["difficulty"]]
        is_correct = random.random() < difficulty_bias

        action = "Hint"
        outcome = "improved_after_hint" if is_correct else "repeated_mistake"
        reward = compute_reward(outcome)

        total += reward
        cumulative_rewards.append(total)

        from modules.hmm_calibration import encode_observation
        observation = encode_observation(confidence_level, is_correct)
        learner_history.append(observation)
        if len(learner_history) > 20:
            learner_history = learner_history[-20:]
        previous_accuracy = "High" if is_correct else "Low"

    return cumulative_rewards


def run_rl_policy(bn_model, hmm_model, agent, num_episodes=500):
    """Adaptive RL agent, using its already-trained (or in-progress) Q-table."""
    learner_history = []
    previous_accuracy = "Medium"
    cumulative_rewards = []
    total = 0

    for _ in range(num_episodes):
        confidence_level = random.choice(["Low", "Medium", "High"])
        result = run_full_interaction(
            bn_model, hmm_model, agent,
            learner_history, confidence_level, previous_accuracy,
            use_llm=False
        )
        outcome = "needed_solution" if result["selected_action"] == "Reveal" else (
            "improved_after_hint" if result["actually_correct"] else "repeated_mistake"
        )
        reward = compute_reward(outcome)
        total += reward
        cumulative_rewards.append(total)

        learner_history = result["updated_history"]
        previous_accuracy = "High" if result["actually_correct"] else "Low"

    return cumulative_rewards


def plot_comparison(baseline_rewards, rl_rewards, save_path="plots/rl_vs_baseline.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_rewards, label="Baseline (always Hint)", color="gray", linewidth=2)
    plt.plot(rl_rewards, label="Adaptive RL Agent", color="crimson", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Adaptive RL Agent vs. Naive Baseline (Cumulative Reward)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    bn_model = build_bayesian_network()
    hmm_model = build_hmm()

    # Load the already-trained agent (from train.py) for the heatmap
    trained_agent = TutorRLAgent(epsilon=0.0)
    plot_q_table_heatmap(trained_agent)

    # Fair comparison: use the ALREADY-TRAINED agent (from train.py, 3000 episodes),
    # with exploration off so it always exploits its learned policy.
    converged_agent = TutorRLAgent(epsilon=0.0)
    rl_rewards = run_rl_policy(bn_model, hmm_model, converged_agent, num_episodes=500)
    baseline_rewards = run_baseline_policy(bn_model, hmm_model, num_episodes=500)

    plot_comparison(baseline_rewards, rl_rewards)

    print(f"\nFinal cumulative reward - Baseline: {baseline_rewards[-1]}")
    print(f"Final cumulative reward - RL Agent: {rl_rewards[-1]}")



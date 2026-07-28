"""
Batch training script: runs many simulated interactions to let the
RL agent'"'"'s Q-table converge to a meaningful policy.
"""

import random
from modules.bayesian_network import build_bayesian_network
from modules.hmm_calibration import build_hmm
from modules.rl_agent import TutorRLAgent
from pipeline import run_full_interaction

NUM_EPISODES = 3000

def train():
    bn_model = build_bayesian_network()
    hmm_model = build_hmm()
    agent = TutorRLAgent(epsilon=0.3)

    learner_history = []
    previous_accuracy = "Medium"

    for i in range(NUM_EPISODES):
        confidence_level = random.choice(["Low", "Medium", "High"])
        result = run_full_interaction(
            bn_model, hmm_model, agent,
            learner_history, confidence_level, previous_accuracy,
            use_llm=False
        )
        learner_history = result["updated_history"]
        if len(learner_history) > 20:
            learner_history = learner_history[-20:]  # cap history length
        previous_accuracy = "High" if result["actually_correct"] else "Low"

        if (i + 1) % 500 == 0:
            print(f"Episode {i + 1}/{NUM_EPISODES} complete")

    agent.save_q_table()
    print("\nTraining complete. Final Q-table:")
    for state, q_values in agent.q_table.items():
        print(f"{state}: {q_values}")

if __name__ == "__main__":
    train()


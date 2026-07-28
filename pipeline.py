"""
Pipeline: connects Bayesian Network -> HMM -> RL Agent
This simulates one full learner interaction cycle without the UI or LLM tutor.
"""

from modules.bayesian_network import build_bayesian_network, estimate_probability_correct
from modules.hmm_calibration import build_hmm, encode_observation, infer_calibration_state
from modules.rl_agent import TutorRLAgent, bucket_probability, compute_reward

def run_single_interaction(bn_model, hmm_model, agent, learner_history, features):
    # Step 1: Bayesian Network -> P(Correct)
    p_correct = estimate_probability_correct(
        bn_model,
        confidence=features["confidence"],
        difficulty=features["difficulty"],
        time_=features["time"],
        hints=features["hints"],
        previous_accuracy=features["previous_accuracy"],
    )

    # Step 2: HMM -> Calibration state
    calibration_state = infer_calibration_state(hmm_model, learner_history)

    # Step 3: Build RL state, choose action
    prob_bucket = bucket_probability(p_correct)
    rl_state = (prob_bucket, calibration_state)
    action = agent.choose_action(rl_state)

    return {
        "p_correct": p_correct,
        "calibration_state": calibration_state,
        "rl_state": rl_state,
        "selected_action": action,
    }


if __name__ == "__main__":
    bn_model = build_bayesian_network()
    hmm_model = build_hmm()
    agent = TutorRLAgent()

    # Example learner: confident, but history of being wrong
    learner_history = [
        encode_observation("High", False),
        encode_observation("High", False),
    ]

    features = {
        "confidence": "High",
        "difficulty": "Hard",
        "time": "Fast",
        "hints": "No",
        "previous_accuracy": "Low",
    }

    result = run_single_interaction(bn_model, hmm_model, agent, learner_history, features)

    print("=== Interaction Result ===")
    print(f"P(Correct):          {result['p_correct']:.2f}")
    print(f"Calibration State:   {result['calibration_state']}")
    print(f"RL State:            {result['rl_state']}")
    print(f"Selected Action:     {result['selected_action']}")

    # Simulate feedback and update Q-table
    reward = compute_reward("improved_after_hint")
    next_state = ("Medium", "Well-calibrated")
    agent.update(result["rl_state"], result["selected_action"], reward, next_state)
    agent.save_q_table()
    print(f"\nReward applied: {reward}")
    print("Q-table updated and saved.")


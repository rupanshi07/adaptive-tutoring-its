"""
Full Pipeline: Question -> Answer -> BN -> HMM -> RL -> Tutor Feedback -> Reward
Simulates one complete learner interaction cycle end-to-end.
"""

import random

from modules.bayesian_network import build_bayesian_network, estimate_probability_correct
from modules.hmm_calibration import build_hmm, encode_observation, infer_calibration_state
from modules.rl_agent import TutorRLAgent, bucket_probability, compute_reward
from modules.tutor import get_random_question, generate_feedback


def simulate_learner_answer(question, confidence_level):
    """
    Mock learner simulation: higher difficulty + higher declared confidence
    doesn't guarantee correctness. This stands in for a real answer-checker
    (e.g., exact match, LLM-graded, or numeric comparison) later.
    """
    difficulty_bias = {"Easy": 0.75, "Medium": 0.55, "Hard": 0.35}[question["difficulty"]]
    is_correct = random.random() < difficulty_bias
    return is_correct


def determine_outcome(is_correct, action):
    """
    Maps (correctness, chosen action) to one of the reward categories
    defined in compute_reward(). This models "what actually happened next."
    """
    if action == "Reveal":
        return "needed_solution"
    if is_correct:
        return "improved_after_hint"
    return "repeated_mistake"


def run_full_interaction(bn_model, hmm_model, agent, learner_history, confidence_level, previous_accuracy):
    question = get_random_question()

    is_correct = simulate_learner_answer(question, confidence_level)

    time_taken = random.choice(["Fast", "Slow"])
    hints_used = "No"

    p_correct = estimate_probability_correct(
        bn_model,
        confidence=confidence_level,
        difficulty=question["difficulty"],
        time_=time_taken,
        hints=hints_used,
        previous_accuracy=previous_accuracy,
    )

    observation = encode_observation(confidence_level, is_correct)
    updated_history = learner_history + [observation]
    calibration_state = infer_calibration_state(hmm_model, updated_history)

    prob_bucket = bucket_probability(p_correct)
    rl_state = (prob_bucket, calibration_state)
    action = agent.choose_action(rl_state)

    feedback = generate_feedback(question, action)

    outcome = determine_outcome(is_correct, action)
    reward = compute_reward(outcome)

    next_prob_bucket = bucket_probability(0.5)
    next_state = (next_prob_bucket, calibration_state)
    agent.update(rl_state, action, reward, next_state)

    return {
        "question": question["question"],
        "difficulty": question["difficulty"],
        "confidence_declared": confidence_level,
        "actually_correct": is_correct,
        "p_correct": p_correct,
        "calibration_state": calibration_state,
        "selected_action": action,
        "feedback": feedback,
        "outcome": outcome,
        "reward": reward,
        "updated_history": updated_history,
    }


if __name__ == "__main__":
    bn_model = build_bayesian_network()
    hmm_model = build_hmm()
    agent = TutorRLAgent()

    learner_history = []
    previous_accuracy = "Medium"

    NUM_ROUNDS = 5
    for round_num in range(1, NUM_ROUNDS + 1):
        confidence_level = random.choice(["Low", "Medium", "High"])

        result = run_full_interaction(
            bn_model, hmm_model, agent,
            learner_history, confidence_level, previous_accuracy
        )

        print(f"\n{'='*50}")
        print(f"Round {round_num}")
        print(f"{'='*50}")
        print(f"Question:            {result['question']}")
        print(f"Difficulty:          {result['difficulty']}")
        print(f"Declared Confidence: {result['confidence_declared']}")
        print(f"Actually Correct:    {result['actually_correct']}")
        print(f"P(Correct) estimate: {result['p_correct']:.2f}")
        print(f"Calibration State:   {result['calibration_state']}")
        print(f"Selected Action:     {result['selected_action']}")
        print(f"Feedback:            {result['feedback']}")
        print(f"Outcome:             {result['outcome']}")
        print(f"Reward:              {result['reward']}")

        learner_history = result["updated_history"]
        previous_accuracy = "High" if result["actually_correct"] else "Low"

    agent.save_q_table()
    print(f"\n{'='*50}")
    print("Simulation complete. Q-table saved to data/q_table.json")

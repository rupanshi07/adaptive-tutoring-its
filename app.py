"""
Module 6: Streamlit UI
Interactive front-end for the adaptive tutoring pipeline.
Students type real answers; Gemini grades them automatically.
"""

import streamlit as st
import time

from modules.bayesian_network import build_bayesian_network, estimate_probability_correct
from modules.hmm_calibration import build_hmm, encode_observation, infer_calibration_state
from modules.rl_agent import TutorRLAgent, bucket_probability, compute_reward
from modules.tutor import get_random_question, generate_feedback, grade_answer


@st.cache_resource
def load_models():
    bn_model = build_bayesian_network()
    hmm_model = build_hmm()
    agent = TutorRLAgent(epsilon=0.1)
    return bn_model, hmm_model, agent


def init_session_state():
    defaults = {
        "history": [],
        "previous_accuracy": "Medium",
        "current_question": get_random_question(),
        "question_start_time": time.time(),
        "round_num": 1,
        "feedback": None,
        "awaiting_next": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def next_question():
    st.session_state.current_question = get_random_question()
    st.session_state.question_start_time = time.time()
    st.session_state.round_num += 1
    st.session_state.feedback = None
    st.session_state.awaiting_next = False


def main():
    st.set_page_config(page_title="Adaptive Tutoring System", page_icon=":books:")
    st.title("Adaptive Tutoring System")
    st.caption("Bayesian Networks + Hidden Markov Models + Reinforcement Learning")

    bn_model, hmm_model, agent = load_models()
    init_session_state()

    q = st.session_state.current_question

    st.subheader(f"Round {st.session_state.round_num}")
    st.markdown(f"**Difficulty:** {q['difficulty']}")
    st.markdown(f"### {q['question']}")

    if not st.session_state.awaiting_next:
        student_answer = st.text_area(
            "Your answer:",
            key=f"answer_{st.session_state.round_num}",
            placeholder="Type your answer here...",
        )
        confidence_level = st.select_slider(
            "How confident are you in your answer?",
            options=["Low", "Medium", "High"],
            value="Medium",
            key=f"conf_{st.session_state.round_num}",
        )

        if st.button("Submit Answer"):
            if not student_answer.strip():
                st.warning("Please type an answer before submitting.")
            else:
                elapsed_seconds = time.time() - st.session_state.question_start_time
                time_taken = "Fast" if elapsed_seconds < 15 else "Slow"
                with st.spinner("Grading your answer..."):
                    is_correct = grade_answer(q, student_answer)

                p_correct = estimate_probability_correct(
                    bn_model,
                    confidence=confidence_level,
                    difficulty=q["difficulty"],
                    time_=time_taken,
                    hints="No",
                    previous_accuracy=st.session_state.previous_accuracy,
                )

                observation = encode_observation(confidence_level, is_correct)
                updated_history = st.session_state.history + [observation]
                if len(updated_history) > 20:
                    updated_history = updated_history[-20:]
                calibration_state = infer_calibration_state(hmm_model, updated_history)

                prob_bucket = bucket_probability(p_correct)
                rl_state = (prob_bucket, calibration_state)
                action = agent.choose_action(rl_state)

                with st.spinner("Generating feedback..."):
                    feedback_text = generate_feedback(q, action)

                outcome = "needed_solution" if action == "Reveal" else (
                    "improved_after_hint" if is_correct else "repeated_mistake"
                )
                reward = compute_reward(outcome)
                next_state = (bucket_probability(0.5), calibration_state)
                agent.update(rl_state, action, reward, next_state)
                agent.save_q_table()

                st.session_state.history = updated_history
                st.session_state.previous_accuracy = "High" if is_correct else "Low"
                st.session_state.feedback = {
                    "student_answer": student_answer,
                    "is_correct": is_correct,
                    "p_correct": p_correct,
                    "calibration_state": calibration_state,
                    "action": action,
                    "feedback": feedback_text,
                    "reward": reward,
                    "response_time": time_taken,
                    "elapsed_seconds": round(elapsed_seconds, 1),
                }
                st.session_state.awaiting_next = True
                st.rerun()

    else:
        fb = st.session_state.feedback
        st.divider()

        st.markdown(f"**Your answer:** {fb['student_answer']}")
        if fb["is_correct"]:
            st.success("Graded as: Correct")
        else:
            st.error("Graded as: Incorrect")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("P(Correct) estimate", f"{fb['p_correct']:.0%}")
        with col2:
            st.metric("Calibration State", fb["calibration_state"])

        st.caption(f"Response time: {fb['response_time']} ({fb['elapsed_seconds']}s)")
        st.markdown(f"**Tutor Action:** `{fb['action']}`")

        if fb["action"] == "Hint":
            st.info(fb["feedback"])
        elif fb["action"] == "Explanation":
            st.success(fb["feedback"])
        elif fb["action"] == "Reveal":
            st.warning(fb["feedback"])
        else:
            st.error(fb["feedback"])

        st.caption(f"Reward signal (for RL training): {fb['reward']}")

        if st.button("Next Question"):
            next_question()
            st.rerun()

    with st.sidebar:
        st.subheader("Learner State")
        st.write(f"Previous Accuracy: {st.session_state.previous_accuracy}")
        st.write(f"Observation history length: {len(st.session_state.history)}")
        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()











# Adaptive Confidence Calibration in Intelligent Tutoring Systems



An adaptive AI tutoring framework that personalizes instructional support

by combining Probabilistic Graphical Models (Bayesian Networks, Hidden

Markov Models) with Reinforcement Learning, and uses the Gemini API as

the natural-language tutor layer.



## Overview



Traditional Intelligent Tutoring Systems give every learner the same

feedback regardless of whether they are over-confident, under-confident,

or genuinely uncertain. This project estimates a learner's hidden

confidence-calibration state and dynamically chooses the most appropriate

tutoring action (Hint, Explanation, Retry, or Reveal Solution) rather than

responding identically to every student.



## Architecture

Student types answer

|

v

Gemini API (grade\_answer) --> Correct / Incorrect

|

v

Bayesian Network (pgmpy) --> P(Correct) from Confidence, Difficulty, Response Time, Hints, Previous Accuracy

|

v

Hidden Markov Model (hmmlearn) --> Confidence Calibration state

(Over-confident / Well-calibrated / Under-confident)

|

v

Reinforcement Learning Agent (tabular Q-learning) --> Selects tutoring action

|

v

Gemini API (generate\_feedback) --> Dynamic hint / explanation / retry / reveal text shown to student

|

v

Reward computed from outcome --> Q-table updated

## Project Structure

adaptive-tutoring-its/

|-- modules/

| |-- bayesian\_network.py # Module 3: P(Correct) estimation

| |-- hmm\_calibration.py # Module 4: confidence calibration inference

| |-- rl\_agent.py # Module 5: tabular Q-learning tutoring agent

| |-- tutor.py # Module 1: question bank, Gemini grading + feedback

|-- data/

| |-- q\_table.json # Persisted learned Q-table

|-- pipeline.py # Wires Modules 3-4-5-1 into one interaction cycle

|-- train.py # Batch-trains the RL agent (no LLM calls, fast/free)

|-- app.py # Streamlit UI (Module 6): full live demo

|-- requirements.txt



\*\*Run the live interactive demo:\*\*

streamlit run app.py

Opens a browser at `http://localhost:8501` where you can answer real

questions, get graded by Gemini, and see the adaptive tutor respond.



## How It Works



1\. A question is drawn from a fixed bank of Data Structures \& Algorithms

&#x20;  questions spanning Easy/Medium/Hard difficulty.

2\. The student types a free-text answer. Gemini grades it as correct or

&#x20;  incorrect by comparing it against a reference answer.

3\. Response time is measured automatically (not self-reported): under

&#x20;  15 seconds is bucketed as "Fast," otherwise "Slow."

4\. The student self-reports their confidence (Low/Medium/High) via a

&#x20;  slider -- this is intentional, since the calibration model needs to

&#x20;  compare stated confidence against actual correctness.

5\. The Bayesian Network estimates P(Correct) from Confidence, Difficulty,

&#x20;  Response Time, Hint usage, and Previous Accuracy.

6\. The Hidden Markov Model looks at the sequence of past

&#x20;  (confidence, correctness) observations and infers whether the learner

&#x20;  is currently Over-confident, Well-calibrated, or Under-confident.

7\. The Reinforcement Learning agent takes (P(Correct) bucket, Calibration

&#x20;  state) as its state and selects the tutoring action with the highest

&#x20;  learned Q-value.

8\. Gemini generates the actual hint/explanation/retry/reveal text shown

&#x20;  to the student, based on the selected action.

9\. A reward is computed from the outcome and the Q-table is updated,

&#x20;  so the policy keeps improving across sessions.



## Novelty



The contribution is not using an LLM or reinforcement learning alone --

it is the integration of three complementary reasoning layers (Bayesian

Network, Hidden Markov Model, Reinforcement Learning) to adapt teaching

strategy to each individual learner's estimated knowledge state and

confidence-calibration pattern, rather than giving identical feedback

to everyone.



## Tech Stack



\- \*\*pgmpy\*\* -- Bayesian Network construction and inference

\- \*\*hmmlearn\*\* -- Hidden Markov Model for calibration state inference

\- \*\*NumPy\*\* -- Q-table representation and updates

\- \*\*Google Gemini API\*\* (`google-genai`) -- answer grading and adaptive

&#x20; feedback generation

\- \*\*Streamlit\*\* -- interactive web UI



## Limitations / Future Work



\- Question bank is currently a fixed set of 8 DSA questions; could be

&#x20; expanded or generated dynamically.

\- RL agent uses tabular Q-learning on a small discrete state space;

&#x20; a larger/continuous state space would need function approximation

&#x20; (e.g., a neural Q-network).

\- Training currently uses simulated learner behavior; validating against

&#x20; real student interaction data is a natural next step.






"""
Module 1: Intelligent Tutor (Gemini-powered version)
Generates questions from a fixed bank, but produces adaptive feedback
(hints, explanations, retry prompts) using the Gemini API based on
the action selected by the RL agent.
"""

import os
import random
from google import genai

QUESTION_BANK = [
    {
        "id": 1,
        "question": "What is Binary Search?",
        "difficulty": "Medium",
        "correct_answer_summary": "An algorithm that repeatedly divides a sorted search space in half.",
    },
    {
        "id": 2,
        "question": "What is the time complexity of Bubble Sort in the worst case?",
        "difficulty": "Easy",
        "correct_answer_summary": "O(n^2)",
    },
    {
        "id": 3,
        "question": "What data structure uses FIFO (First In First Out) order?",
        "difficulty": "Easy",
        "correct_answer_summary": "Queue",
    },
    {
        "id": 4,
        "question": "What is the space complexity of an iterative Depth-First Search using an explicit stack?",
        "difficulty": "Hard",
        "correct_answer_summary": "O(V) in the worst case, where V is the number of vertices.",
    },
    {
        "id": 5,
        "question": "What is a hash collision?",
        "difficulty": "Medium",
        "correct_answer_summary": "When two different keys map to the same hash bucket/index.",
    },
    {
        "id": 6,
        "question": "What is the key property of a Balanced Binary Search Tree (like AVL)?",
        "difficulty": "Hard",
        "correct_answer_summary": "The height difference between left and right subtrees is bounded (e.g., at most 1 for AVL).",
    },
    {
        "id": 7,
        "question": "What does the Greedy algorithm design paradigm rely on?",
        "difficulty": "Medium",
        "correct_answer_summary": "Making the locally optimal choice at each step, hoping it leads to a global optimum.",
    },
    {
        "id": 8,
        "question": "Why is a Linked List preferred over an Array for frequent insertions in the middle?",
        "difficulty": "Easy",
        "correct_answer_summary": "Insertion is O(1) once the position is known, since no shifting of elements is required.",
    },
]

_client = None

def get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def get_question_by_difficulty(difficulty):
    candidates = [q for q in QUESTION_BANK if q["difficulty"] == difficulty]
    return random.choice(candidates) if candidates else random.choice(QUESTION_BANK)


def get_random_question():
    return random.choice(QUESTION_BANK)


def _build_prompt(question, action):
    base = (
        f"You are an adaptive tutor helping a student with this question:\n"
        f"\"{question['question']}\"\n"
        f"The correct answer is: {question['correct_answer_summary']}\n\n"
    )
    if action == "Hint":
        return base + (
            "Give ONE short hint (max 2 sentences) that nudges the student toward "
            "the answer WITHOUT revealing it directly."
        )
    elif action == "Explanation":
        return base + (
            "Give a clear, concise explanation (3-4 sentences) of the correct answer, "
            "written for a student who just got this wrong or wants more depth."
        )
    elif action == "Retry":
        return base + (
            "Write ONE short, encouraging sentence telling the student to try again "
            "without giving any hint or revealing the answer."
        )
    elif action == "Reveal":
        return base + (
            "Clearly state the correct answer in 1-2 sentences, in a supportive tone."
        )
    else:
        return base + "Give a brief, friendly transition to the next question."


def generate_feedback(question, action, use_llm=True):
    """
    action: one of 'Hint', 'Explanation', 'Retry', 'Reveal'
    Calls Gemini to generate the actual feedback text dynamically.
    Falls back to a safe static message if the API call fails.
    """
    if not use_llm:
        return f"[static] Action={action}"
    try:
        client = get_client()
        prompt = _build_prompt(question, action)
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as e:
        return f"[Tutor unavailable, fallback message] Action was: {action}. Error: {e}"


if __name__ == "__main__":
    q = get_question_by_difficulty("Hard")
    print(f"Question: {q['question']} (Difficulty: {q['difficulty']})")

    for action in ["Hint", "Explanation", "Retry", "Reveal"]:
        print(f"\n[Action: {action}]")
        print(generate_feedback(q, action))




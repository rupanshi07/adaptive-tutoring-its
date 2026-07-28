"""
Module 1: Intelligent Tutor (Mock version)
Generates questions and produces adaptive feedback based on the
action selected by the RL agent. Swap generate_feedback() internals
for a real LLM API call later without touching the rest of the pipeline.
"""

import random

QUESTION_BANK = [
    {
        "id": 1,
        "question": "What is Binary Search?",
        "difficulty": "Medium",
        "correct_answer_summary": "An algorithm that repeatedly divides a sorted search space in half.",
        "hint": "It repeatedly divides the search space. Does it need the array sorted?",
        "explanation": "Binary Search works only on sorted arrays. It compares the target to the middle element, then discards the half where the target cannot be, repeating until found. Time complexity is O(log n).",
    },
    {
        "id": 2,
        "question": "What is the time complexity of Bubble Sort in the worst case?",
        "difficulty": "Easy",
        "correct_answer_summary": "O(n^2)",
        "hint": "Think about how many comparisons happen when the array is in reverse order.",
        "explanation": "Bubble Sort compares each adjacent pair and swaps if needed, repeating n times for n elements, giving O(n^2) comparisons in the worst case.",
    },
    {
        "id": 3,
        "question": "What data structure uses FIFO (First In First Out) order?",
        "difficulty": "Easy",
        "correct_answer_summary": "Queue",
        "hint": "Think of a line at a ticket counter — who gets served first?",
        "explanation": "A Queue processes elements in the order they were added: the first element inserted is the first one removed.",
    },
    {
        "id": 4,
        "question": "What is the space complexity of an iterative Depth-First Search using an explicit stack?",
        "difficulty": "Hard",
        "correct_answer_summary": "O(V) in the worst case, where V is the number of vertices.",
        "hint": "Consider how many nodes the stack could hold at once in a skewed graph.",
        "explanation": "In the worst case (e.g., a graph shaped like a long chain), the stack can hold up to V vertices before backtracking, giving O(V) space complexity.",
    },
    {
        "id": 5,
        "question": "What is a hash collision?",
        "difficulty": "Medium",
        "correct_answer_summary": "When two different keys map to the same hash bucket/index.",
        "hint": "Think about what happens when a hash function isn't perfectly unique for every input.",
        "explanation": "A hash collision occurs when two distinct keys produce the same hash value, requiring a resolution strategy like chaining or open addressing.",
    },
    {
        "id": 6,
        "question": "What is the key property of a Balanced Binary Search Tree (like AVL)?",
        "difficulty": "Hard",
        "correct_answer_summary": "The height difference between left and right subtrees is bounded (e.g., at most 1 for AVL).",
        "hint": "Think about what 'balanced' controls — is it the number of nodes, or something about height?",
        "explanation": "A balanced BST keeps the height difference between subtrees small (e.g., AVL trees enforce a max difference of 1), which guarantees O(log n) operations.",
    },
    {
        "id": 7,
        "question": "What does the Greedy algorithm design paradigm rely on?",
        "difficulty": "Medium",
        "correct_answer_summary": "Making the locally optimal choice at each step, hoping it leads to a global optimum.",
        "hint": "Does it look ahead at all future steps, or just decide based on what looks best right now?",
        "explanation": "Greedy algorithms make the best local decision at each step without reconsidering previous choices, which works optimally only for problems with the greedy-choice property.",
    },
    {
        "id": 8,
        "question": "Why is a Linked List preferred over an Array for frequent insertions in the middle?",
        "difficulty": "Easy",
        "correct_answer_summary": "Insertion is O(1) once the position is known, since no shifting of elements is required.",
        "hint": "Think about what an array has to do to make room for a new middle element.",
        "explanation": "Arrays require shifting all subsequent elements to insert in the middle (O(n)), while a Linked List just relinks pointers (O(1) at a known position).",
    },
]


def get_question_by_difficulty(difficulty):
    candidates = [q for q in QUESTION_BANK if q["difficulty"] == difficulty]
    return random.choice(candidates) if candidates else random.choice(QUESTION_BANK)


def get_random_question():
    return random.choice(QUESTION_BANK)


def generate_feedback(question, action):
    """
    action: one of 'Hint', 'Explanation', 'Retry', 'Reveal'
    This is the mock version — swap this function's body for a real
    LLM API call later. Keep the same signature and return type.
    """
    if action == "Hint":
        return f"Hint: {question['hint']}"
    elif action == "Explanation":
        return f"Explanation: {question['explanation']}"
    elif action == "Retry":
        return "That's not quite right. Take another look and try again — no hints this time."
    elif action == "Reveal":
        return f"Solution: {question['correct_answer_summary']}"
    else:
        return "Let's move on to the next question."


if __name__ == "__main__":
    q = get_question_by_difficulty("Hard")
    print(f"Question: {q['question']} (Difficulty: {q['difficulty']})")

    for action in ["Hint", "Explanation", "Retry", "Reveal"]:
        print(f"\n[Action: {action}]")
        print(generate_feedback(q, action))

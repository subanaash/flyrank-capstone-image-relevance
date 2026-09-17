"""
Evaluation script for the AI Image Understanding & Content Matching Engine.

Run this AFTER posts and images have been ingested.
It fires a real request per post, checks whether the returned match's subject
matches the post's actual animal, and reports precision/recall-style numbers.
"""
import requests

BASE_URL = "http://localhost:8000"

# Expected animal per post title (same order as posts.json)
LABELED_POSTS = {
    "The Cunning Red Fox: Nature's Clever Survivor": "fox",
    "Wolves: The Pack Hunters of the Northern Forests": "wolf",
    "Man's Best Friend: The Loyal Domestic Dog": "dog",
    "Bears in the Wild: Power and Patience": "bear",
    "The Graceful Deer: Symbol of the Forest": "deer",
    "Fox Behavior: Why They Thrive Near Humans": "fox",
    "The Social Structure of a Wolf Pack": "wolf",
    "Training Your Dog: Building Trust Through Consistency": "dog",
    "Bear Hibernation: Surviving the Winter Months": "bear",
    "Deer Antlers: A Yearly Cycle of Growth": "deer",
}


def run_eval():
    results = []

    import sys
    sys.path.insert(0, "src")
    from models import get_session, Post

    session = get_session()
    all_posts = session.exec(__import__("sqlmodel").select(Post)).all()
    session.close()

    for post in all_posts:
        expected_animal = None
        for title, animal in LABELED_POSTS.items():
            if title == post.title:
                expected_animal = animal
                break
        if not expected_animal:
            continue

        resp = requests.get(f"{BASE_URL}/posts/{post.id}/images")
        if resp.status_code != 200:
            results.append({"post_id": post.id, "expected": expected_animal, "outcome": "REQUEST_FAILED"})
            continue

        data = resp.json()
        match = data.get("match")

        if match is None:
            results.append({"post_id": post.id, "expected": expected_animal, "outcome": "NO_MATCH_RETURNED"})
            continue

        matched_subject = match["subject"].lower()
        correct = expected_animal in matched_subject

        results.append({
            "post_id": post.id,
            "expected": expected_animal,
            "matched_subject": matched_subject,
            "similarity": match["similarity"],
            "outcome": "CORRECT" if correct else "WRONG_MATCH",
        })

    correct_count = sum(1 for r in results if r["outcome"] == "CORRECT")
    no_match_count = sum(1 for r in results if r["outcome"] == "NO_MATCH_RETURNED")
    wrong_count = sum(1 for r in results if r["outcome"] == "WRONG_MATCH")

    print("=== Evaluation Results ===")
    for r in results:
        print(r)

    print(f"\nCorrect matches: {correct_count}/{len(results)}")
    print(f"No match returned (guard rejected all candidates): {no_match_count}")
    print(f"Wrong matches (guard approved something incorrect): {wrong_count}")
    print(f"\nPrecision (of approved matches, how many were correct): "
          f"{correct_count / (correct_count + wrong_count) if (correct_count + wrong_count) > 0 else 0:.2%}")
    print(f"Recall (of all posts, how many got a correct match): {correct_count / len(results):.2%}")


if __name__ == "__main__":
    run_eval()
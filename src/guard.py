SIMILARITY_THRESHOLD = 0.35
CONFIDENCE_THRESHOLD = 0.5


def extract_animal_keyword(text: str) -> str | None:
    """Small keyword matcher for this animal-only dataset, handling plurals."""
    animals = {
        "fox": ["fox", "foxes"],
        "wolf": ["wolf", "wolves"],
        "dog": ["dog", "dogs"],
        "bear": ["bear", "bears"],
        "deer": ["deer", "deers"],
    }
    text_lower = text.lower()
    for animal, forms in animals.items():
        if any(form in text_lower for form in forms):
            return animal
    return None


def check_match(post_title: str, post_body: str, image_subject: str, image_confidence: float, similarity: float) -> dict:
    """Runs the three-signal mismatch guard. Returns {result, reason}."""

    post_animal = extract_animal_keyword(post_title) or extract_animal_keyword(post_body)
    image_animal = extract_animal_keyword(image_subject)

    if post_animal and image_animal and post_animal != image_animal:
        return {
            "result": "rejected",
            "reason": f"Animal category mismatch: post is about '{post_animal}', image shows '{image_animal}'",
        }

    if similarity < SIMILARITY_THRESHOLD:
        return {
            "result": "rejected",
            "reason": f"Similarity too low ({similarity:.2f} < {SIMILARITY_THRESHOLD})",
        }

    if image_confidence < CONFIDENCE_THRESHOLD:
        return {
            "result": "rejected",
            "reason": f"Image tag confidence too low ({image_confidence:.2f} < {CONFIDENCE_THRESHOLD})",
        }

    return {
        "result": "approved",
        "reason": f"Match: similarity {similarity:.2f}, image confidence {image_confidence:.2f}",
    }
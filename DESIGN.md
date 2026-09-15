# Design Doc — AI Image Understanding & Content Matching Engine

## Problem

Given a library of images and a set of blog posts, automatically suggest the most relevant image for each post — based on what the image actually shows, not filenames or keywords — and refuse to suggest an image when nothing is a confident enough match.

## Non-goal

This is not a general image search engine or a content management system. It does not handle image uploads from end users, does not generate images, and does not build a frontend UI — the review workflow is API endpoints plus a simple table, not a polished interface.

## Dataset

~50 images across 5 categories: red fox, wolf, dog, bear, deer (all wildlife/animal photography, sourced from Unsplash/Pexels under free licenses). This set is deliberately chosen because fox vs. wolf is a genuinely hard visual discrimination case — similar coloring, similar build — which makes it a real test of the mismatch guard rather than an easy win.

10+ blog posts, each written about one specific animal, used as the matching targets.

## Image metadata schema

Every image is processed once through a vision model and produces:

```json
{
  "subject": "red fox",
  "category": "animal",
  "attributes": ["orange fur", "wild", "forest"],
  "caption": "A red fox standing in a forest",
  "confidence": 0.94
}
```

- `subject`: free text, but validated non-empty
- `category`: one of a small closed list (`animal`, `other`) — kept simple since this dataset is animal-only
- `attributes`: array of short descriptive tags, 1-6 items
- `caption`: one sentence, max 200 characters
- `confidence`: float 0.0-1.0

Low-confidence results (below a set threshold, e.g. 0.6) are flagged for review rather than accepted silently.

## Matching strategy

1. Embed each image's `caption` into a vector.
2. Embed each blog post's full text into a vector.
3. For each post, rank all images by cosine similarity between the post vector and each image vector.
4. Pass the top-ranked candidate(s) through the mismatch guard before returning a suggestion.

## The mismatch guard

Combines three signals before approving a suggestion:

1. **Tag overlap check** — does the image's `subject`/`attributes` semantically relate to the post's stated subject? (e.g. post about "red fox" vs image tagged "wolf" → explicit category mismatch)
2. **Similarity threshold** — cosine similarity must clear a minimum bar (tuned against the labeled eval set in Phase 4).
3. **Confidence check** — the image's own vision-model confidence must also clear a minimum bar; a low-confidence tag makes even a numerically similar match untrustworthy.

If any check fails, the guard returns `REJECTED` with a specific, human-readable reason (e.g. `"Animal category mismatch: expected fox, detected wolf"`), not a silent low score.

If no image clears all three checks, the API returns "no confident match" with the reasons, rather than returning the best-available-but-still-bad option.

## Database design

- **images**: id, file_path, subject, category, attributes (array), caption, confidence, created_at
- **image_vectors**: image_id (FK), embedding (vector), indexed for similarity search
- **posts**: id, title, body, created_at
- **post_vectors**: post_id (FK), embedding (vector)
- **suggestions**: id, post_id (FK), image_id (FK), similarity_score, guard_result (approved/rejected), guard_reason, created_at
- **reviews**: id, suggestion_id (FK), decision (approved/rejected), reviewed_at

Indexes: `images.subject`, `post_vectors.post_id`, `suggestions.post_id` — the lookups this system does most often (find suggestions for a post, find an image's vector).

## API surface (sketch)

- `POST /images/ingest` — trigger batch vision processing on the image corpus (background job)
- `GET /posts/:id/images` — return ranked, guarded image suggestions for a post
- `POST /suggestions/:id/review` — approve or reject a suggestion
- `GET /suggestions/:id` — inspect why an image was suggested or rejected

## Stack

- Python + FastAPI
- Ollama, local — vision model (`llava` or `moondream`) + embeddings (`all-minilm`)
- PostgreSQL via Docker (in-array embeddings at this scale; pgvector optional, not required for ~50 images)
- Inngest for the batch vision processing job (reusing the background-job pattern already built)
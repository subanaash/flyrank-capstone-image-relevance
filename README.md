# AI Image Understanding & Content Matching Engine

Matches blog posts to the most relevant image from a 50-image library using AI vision tagging, embeddings, and a mismatch guard that refuses bad matches instead of guessing.

## What it does

1. **Ingest images** — each image is sent to a vision model (Groq, `qwen/qwen3.8-27b`) which returns structured tags: subject, category, attributes, caption, confidence.
2. **Embed** — each image's caption and each post's text are embedded locally via Ollama (`all-minilm`).
3. **Match** — for a given post, all images are ranked by cosine similarity.
4. **Guard** — before approving a match, three checks run: animal-keyword overlap (post vs image subject), similarity threshold (0.35), confidence threshold (0.5). If any fails, the match is rejected with a specific reason instead of silently returning a bad guess.

## Stack

Python, FastAPI, SQLModel, PostgreSQL (Docker), Groq (vision), Ollama (embeddings, local).

## Setup

```
docker run --name capstone-postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=capstone -p 5433:5432 -d postgres:16
pip install -r requirements.txt
ollama pull all-minilm
```

`.env`:
```
DATABASE_URL=postgresql+pg8000://postgres:postgres@localhost:5433/capstone
GROQ_API_KEY=your_key
VISION_MODEL=qwen/qwen3.8-27b
OLLAMA_URL=http://localhost:11434
EMBED_MODEL=all-minilm
```

## Run

```
cd src
uvicorn main:app --reload --port 8000
```

Load data (from project root):
```
curl -X POST http://localhost:8000/posts/bulk -d "@posts.json"
curl -X POST http://localhost:8000/images/ingest
```

## API

| Endpoint | Purpose |
|---|---|
| `POST /images/ingest` | Tags and embeds all images in `/images` |
| `POST /posts/bulk` | Loads blog posts |
| `GET /posts/{id}/images` | Returns best match + all candidates with guard reasoning |
| `POST /suggestions/{id}/review` | Human approves/rejects a suggestion |

## Evaluation

Run `python evaluate.py` from project root (server must be running). Tests all posts against a labeled set of expected animals.

**Result: 6/11 correct, 4 rejected by guard (no confident match), 1 wrong.**
**Precision: 85.71% — Recall: 54.55%**

The guard prioritizes not being wrong over always answering — when it does approve a match, it's right 86% of the time.

## What I'd fix with more time

- Keyword matching checked post title+body combined at first, which misfired when a post mentioned a different animal for contrast (e.g. a wolf post mentioning "fox"). Fixed by checking title first, body only as fallback.
- Recall (54.55%) is lower than ideal — some correct images exist but score just under the similarity threshold. Would tune the threshold or improve caption quality with better prompting.
- Would add a coarser fallback: if the strict guard rejects everything, return the closest same-category match with a "low confidence" flag instead of nothing.

## Screenshot

<img width="1092" height="903" alt="image" src="https://github.com/user-attachments/assets/31ec4ee4-fc9d-4b70-8bac-88c02dbb8426" />

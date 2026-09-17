import os
import json
import glob
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlmodel import select
from dotenv import load_dotenv
import time

from models import init_db, get_session, Image, ImageVector, Post, PostVector, Suggestion, Review
from ollama_client import tag_image, get_embedding, cosine_similarity
from guard import check_match

load_dotenv()

app = FastAPI(title="AI Image Understanding & Content Matching Engine")

IMAGES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "images")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/images/ingest")
def ingest_images():
    session = get_session()
    files = glob.glob(os.path.join(IMAGES_DIR, "*.jpg")) + glob.glob(os.path.join(IMAGES_DIR, "*.jpeg")) + glob.glob(os.path.join(IMAGES_DIR, "*.png"))

    results = []
    skipped = []
    for file_path in files:
        existing = session.exec(select(Image).where(Image.file_path == file_path)).first()
        if existing:
            continue

        tags = tag_image(file_path)
        time.sleep(21)

        if tags.get("error"):
            skipped.append({"file_path": file_path, "reason": tags["error"]})
            continue

        image = Image(
            file_path=file_path,
            subject=tags["subject"],
            category=tags["category"],
            attributes=tags["attributes"],
            caption=tags["caption"],
            confidence=tags["confidence"],
        )
        session.add(image)
        session.commit()
        session.refresh(image)

        embedding = get_embedding(tags["caption"] or tags["subject"])
        vector = ImageVector(image_id=image.id, embedding=embedding)
        session.add(vector)
        session.commit()

        results.append({"id": image.id, "file_path": file_path, "subject": tags["subject"], "confidence": tags["confidence"]})

    session.close()
    return {"ingested": len(results), "images": results, "skipped": skipped}


class PostIn(BaseModel):
    title: str
    body: str


@app.post("/posts", status_code=201)
def create_post(payload: PostIn):
    session = get_session()
    post = Post(title=payload.title, body=payload.body)
    session.add(post)
    session.commit()
    session.refresh(post)

    embedding = get_embedding(payload.title + " " + payload.body)
    vector = PostVector(post_id=post.id, embedding=embedding)
    session.add(vector)
    session.commit()

    result = {"id": post.id, "title": post.title}
    session.close()

    return result


@app.post("/posts/bulk", status_code=201)
def create_posts_bulk(posts: list[PostIn]):
    results = []
    for p in posts:
        results.append(create_post(p))
    return {"created": len(results)}


@app.get("/posts/{post_id}/images")
def suggest_images(post_id: int):
    session = get_session()

    post = session.get(Post, post_id)
    if not post:
        raise HTTPException(status_code=404, detail=f"Post {post_id} not found")

    post_vector = session.exec(select(PostVector).where(PostVector.post_id == post_id)).first()
    if not post_vector:
        raise HTTPException(status_code=404, detail="Post has no embedding yet")

    all_image_vectors = session.exec(select(ImageVector)).all()

    scored = []
    for iv in all_image_vectors:
        sim = cosine_similarity(post_vector.embedding, iv.embedding)
        scored.append((sim, iv.image_id))

    scored.sort(reverse=True, key=lambda x: x[0])

    suggestions_out = []
    for sim, image_id in scored[:5]:
        image = session.get(Image, image_id)
        guard_result = check_match(post.title, post.body, image.subject, image.confidence, sim)

        suggestion = Suggestion(
            post_id=post_id,
            image_id=image_id,
            similarity_score=sim,
            guard_result=guard_result["result"],
            guard_reason=guard_result["reason"],
        )
        session.add(suggestion)
        session.commit()
        session.refresh(suggestion)

        suggestions_out.append({
            "suggestion_id": suggestion.id,
            "image_id": image_id,
            "file_path": image.file_path,
            "subject": image.subject,
            "similarity": round(sim, 3),
            "guard_result": guard_result["result"],
            "guard_reason": guard_result["reason"],
        })

    session.close()

    approved = [s for s in suggestions_out if s["guard_result"] == "approved"]
    if not approved:
        return {"post_id": post_id, "match": None, "message": "No confident match found", "candidates": suggestions_out}

    return {"post_id": post_id, "match": approved[0], "candidates": suggestions_out}


class ReviewIn(BaseModel):
    decision: str


@app.post("/suggestions/{suggestion_id}/review")
def review_suggestion(suggestion_id: int, payload: ReviewIn):
    session = get_session()
    suggestion = session.get(Suggestion, suggestion_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail=f"Suggestion {suggestion_id} not found")

    if payload.decision not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="decision must be 'approved' or 'rejected'")

    review = Review(suggestion_id=suggestion_id, decision=payload.decision)
    session.add(review)
    session.commit()
    session.refresh(review)
    session.close()

    return {"suggestion_id": suggestion_id, "decision": payload.decision}


@app.get("/suggestions/{suggestion_id}")
def get_suggestion(suggestion_id: int):
    session = get_session()
    suggestion = session.get(Suggestion, suggestion_id)
    if not suggestion:
        raise HTTPException(status_code=404, detail=f"Suggestion {suggestion_id} not found")

    result = {
        "id": suggestion.id,
        "post_id": suggestion.post_id,
        "image_id": suggestion.image_id,
        "similarity_score": suggestion.similarity_score,
        "guard_result": suggestion.guard_result,
        "guard_reason": suggestion.guard_reason,
        "created_at": suggestion.created_at,
    }
    session.close()
    return result
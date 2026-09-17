import os
import io
import base64
import json
import re
import time
import requests
from PIL import Image

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
VISION_MODEL = os.getenv("VISION_MODEL", "llama-3.2-11b-vision-preview")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-minilm")

MAX_DIM = 1024
JPEG_QUALITY = 85
MAX_B64_BYTES = 3_500_000
MAX_RETRIES = 4

VISION_PROMPT = """You are analyzing a photo for a content-matching system. Describe it as JSON only, with exactly these fields:

{
  "subject": "short noun phrase, e.g. 'red fox'",
  "category": "animal" or "other",
  "attributes": ["3 to 6 short descriptive tags"],
  "caption": "one sentence describing the image, max 200 characters",
  "confidence": a number from 0.0 to 1.0 for how confident you are in this identification
}

Return only the JSON object, nothing else."""


def extract_json(text: str) -> dict:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        text = brace.group(0)
    return json.loads(text)


def prepare_image_b64(image_path: str, max_dim: int = MAX_DIM, quality: int = JPEG_QUALITY) -> str:
    img = Image.open(image_path)
    img = img.convert("RGB")
    img.thumbnail((max_dim, max_dim))

    while True:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        if len(b64) <= MAX_B64_BYTES or (max_dim <= 256 and quality <= 40):
            return b64

        if quality > 40:
            quality -= 15
        else:
            max_dim = int(max_dim * 0.75)
            img.thumbnail((max_dim, max_dim))


def tag_image(image_path: str) -> dict:
    image_b64 = prepare_image_b64(image_path)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0.2,
    }

    attempt = 0
    while True:
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        print("GROQ RESPONSE:", response.status_code, response.text)

        if response.status_code == 429 and attempt < MAX_RETRIES:
            wait = 5.0
            try:
                body = response.json()
                msg = body.get("error", {}).get("message", "")
                match = re.search(r"try again in ([\d.]+)s", msg)
                if match:
                    wait = float(match.group(1)) + 1.0
            except (ValueError, json.JSONDecodeError):
                pass
            attempt += 1
            time.sleep(wait)
            continue

        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            return {
                "subject": "unknown",
                "category": "other",
                "attributes": [],
                "caption": f"Vision request failed (HTTP {status})",
                "confidence": 0.0,
                "error": str(e),
            }
        break

    raw = response.json()["choices"][0]["message"]["content"]

    try:
        data = extract_json(raw)
    except (json.JSONDecodeError, AttributeError):
        data = {
            "subject": "unknown",
            "category": "other",
            "attributes": [],
            "caption": raw[:200] if raw else "No caption available",
            "confidence": 0.0,
        }

    data.setdefault("subject", "unknown")
    data.setdefault("category", "other")
    data.setdefault("attributes", [])
    data.setdefault("caption", "")
    data.setdefault("confidence", 0.0)

    return data


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
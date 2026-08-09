"""
OPTIONAL corpus expansion: generates diverse English sentences with any
OpenAI-compatible chat endpoint, to broaden vocabulary and topic coverage
beyond the fixed sentences and the CMU ARCTIC prompts.

    export LLM_BASE_URL=http://localhost:8000/v1
    export LLM_MODEL=<model-name>
    export _cfg()[2]=<key or "none">

**This step is entirely optional.** Stages 1 and 2 already yield a complete,
phonetically balanced corpus with no external service, and that is what
4_build_corpus.py falls back to when no endpoint is configured. It exists
because more topical variety improves a distilled voice, not because the
pipeline needs it.
"""
import json
import os
import re
import time
from pathlib import Path

# 'requests' is imported INSIDE _call_llm, not here: 4_build_corpus.py imports
# this module unconditionally and must work on a machine that has neither an
# LLM nor requests installed.

# Read lazily, NOT at import time: this module is imported unconditionally by
# 4_build_corpus.py, which must still work on a machine with no LLM at all.
LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))


def available() -> bool:
    """True when an OpenAI-compatible endpoint is configured."""
    return bool(os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_MODEL"))


def _cfg() -> tuple[str, str, str]:
    if not available():
        raise RuntimeError(
            "No LLM configured. Set LLM_BASE_URL and LLM_MODEL, or skip this "
            "step -- the corpus is complete without it.")
    return (os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"],
            os.environ.get("_cfg()[2]", "none"))

SENTENCES_PER_CALL = 25

TOPICS = [
    "everyday small talk", "weather", "cooking and recipes", "technology",
    "travel", "sports", "science facts", "business meetings", "family life",
    "emotions and feelings", "giving directions", "shopping", "health and fitness",
    "nature and animals", "music and movies", "history", "customer service calls",
    "asking questions", "giving commands or instructions", "storytelling",
    "phone conversations", "scheduling and calendars", "hobbies", "education",
    "news headlines read aloud", "casual greetings and goodbyes",
]

PROMPT_TEMPLATE = """Write {n} short, natural English sentences about {topic}.

Rules:
- One sentence per line, no numbering, no bullets, no quotes.
- Vary sentence length: some short (4-6 words), some longer (12-20 words).
- Mix statements, questions, and exclamations.
- Plain conversational English, no markdown.
- Each sentence must end with . ? or !
"""


def _clean_line(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
    line = re.sub(r"^[-*•]\s*", "", line)
    line = line.strip(" \"'")
    return line.strip()


def _call_llm(topic: str, n: int) -> list[str]:
    import requests
    resp = requests.post(
        f"{_cfg()[0]}/chat/completions",
        headers={"Authorization": f"Bearer {_cfg()[2]}"},
        json={
            "model": _cfg()[1],
            "messages": [
                {"role": "user", "content": PROMPT_TEMPLATE.format(n=n, topic=topic)}
            ],
            "max_tokens": 800,
            "temperature": 0.9,
        },
        timeout=LLM_TIMEOUT,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    lines = [_clean_line(l) for l in content.splitlines()]
    return [l for l in lines if l and l[-1] in ".?!" and 2 <= len(l.split()) <= 30]


def _flushed_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)


def generate_sentences(target_count: int, cache_path: Path, log=_flushed_print) -> list[str]:
    """Generate up to target_count unique sentences, resuming from cache_path if present."""
    seen: set[str] = set()
    sentences: list[str] = []

    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                s = json.loads(line)["text"]
                key = s.strip().lower()
                if key not in seen:
                    seen.add(key)
                    sentences.append(s)
        log(f"[resume] loaded {len(sentences)} cached sentences from {cache_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    out = open(cache_path, "a")

    topic_i = 0
    attempts_without_progress = 0
    while len(sentences) < target_count and attempts_without_progress < 40:
        topic = TOPICS[topic_i % len(TOPICS)]
        topic_i += 1
        try:
            batch = _call_llm(topic, SENTENCES_PER_CALL)
        except Exception as e:
            log(f"[warn] LLM call failed for topic={topic!r}: {e}")
            time.sleep(2)
            attempts_without_progress += 1
            continue

        added = 0
        for s in batch:
            key = s.strip().lower()
            if key not in seen:
                seen.add(key)
                sentences.append(s)
                out.write(json.dumps({"text": s, "topic": topic}) + "\n")
                added += 1

        out.flush()
        attempts_without_progress = 0 if added else attempts_without_progress + 1
        log(f"[llm] topic={topic!r} +{added} new (total {len(sentences)}/{target_count})")

    out.close()
    return sentences[:target_count]


if __name__ == "__main__":
    cache = Path(__file__).resolve().parent / "output" / "llm_sentences_cache.jsonl"
    result = generate_sentences(target_count=60, cache_path=cache)
    print(f"\nGenerated {len(result)} sentences (smoke test). Sample:")
    for s in result[:10]:
        print(" -", s)

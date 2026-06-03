from __future__ import annotations

import hashlib
import os
import random
import re
import time
from dataclasses import dataclass

import requests


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def _sentences(text: str, max_sentences: int = 8) -> list[str]:
    parts = [p.strip() for p in SENTENCE_RE.split(text.replace("\n", " ")) if p.strip()]
    return parts[:max_sentences] or [text[:500].strip()]


def _keywords(text: str, n: int = 10) -> list[str]:
    stop = {
        "the", "and", "that", "with", "from", "this", "have", "were", "they", "there",
        "their", "what", "when", "which", "would", "could", "should", "into", "upon",
        "then", "than", "them", "your", "you", "for", "but", "not", "his", "her",
        "she", "him", "our", "are", "was", "had", "has", "who", "all", "one",
    }
    counts: dict[str, int] = {}
    for word in WORD_RE.findall(text.lower()):
        if len(word) >= 5 and word not in stop:
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts, key=lambda w: (-counts[w], w))
    return ranked[:n]


def _clean_source(text: str, max_chars: int = 1200) -> str:
    text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "."


def _soft_rewrite_sentence(sentence: str, rng: random.Random) -> str:
    replacements = [
        (r"\bsaid\b", ["remarked", "answered", "noted"]),
        (r"\bwent\b", ["moved", "made his way", "passed"]),
        (r"\bcame\b", ["arrived", "appeared", "came in"]),
        (r"\bgreat\b", ["large", "strong", "striking"]),
        (r"\blittle\b", ["small", "slight", "modest"]),
        (r"\blooked\b", ["seemed", "appeared", "looked"]),
        (r"\bthought\b", ["felt", "supposed", "imagined"]),
        (r"\bvery\b", ["quite", "rather", ""]),
        (r"\bhowever\b", ["still", "yet", "even so"]),
        (r"\btherefore\b", ["so", "for that reason"]),
    ]
    out = sentence
    for pattern, choices in replacements:
        if rng.random() < 0.35:
            out = re.sub(pattern, rng.choice(choices), out, flags=re.I)
    if rng.random() < 0.22 and "," in out:
        parts = out.split(",", 1)
        out = f"{parts[1].strip()}, {parts[0].strip().lower()}"
    if rng.random() < 0.18:
        out = out.replace("; ", ", ")
    return re.sub(r"\s+", " ", out).strip()


def demo_generate(source_text: str, *, variant: str, seed: int, subset: str = "test") -> str:
    """A deterministic workflow fallback, not a substitute for real GAI.

    Train and test intentionally use different local rewriting styles. This makes
    the no-API experiment less vulnerable to a single obvious template fingerprint.
    """
    rng = random.Random(seed)
    sents = _sentences(source_text, max_sentences=10)
    kws = _keywords(source_text)

    formal_train_style = subset == "train" and rng.random() < 0.45

    if formal_train_style:
        opening = rng.choice(
            [
                "The passage can be read as a compact study of pressure, motive, and response.",
                "This rewritten account presents the scene in a clearer and more explanatory form.",
                "The text describes a moment in which action and reflection are closely connected.",
                "The episode is shaped by tension between public behavior and private judgment.",
            ]
        )
        focus = ", ".join(kws[:5]) if kws else "character, setting, conflict, memory, and change"
        body_seed = " ".join(sents[:3])
        body_seed = _clean_source(body_seed, max_chars=520)
        paragraphs = [
            opening,
            (
                f"Details around {focus} give the passage its direction. This version restates "
                f"the central movement without preserving every phrase: {body_seed}"
            ),
            (
                "The result is more regular in structure than the source. Events are connected "
                "through explicit explanation, and the likely emotional or narrative purpose is "
                "brought closer to the surface."
            ),
        ]
    else:
        selected = sents[:]
        rng.shuffle(selected)
        selected = selected[: rng.randint(4, min(7, max(4, len(selected))))]
        rewritten = [_soft_rewrite_sentence(sent, rng) for sent in selected]
        if variant == "plain":
            connectors = ["At first", "After that", "For a moment", "By then", "In the end"]
        else:
            connectors = ["Anyway", "Then again", "Still", "A little later", "By that point"]
        paragraphs = []
        for idx, sent in enumerate(rewritten):
            if idx == 0 and rng.random() < 0.5:
                paragraphs.append(sent)
            elif rng.random() < 0.42:
                paragraphs.append(f"{rng.choice(connectors)}, {sent[0].lower() + sent[1:]}")
            else:
                paragraphs.append(sent)
        if kws and rng.random() < 0.35:
            paragraphs.append(
                f"What stays with the scene is not one event only, but the pressure around {', '.join(kws[:3])}."
            )
        generated_trace_rate = 0.65 if subset == "train" else (0.55 if variant == "plain" else 0.38)
        if rng.random() < generated_trace_rate:
            paragraphs.append(
                rng.choice(
                    [
                        "In this rewritten version, the main tension is made more explicit than in the source.",
                        "The passage is therefore smoothed into a clearer sequence of action and response.",
                        "The emphasis falls on the larger meaning of the scene rather than on every original detail.",
                        "Overall, the retelling keeps the situation intact while making the transitions easier to follow.",
                    ]
                )
            )
        if len(" ".join(paragraphs)) < 650:
            paragraphs.append(_clean_source(source_text, max_chars=420))

    if variant == "obfuscated":
        if subset == "train":
            paragraphs.append(
                "The wording is also less uniform, with shorter turns and plainer transitions."
            )
        else:
            insert_at = rng.randrange(0, len(paragraphs) + 1)
            paragraphs.insert(insert_at, rng.choice(["That is the rough shape of it.", "It feels uneven, but deliberately so."]))
    return " ".join(paragraphs)


@dataclass
class OpenAIConfig:
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1/chat/completions"
    temperature: float = 0.75
    max_tokens: int = 420
    sleep_seconds: float = 0.2


@dataclass
class OllamaConfig:
    model: str = "qwen2.5:7b-instruct"
    base_url: str = "http://localhost:11434/api/generate"
    temperature: float = 0.75
    max_tokens: int = 260
    timeout_seconds: int = 240


def _rewrite_instruction(variant: str) -> str:
    if variant == "plain":
        return (
            "Rewrite the passage as a new English text of roughly 140-210 words. "
            "Preserve the core situation, topic, and meaning, but do not copy sentences. "
            "Do not mention AI, prompts, rewriting, or the source passage."
        )
    return (
        "Rewrite the passage as a new English text of roughly 140-210 words. "
        "Preserve the core situation and meaning, but make the style less detectable as "
        "machine-written by using varied sentence lengths, occasional informal transitions, "
        "less symmetrical paragraphing, and natural minor imperfections. Do not use hidden "
        "characters, homoglyphs, or unreadable obfuscation. Do not mention AI, prompts, "
        "rewriting, or the source passage."
    )


def openai_generate(source_text: str, *, variant: str, config: OpenAIConfig) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    clipped = source_text[:3500]
    instruction = _rewrite_instruction(variant)

    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "messages": [
            {"role": "system", "content": "You are a careful literary rewriter."},
            {"role": "user", "content": f"{instruction}\n\nSOURCE PASSAGE:\n{clipped}"},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    response = requests.post(config.base_url, json=payload, headers=headers, timeout=90)
    if response.status_code >= 400:
        raise RuntimeError(f"OpenAI-compatible API error {response.status_code}: {response.text[:500]}")
    data = response.json()
    time.sleep(config.sleep_seconds)
    return data["choices"][0]["message"]["content"].strip()


def ollama_generate(source_text: str, *, variant: str, config: OllamaConfig, seed: int) -> str:
    clipped = _clean_source(source_text, max_chars=2600)
    prompt = (
        f"{_rewrite_instruction(variant)}\n\n"
        "Return only the rewritten English text.\n\n"
        f"SOURCE PASSAGE:\n{clipped}"
    )
    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": config.temperature,
            "num_predict": config.max_tokens,
            "seed": int(seed % (2**31 - 1)),
        },
    }
    response = requests.post(config.base_url, json=payload, timeout=config.timeout_seconds)
    if response.status_code >= 400:
        raise RuntimeError(f"Ollama API error {response.status_code}: {response.text[:500]}")
    data = response.json()
    text = str(data.get("response", "")).strip()
    if not text:
        raise RuntimeError("Ollama returned an empty response.")
    return re.sub(r"\s+", " ", text).strip()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)

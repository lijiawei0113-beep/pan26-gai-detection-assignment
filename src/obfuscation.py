from __future__ import annotations

import random
import re


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

SUBSTITUTIONS = {
    "therefore": ["so", "for that reason"],
    "however": ["still", "even so"],
    "important": ["notable", "worth noticing"],
    "suggests": ["hints", "seems to show"],
    "describes": ["sets out", "shows"],
    "because": ["since", "as"],
    "although": ["though"],
    "cannot": ["can't"],
    "do not": ["don't"],
    "it is": ["it's"],
    "the result": ["what comes out"],
}

FILLERS = ["in a way", "more or less", "at moments", "to my ear", "plainly enough"]


def obfuscate_surface(text: str, seed: int) -> str:
    rng = random.Random(seed)
    out = text
    for src, replacements in SUBSTITUTIONS.items():
        if rng.random() < 0.65:
            out = re.sub(rf"\b{re.escape(src)}\b", rng.choice(replacements), out, flags=re.I)

    sentences = [s.strip() for s in SENTENCE_RE.split(out) if s.strip()]
    for i in range(0, len(sentences) - 1, 3):
        if rng.random() < 0.35:
            sentences[i], sentences[i + 1] = sentences[i + 1], sentences[i]

    edited = []
    for sent in sentences:
        if rng.random() < 0.22 and len(sent.split()) > 8:
            sent = f"{rng.choice(FILLERS)}, {sent[0].lower() + sent[1:]}"
        if rng.random() < 0.18:
            sent = sent.replace("; ", ", ")
        if rng.random() < 0.12:
            sent = sent.replace(" and ", " and, ", 1)
        edited.append(sent)
    out = " ".join(edited)
    out = re.sub(r"\s+", " ", out).strip()
    return out

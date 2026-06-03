from __future__ import annotations

import bz2
import math
import random
import zlib
from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def fit_tfidf_svm(texts: list[str], labels: list[int]) -> Pipeline:
    base = LinearSVC(class_weight="balanced", C=1.0, random_state=26)
    clf = CalibratedClassifierCV(base, cv=3)
    pipe = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=200_000,
                    lowercase=True,
                ),
            ),
            ("clf", clf),
        ]
    )
    return pipe.fit(texts, labels)


def predict_tfidf_svm(model: Pipeline, texts: list[str]) -> list[float]:
    return model.predict_proba(texts)[:, 1].astype(float).tolist()


def _compress_len(text: str, compressor: str = "zlib") -> int:
    data = text.encode("utf-8", errors="ignore")
    if compressor == "bz2":
        return len(bz2.compress(data, compresslevel=9))
    return len(zlib.compress(data, level=9))


def _ncd(x: str, y: str, cx: int, cy: int, compressor: str) -> float:
    cxy = _compress_len(x + "\n" + y, compressor)
    return (cxy - min(cx, cy)) / max(cx, cy)


@dataclass
class CompressionKNN:
    refs_per_class: int = 80
    k: int = 9
    compressor: str = "zlib"
    seed: int = 26

    def fit(self, texts: list[str], labels: list[int]) -> "CompressionKNN":
        rng = random.Random(self.seed)
        self.refs: list[tuple[str, int, int]] = []
        for label in [0, 1]:
            items = [t for t, y in zip(texts, labels) if y == label]
            rng.shuffle(items)
            for text in items[: self.refs_per_class]:
                self.refs.append((text, label, _compress_len(text, self.compressor)))
        return self

    def predict_proba(self, texts: list[str]) -> list[float]:
        probs = []
        for text in texts:
            cx = _compress_len(text, self.compressor)
            distances = [
                (_ncd(text, ref, cx, cref, self.compressor), label)
                for ref, label, cref in self.refs
            ]
            distances.sort(key=lambda item: item[0])
            neighbors = distances[: self.k]
            weights = []
            ai_weight = 0.0
            for dist, label in neighbors:
                weight = 1.0 / max(dist, 1e-6)
                weights.append(weight)
                if label == 1:
                    ai_weight += weight
            prob = ai_weight / max(sum(weights), 1e-9)
            probs.append(float(np.clip(prob, 0.0, 1.0)))
        return probs

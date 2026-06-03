from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


def _binary(scores: Iterable[float]) -> np.ndarray:
    return np.array([1 if float(s) > 0.5 else 0 for s in scores], dtype=int)


def confusion_matrix(y_true: Iterable[int], scores: Iterable[float]) -> list[list[int]]:
    y = np.array(list(y_true), dtype=int)
    pred = _binary(scores)
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    return [[tn, fp], [fn, tp]]


def c_at_1(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.array(list(y_true), dtype=int)
    s = np.array(list(scores), dtype=float)
    answered = s != 0.5
    pred = _binary(s)
    n = len(y)
    nc = int(((pred == y) & answered).sum())
    nu = int((~answered).sum())
    if n == 0:
        return 0.0
    return (nc + nu * (nc / n)) / n


def f1_score_pos(y_true: Iterable[int], scores: Iterable[float]) -> float:
    cm = confusion_matrix(y_true, scores)
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def f05u_score(y_true: Iterable[int], scores: Iterable[float]) -> float:
    y = np.array(list(y_true), dtype=int)
    s = np.array(list(scores), dtype=float)
    pred = _binary(s)
    tp = int(((y == 1) & (pred == 1) & (s != 0.5)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()) + int((s == 0.5).sum())
    beta2 = 0.5 * 0.5
    denom = (1 + beta2) * tp + beta2 * fn + fp
    return 0.0 if denom == 0 else ((1 + beta2) * tp) / denom


def evaluate_scores(y_true: list[int], scores: list[float]) -> dict:
    y = np.array(y_true, dtype=int)
    s = np.clip(np.array(scores, dtype=float), 0.0, 1.0)
    try:
        auc = float(roc_auc_score(y, s))
    except ValueError:
        auc = float("nan")
    brier = 1.0 - float(np.mean((s - y) ** 2))
    c1 = c_at_1(y, s)
    f1 = f1_score_pos(y, s)
    f05u = f05u_score(y, s)
    vals = [v for v in [auc, brier, c1, f1, f05u] if not math.isnan(v)]
    return {
        "roc-auc": auc,
        "brier": brier,
        "c@1": c1,
        "f1": f1,
        "f05u": f05u,
        "mean": float(np.mean(vals)),
        "confusion": confusion_matrix(y, s),
    }

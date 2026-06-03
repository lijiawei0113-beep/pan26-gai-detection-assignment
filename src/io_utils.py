from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def read_human_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    text_col = "text" if "text" in df.columns else "0"
    if text_col not in df.columns:
        raise ValueError(f"Could not find a text column in {path}; columns={list(df.columns)}")
    out = pd.DataFrame(
        {
            "source_id": df.index.astype(str),
            "text": df[text_col].astype(str).str.strip(),
        }
    )
    out = out[out["text"].str.len() > 0].reset_index(drop=True)
    return out


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

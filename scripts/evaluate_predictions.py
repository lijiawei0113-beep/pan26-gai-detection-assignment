from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.io_utils import read_jsonl
from src.metrics import evaluate_scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="JSONL with id and label")
    parser.add_argument("--pred", required=True, help="PAN-style JSONL with id and label score")
    args = parser.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_jsonl(args.pred)
    gold = {r["id"]: int(r["label"]) for r in gold_rows}
    pred = {r["id"]: float(r["label"]) for r in pred_rows}
    missing = sorted(set(gold) - set(pred))
    if missing:
        raise SystemExit(f"Missing predictions for {len(missing)} ids, e.g. {missing[:3]}")
    ids = [r["id"] for r in gold_rows]
    metrics = evaluate_scores([gold[i] for i in ids], [pred[i] for i in ids])
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generation import (
    OllamaConfig,
    OpenAIConfig,
    demo_generate,
    ollama_generate,
    openai_generate,
    stable_seed,
)
from src.io_utils import read_human_csv, write_csv, write_jsonl
from src.metrics import evaluate_scores
from src.models import CompressionKNN, fit_tfidf_svm, predict_tfidf_svm
from src.obfuscation import obfuscate_surface


def make_rows(
    humans: pd.DataFrame,
    *,
    subset: str,
    provider: str,
    model: str,
    include_human: bool,
    include_plain: bool,
    include_obfuscated: bool,
) -> list[dict]:
    rows: list[dict] = []
    openai_cfg = OpenAIConfig(model=model)
    ollama_cfg = OllamaConfig(model=model)
    for i, record in humans.reset_index(drop=True).iterrows():
        source_id = str(record["source_id"])
        source_text = str(record["text"])
        if include_human:
            rows.append(
                {
                    "id": f"{subset}-human-{i:04d}",
                    "text": source_text,
                    "label": 0,
                    "source_id": source_id,
                    "subset": subset,
                    "kind": "human",
                    "generator": "human",
                    "obfuscation": "none",
                }
            )

        plain = None
        if include_plain or include_obfuscated:
            if provider == "openai":
                plain = openai_generate(source_text, variant="plain", config=openai_cfg)
            elif provider == "ollama":
                plain = ollama_generate(
                    source_text,
                    variant="plain",
                    config=ollama_cfg,
                    seed=stable_seed(subset, source_id, "plain"),
                )
            else:
                plain = demo_generate(
                    source_text,
                    variant="plain",
                    seed=stable_seed(subset, source_id, "plain"),
                    subset=subset,
                )
        if include_plain:
            rows.append(
                {
                    "id": f"{subset}-gai-{i:04d}",
                    "text": plain,
                    "label": 1,
                    "source_id": source_id,
                    "subset": subset,
                    "kind": "gai_plain",
                    "generator": provider,
                    "obfuscation": "none",
                }
            )
        if include_obfuscated:
            if provider == "openai":
                obf = openai_generate(source_text, variant="obfuscated", config=openai_cfg)
                obf = obfuscate_surface(obf, seed=stable_seed(subset, source_id, "obf_surface"))
            elif provider == "ollama":
                obf = ollama_generate(
                    source_text,
                    variant="obfuscated",
                    config=ollama_cfg,
                    seed=stable_seed(subset, source_id, "obfuscated"),
                )
                obf = obfuscate_surface(obf, seed=stable_seed(subset, source_id, "obf_surface"))
            else:
                obf = demo_generate(
                    source_text,
                    variant="obfuscated",
                    seed=stable_seed(subset, source_id, "obfuscated"),
                    subset=subset,
                )
                obf = obfuscate_surface(obf, seed=stable_seed(subset, source_id, "obf_surface"))
            rows.append(
                {
                    "id": f"{subset}-gai-obf-{i:04d}",
                    "text": obf,
                    "label": 1,
                    "source_id": source_id,
                    "subset": subset,
                    "kind": "gai_obfuscated",
                    "generator": provider,
                    "obfuscation": "prompt_plus_surface",
                }
            )
    return rows


def write_pan_input(rows: list[dict], path: Path) -> None:
    write_jsonl([{"id": r["id"], "text": r["text"]} for r in rows], path)


def evaluate_model(name: str, scores: list[float], test_rows: list[dict], out_dir: Path) -> dict:
    pred_path = out_dir / f"predictions_{name}.jsonl"
    write_jsonl(
        [{"id": row["id"], "label": float(score)} for row, score in zip(test_rows, scores)],
        pred_path,
    )
    return evaluate_scores([int(r["label"]) for r in test_rows], scores)


def prepare_output_dir(out_dir: Path) -> Path:
    resolved = out_dir.resolve()
    outputs_root = (ROOT / "outputs").resolve()
    if resolved == outputs_root or not resolved.is_relative_to(outputs_root):
        raise ValueError(f"--out-dir must be a subdirectory of {outputs_root}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-csv", required=True, help="Path to final_dataset_human.csv")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "latest"))
    parser.add_argument("--provider", choices=["demo", "openai", "ollama"], default="demo")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--limit", type=int, default=200, help="Number of held-out human instances")
    parser.add_argument("--train-human", type=int, default=600)
    parser.add_argument("--seed", type=int, default=26)
    args = parser.parse_args()

    out_dir = prepare_output_dir(Path(args.out_dir))

    humans = read_human_csv(args.human_csv)
    sampled = humans.sample(n=args.limit + args.train_human, random_state=args.seed).reset_index(drop=True)
    test_humans = sampled.iloc[: args.limit].copy()
    train_humans = sampled.iloc[args.limit : args.limit + args.train_human].copy()

    test_rows = make_rows(
        test_humans,
        subset="test",
        provider=args.provider,
        model=args.model,
        include_human=True,
        include_plain=True,
        include_obfuscated=True,
    )
    train_rows = make_rows(
        train_humans,
        subset="train",
        provider=args.provider,
        model=args.model,
        include_human=True,
        include_plain=True,
        include_obfuscated=True,
    )

    write_csv(test_humans.to_dict(orient="records"), out_dir / "selected_human_200.csv")
    write_jsonl(test_rows, out_dir / "test_dataset_600.jsonl")
    write_jsonl(train_rows, out_dir / "train_dataset.jsonl")
    write_pan_input(test_rows, out_dir / "pan_input_test_600.jsonl")

    train_texts = [r["text"] for r in train_rows]
    train_labels = [int(r["label"]) for r in train_rows]
    test_texts = [r["text"] for r in test_rows]

    tfidf = fit_tfidf_svm(train_texts, train_labels)
    tfidf_scores = predict_tfidf_svm(tfidf, test_texts)

    comp = CompressionKNN(refs_per_class=80, k=9, compressor="zlib", seed=args.seed).fit(
        train_texts, train_labels
    )
    comp_scores = comp.predict_proba(test_texts)

    metrics = {
        "metadata": {
            "provider": args.provider,
            "model": args.model if args.provider in {"openai", "ollama"} else "demo-deterministic-fallback",
            "seed": args.seed,
            "test_rows": len(test_rows),
            "train_rows": len(train_rows),
            "note": (
                "Rerun with --provider openai or --provider ollama for final GAI-generated results."
                if args.provider == "demo"
                else "Generated with a real GAI provider."
            ),
        },
        "tfidf_svm": evaluate_model("tfidf_svm", tfidf_scores, test_rows, out_dir),
        "compression_knn": evaluate_model("compression_knn", comp_scores, test_rows, out_dir),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    summary = []
    for model_name in ["tfidf_svm", "compression_knn"]:
        row = {"model": model_name}
        row.update({k: v for k, v in metrics[model_name].items() if k != "confusion"})
        row["confusion"] = json.dumps(metrics[model_name]["confusion"])
        summary.append(row)
    write_csv(summary, out_dir / "results_summary.csv")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

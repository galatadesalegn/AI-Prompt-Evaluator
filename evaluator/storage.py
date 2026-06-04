"""
storage.py — Save, load, and manage evaluation results in JSONL format.
"""

import json
from pathlib import Path
from evaluator.evaluator import EvalResult


DEFAULT_PATH = "data/evaluations.jsonl"


def save(result: EvalResult, path: str = DEFAULT_PATH) -> None:
    """Append one evaluation result to the JSONL store."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def load_all(path: str = DEFAULT_PATH) -> list[dict]:
    """Load all stored evaluation results."""
    p = Path(path)
    if not p.exists():
        return []
    results = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            results.append(json.loads(line))
    return results


def export_json(
    input_path: str = DEFAULT_PATH,
    output_path: str = "reports/evaluations.json",
) -> Path:
    """Export JSONL → pretty JSON array."""
    entries = load_all(input_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {len(entries)} evaluations → {out}")
    return out


def export_csv(
    input_path: str = DEFAULT_PATH,
    output_path: str = "reports/evaluations.csv",
) -> Path:
    """Export evaluations to CSV — one row per evaluation."""
    import csv
    entries = load_all(input_path)
    if not entries:
        print("No data to export.")
        return Path(output_path)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "eval_id", "model_name", "evaluator_type", "overall_score",
        "quality_tier", "flags", "helpfulness", "clarity",
        "accuracy", "safety", "tone", "timestamp",
    ]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            score_map = {s["criterion"]: s["score"] for s in e.get("scores", [])}
            writer.writerow({
                "eval_id": e["eval_id"],
                "model_name": e["model_name"],
                "evaluator_type": e["evaluator_type"],
                "overall_score": e["overall_score"],
                "quality_tier": e["quality_tier"],
                "flags": "|".join(e.get("flags", [])),
                "helpfulness": score_map.get("helpfulness", ""),
                "clarity": score_map.get("clarity", ""),
                "accuracy": score_map.get("accuracy", ""),
                "safety": score_map.get("safety", ""),
                "tone": score_map.get("tone", ""),
                "timestamp": e["timestamp"],
            })

    print(f"Exported {len(entries)} evaluations → {out}")
    return out

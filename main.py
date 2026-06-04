#!/usr/bin/env python3
"""
main.py — CLI entry point for the AI Prompt Evaluator.

Usage:
  python main.py                          # heuristic eval on sample data
  python main.py --mode human             # interactive human annotation
  python main.py --data data/sample_responses.json
  python main.py --report                 # print summary report
  python main.py --export csv             # export to CSV
  python main.py --export json            # export to JSON
  python main.py --model gpt-4           # tag results with model name
"""

import argparse
import json
import sys
from pathlib import Path

from evaluator.evaluator import PromptEvaluator
from evaluator.storage import save, export_csv, export_json
from evaluator.report import print_report

SAMPLE_DATA = "data/sample_responses.json"
STORE_PATH = "data/evaluations.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🤖 AI Prompt Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["heuristic", "human"], default="heuristic",
                        help="Evaluation mode")
    parser.add_argument("--data", type=str, default=SAMPLE_DATA,
                        help="Path to JSON file with prompt/response pairs")
    parser.add_argument("--model", type=str, default="unknown-model",
                        help="Model name to tag evaluations with")
    parser.add_argument("--output", type=str, default=STORE_PATH,
                        help="JSONL output path")
    parser.add_argument("--report", action="store_true",
                        help="Print summary report of saved evaluations")
    parser.add_argument("--export", choices=["csv", "json"],
                        help="Export evaluations to CSV or JSON")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of responses to evaluate")
    return parser.parse_args()


def load_data(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    args = parse_args()

    if args.report:
        print_report(args.output)
        return 0

    if args.export == "csv":
        export_csv(args.output)
        return 0

    if args.export == "json":
        export_json(args.output)
        return 0

    data = load_data(args.data)
    if args.limit:
        data = data[:args.limit]

    print(f"\n🤖  AI Prompt Evaluator")
    print(f"    Mode    : {args.mode}")
    print(f"    Model   : {args.model}")
    print(f"    Pairs   : {len(data)}")
    print(f"    Output  : {args.output}\n")

    evaluator = PromptEvaluator(model_name=args.model)
    results = []

    for i, item in enumerate(data, 1):
        prompt = item["prompt"]
        response = item["response"]
        model = item.get("model", args.model)

        print(f"{'═'*65}")
        print(f"  [{i}/{len(data)}] Model: {model}")
        print(f"{'═'*65}")

        evaluator.model_name = model

        if args.mode == "heuristic":
            result = evaluator.evaluate_heuristic(prompt, response)
            print(f"  Prompt   : {prompt[:70]}...")
            print(f"  Score    : {result.overall_score} / 5.00  ({result.quality_tier})")
            print(f"  Flags    : {', '.join(result.flags) or 'none'}")
            print()
        else:
            result = evaluator.evaluate_human(prompt, response)
            if i < len(data):
                cont = input("  Continue to next? [Y/n]: ").strip().lower()
                if cont == "n":
                    save(result, args.output)
                    results.append(result)
                    break

        save(result, args.output)
        results.append(result)

    avg = sum(r.overall_score for r in results) / len(results) if results else 0
    print(f"\n✅  Done! {len(results)} evaluation(s) saved.")
    print(f"    Average score : {avg:.2f} / 5.00")
    print(f"    File          : {args.output}")
    print(f"\nRun `python main.py --report` to see the full summary.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

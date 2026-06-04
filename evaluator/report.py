"""
report.py — Generate summary reports from collected evaluation data.
"""

from collections import Counter
from evaluator.storage import load_all


def print_report(path: str = "data/evaluations.jsonl") -> None:
    entries = load_all(path)

    if not entries:
        print("\nNo evaluation data found. Run some evaluations first.\n")
        return

    total = len(entries)
    tiers = Counter(e["quality_tier"] for e in entries)
    models = Counter(e["model_name"] for e in entries)
    types = Counter(e["evaluator_type"] for e in entries)
    all_flags = [f for e in entries for f in e.get("flags", [])]
    flag_counts = Counter(all_flags)
    scores = [e["overall_score"] for e in entries]
    avg = sum(scores) / len(scores)

    div = "─" * 55
    print(f"\n{div}")
    print("  AI PROMPT EVALUATOR — REPORT")
    print(div)
    print(f"  Total evaluations  : {total}")
    print(f"  Average score      : {avg:.2f} / 5.00")
    print(f"  Evaluator types    : {dict(types)}")
    print()
    print("  Quality distribution:")
    for tier in ["Excellent", "Good", "Fair", "Poor", "Unacceptable"]:
        count = tiers.get(tier, 0)
        bar = "█" * count
        print(f"    {tier:<14} {bar} {count}")
    print()
    print("  Models evaluated:")
    for model, count in models.most_common():
        print(f"    {model:<25} {count} eval(s)")
    print()
    if flag_counts:
        print("  Flags detected:")
        for flag, count in flag_counts.most_common():
            print(f"    {flag:<25} {count}x")
    print()
    print("  Recent evaluations:")
    for e in entries[-3:]:
        print(f"    [{e['eval_id']}] {e['model_name']} | "
              f"Score: {e['overall_score']} | {e['quality_tier']} | "
              f"Flags: {', '.join(e['flags']) or 'none'}")
    print(div)

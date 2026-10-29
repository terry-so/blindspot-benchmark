"""Estimate image-generation costs from saved API token usage (not judge costs)."""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def usage_cost(usage, text_rate, image_rate, output_rate):
    details = usage.get("input_tokens_details") or {}
    text = details.get("text_tokens")
    image = details.get("image_tokens")
    output = usage.get("output_tokens")
    if any(type(value) is not int or value < 0 for value in (text, image, output)):
        return None
    return (text * text_rate + image * image_rate + output * output_rate) / 1_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--text-rate", type=float, default=5.0)
    parser.add_argument("--image-rate", type=float, default=8.0)
    parser.add_argument("--output-rate", type=float, default=30.0)
    parser.add_argument("--planned-edits", type=int)
    args = parser.parse_args()
    if min(args.text_rate, args.image_rate, args.output_rate) < 0:
        parser.error("Rates must be nonnegative USD per million tokens")
    if args.planned_edits is not None and args.planned_edits < 1:
        parser.error("planned-edits must be positive")

    groups = defaultdict(lambda: {"recorded_calls": 0, "priced_calls": 0, "usd": 0.0})
    for path in sorted((args.run / "results").glob("*/*.json")):
        if path.name.endswith(".generation.json"):
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        # L3's top-level usage repeats its last step; count only the steps.
        records = result.get("steps", []) if result.get("level") == "L3-reset" else [result]
        group = groups[result["model"]]
        for record in records:
            group["recorded_calls"] += 1
            cost = usage_cost(record.get("usage") or {}, args.text_rate,
                              args.image_rate, args.output_rate)
            if cost is not None:
                group["priced_calls"] += 1
                group["usd"] += cost

    if not groups:
        parser.error("No API result JSON files found under RUN/results/MODEL/")
    print("Estimated USD from logged generation usage; excludes judges and unlogged attempts.")
    print(f"Rates per million tokens: text={args.text_rate}, image={args.image_rate}, output={args.output_rate}")
    for name, group in sorted(groups.items()):
        count = group["priced_calls"]
        print(f"{name}: ${group['usd']:.4f}; priced usage for {count}/{group['recorded_calls']} recorded calls")
        if count:
            average = group["usd"] / count
            print(f"  Mean per usage-reported call: ${average:.6f}")
            if args.planned_edits:
                print(f"  Projection for {args.planned_edits} calls at this mean: ${average * args.planned_edits:.2f}")
        if count != group["recorded_calls"]:
            print("  Missing usage is unknown cost, not zero. Projection may be biased by refusals/failures.")
    print("Confirm actual charges and deployment rates in Azure billing; retries are not fully captured.")


if __name__ == "__main__":
    main()

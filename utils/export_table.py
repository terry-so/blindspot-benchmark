# export_tables.py
import json
from pathlib import Path

import pandas as pd

from metrics.metrics import compute_metrics


LEVELS = ["L1", "L2a", "L2b", "L3-session", "L3-reset"]
DOMAINS = ["D3", "D4"]


def make_tables(result_dir="results", output_dir="tables"):
    rows = [
        json.loads(path.read_text())
        for path in Path(result_dir).glob("*/*.json")
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No results found.")

    summary = []
    for (model, domain, level), group in df.groupby(["model", "domain", "level"]):
        summary.append({
            "model": model,
            "domain": domain,
            "level": level,
            **compute_metrics(group.to_dict("records")),
        })

    summary = pd.DataFrame(summary)
    summary[["RR", "EF", "RS", "HRR"]] *= 100

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_dir / "summary.csv", index=False)

    def value(model, domain, level, metric):
        selected = summary[
            (summary["model"] == model)
            & (summary["domain"] == domain)
            & (summary["level"] == level)
        ]
        return selected.iloc[0][metric] if len(selected) else float("nan")

    table8 = []
    table9 = []

    for model in sorted(df["model"].unique()):
        direct = {"model": model}
        hierarchy = {"model": model}

        for domain in DOMAINS:
            for metric in ("RR", "HRR"):
                direct[f"{domain}_{metric}"] = value(
                    model, domain, "L1", metric
                )

        direct["Mean_HRR"] = (
            direct["D3_HRR"] + direct["D4_HRR"]
        ) / 2
        table8.append(direct)

        for level in LEVELS:
            for metric in ("RR", "HRR"):
                # Equal weight for each domain; missing domains stay missing.
                hierarchy[f"{level}_{metric}"] = sum(
                    value(model, domain, level, metric)
                    for domain in DOMAINS
                ) / len(DOMAINS)

        hierarchy["EG_pp"] = (
            hierarchy["L3-session_HRR"] - hierarchy["L1_HRR"]
        )

        first_steps = []
        for domain in DOMAINS:
            selected = df[
                (df["model"] == model)
                & (df["domain"] == domain)
                & (df["level"] == "L3-session")
                & (df["status"] == "ok")
            ]
            first_steps.append(
                selected["first_refusal_step"].mean()
                if len(selected) and "first_refusal_step" in selected
                else float("nan")
            )

        hierarchy["First_refusal_step"] = sum(first_steps) / 2
        table9.append(hierarchy)

    pd.DataFrame(table8).round(1).to_csv(
        output_dir / "table8.csv", index=False
    )
    pd.DataFrame(table9).round(1).to_csv(
        output_dir / "table9.csv", index=False
    )
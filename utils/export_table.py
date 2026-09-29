# export_tables.py
import json
from pathlib import Path

import pandas as pd

from metrics.metrics import compute_metrics


LEVELS = ["L1", "L2a", "L2b", "L3-session", "L3-reset"]
DOMAINS = [f"D{i}" for i in range(1, 7)]


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

    subcategory_rows = []
    for (model, domain, subcategory, level), group in df.groupby(
        ["model", "domain", "subcategory", "level"]
    ):
        metrics = compute_metrics(group.to_dict("records"))
        subcategory_rows.append({
            "model": model,
            "domain": domain,
            "subcategory": subcategory,
            "level": level,
            "N_images": group["case_id"].nunique(),
            "N_valid_runs": metrics["n"],
            "N_error_runs": metrics["n_error"],
            "RR (%)": None if metrics["RR"] is None else 100 * metrics["RR"],
            "HRR (%)": None if metrics["HRR"] is None else 100 * metrics["HRR"],
        })
    subcategories = pd.DataFrame(subcategory_rows).round(1)
    subcategories.to_csv(output_dir / "subcategories.csv", index=False)

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

        direct["Mean_HRR"] = pd.Series(
            [direct[f"{domain}_HRR"] for domain in DOMAINS]
        ).mean()
        table8.append(direct)

        for level in LEVELS:
            for metric in ("RR", "HRR"):
                # Equal weight among domains measured in this pilot run.
                hierarchy[f"{level}_{metric}"] = pd.Series([
                    value(model, domain, level, metric)
                    for domain in DOMAINS
                ]).mean()

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

        hierarchy["First_refusal_step"] = pd.Series(first_steps).mean()
        table9.append(hierarchy)

    table8 = pd.DataFrame(table8).round(1)
    table9 = pd.DataFrame(table9).round(1)
    table8.to_csv(output_dir / "table8.csv", index=False)
    table9.to_csv(output_dir / "table9.csv", index=False)
    return table8, table9, subcategories


def make_comparison_table(subcategories, output_dir="tables"):
    """Show each model's L1, L2a, and L2b results by subcategory."""
    levels = ("L1", "L2a", "L2b")
    rows = []
    for (domain, subcategory, model), group in subcategories.groupby(
        ["domain", "subcategory", "model"]
    ):
        row = {
            "Domain": domain,
            "Subcategory": subcategory,
            "Images": int(group["N_images"].max()),
            "Model": model,
        }
        for level in levels:
            match = group[group["level"] == level]
            row[f"{level} RR (%)"] = match.iloc[0]["RR (%)"] if len(match) else None
            row[f"{level} HRR (%)"] = match.iloc[0]["HRR (%)"] if len(match) else None
            row[f"{level} valid"] = int(match.iloc[0]["N_valid_runs"]) if len(match) else None
        row["Errors"] = int(group[group["level"].isin(levels)]["N_error_runs"].sum())
        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(["Domain", "Subcategory", "Model"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    return comparison

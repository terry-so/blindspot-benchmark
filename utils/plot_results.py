"""Plain Matplotlib plots for the named tables from make_analysis_tables."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

LEVELS = ["L1", "L2a", "L2b", "L3-reset"]
LABELS = {
    "flare": "Flare",
    "sunburst": "Sunburst",
    "gemini-3.1-flash-image": "Gemini Flash Image",
}


def plot_model_comparison(tables, models, colors):
    matched = tables["overall_matched"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    x = np.arange(len(LEVELS))
    width = 0.8 / len(models)
    counts = matched.groupby("Level")["N_valid"].first()
    ticks = [f"{level}\nn={int(counts[level])}" for level in LEVELS]
    for ax, metric, title in zip(
        axes, ["HRR (%)", "RR (%)"], ["Harmful realization rate", "Refusal rate"]
    ):
        for i, model in enumerate(models):
            group = (
                matched[matched["Model"] == model].set_index("Level").reindex(LEVELS)
            )
            bars = ax.bar(
                x + (i - (len(models) - 1) / 2) * width,
                group[metric],
                width,
                color=colors[model],
                label=LABELS.get(model, model),
            )
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.set(
            xticks=x, xticklabels=ticks, ylim=(0, 108), ylabel="Percent", title=title
        )
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    axes[0].legend(fontsize=8)
    fig.suptitle("Model comparison — same valid cases within each level")
    return fig


def plot_domains(tables, models, colors):
    domain = tables["domain_summary"]
    domains = sorted(domain["Domain"].unique())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    cmap = plt.colormaps["YlOrRd"].copy()
    cmap.set_bad("#eeeeee")
    for ax, level in zip(axes.flat, LEVELS):
        group = domain[domain["Level"] == level]
        values = group.pivot(index="Domain", columns="Model", values="HRR (%)").reindex(
            index=domains, columns=models
        )
        image = ax.imshow(
            np.ma.masked_invalid(values.to_numpy()),
            vmin=0,
            vmax=100,
            cmap=cmap,
            aspect="auto",
        )
        for y, d in enumerate(domains):
            for x, model in enumerate(models):
                row = group[(group["Domain"] == d) & (group["Model"] == model)].iloc[0]
                value = row["HRR (%)"]
                text = (
                    f"{value:.1f}%\n{row['HRR_count']}"
                    if np.isfinite(value)
                    else "N/A\n0/0"
                )
                ax.text(
                    x,
                    y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if np.isfinite(value) and value >= 65 else "black",
                )
        ax.set(
            xticks=range(len(models)),
            xticklabels=[LABELS.get(m, m) for m in models],
            yticks=range(len(domains)),
            yticklabels=domains,
            title=level,
        )
    fig.colorbar(image, ax=list(axes.flat), label="HRR (%)", shrink=0.8)
    fig.suptitle("Domain harmful realization — all available valid results")
    return fig


def plot_paired_changes(tables, models, colors):
    paired = tables["paired_levels"]
    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained")
    tick_labels = []
    for y, (_, row) in enumerate(paired.iterrows()):
        value = row["HRR_change (pp)"]
        tick_labels.append(
            f"{LABELS.get(row['Model'], row['Model'])}: {row['Comparison']} (n={row['N_valid_pairs']})"
        )
        if np.isfinite(value):
            ax.barh(y, value, color=colors[row["Model"]])
            ax.annotate(
                f"{value:+.1f}",
                (value, y),
                xytext=(7 if value >= 0 else -7, 0),
                textcoords="offset points",
                va="center",
                ha="left" if value >= 0 else "right",
            )
        else:
            ax.text(0, y, "N/A", va="center")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)
    ax.set(
        yticks=range(len(tick_labels)),
        yticklabels=tick_labels,
        xlabel="Change in HRR from L1 (percentage points)",
        title="Within-model paired level comparisons",
    )
    ax.invert_yaxis()
    ax.margins(x=0.25)
    ax.grid(axis="x", alpha=0.2)
    return fig


def plot_l3(tables, models, colors):
    chains = tables["l3_summary"].set_index("Model").reindex(models)
    steps = tables["l3_steps"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    categories = [
        ("N_refused", "Refused", "#0072B2"),
        ("N_harmful", "Harmful realization", "#D55E00"),
        ("Other_nonrefused", "Other nonrefused", "#E69F00"),
        ("N_errors", "Technical error", "#777777"),
        ("N_missing", "Missing", "#dddddd"),
    ]
    chains["Other_nonrefused"] = chains["N_nonrefused"] - chains["N_harmful"]
    left = np.zeros(len(models))
    for column, label, color in categories:
        values = 100 * chains[column].to_numpy() / chains["N_expected"].to_numpy()
        axes[0].barh(range(len(models)), values, left=left, color=color, label=label)
        for y, value in enumerate(values):
            if value >= 5:
                axes[0].text(
                    left[y] + value / 2,
                    y,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if column in ("N_refused", "N_errors") else "black",
                )
        left += values
    axes[0].set(
        yticks=range(len(models)),
        yticklabels=[LABELS.get(m, m) for m in models],
        xlim=(0, 100),
        xlabel="Percent of expected chains",
        title="L3-reset outcomes",
    )
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8)
    positions = sorted(steps["Step"].unique())
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        group = steps[steps["Model"] == model].set_index("Step").reindex(positions)
        bars = axes[1].bar(
            np.arange(len(positions)) + (i - (len(models) - 1) / 2) * width,
            group["Refusal_among_reached (%)"],
            width,
            color=colors[model],
            label=LABELS.get(model, model),
        )
        for bar, (_, row) in zip(bars, group.iterrows()):
            if np.isfinite(bar.get_height()):
                axes[1].text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 2,
                    f"n={int(row['N_reached_step'])}",
                    ha="center",
                    fontsize=8,
                )
    highest_rate = steps["Refusal_among_reached (%)"].max()
    upper = max(10, highest_rate * 1.4 + 3) if np.isfinite(highest_rate) else 10
    axes[1].set(
        xticks=range(len(positions)),
        xticklabels=[f"Step {s}" for s in positions],
        ylim=(0, min(110, upper)),
        ylabel="Refusal among chains reaching step (%)",
        title="Where valid chains refuse",
    )
    axes[1].legend(fontsize=8)
    return fig


def plot_subcategories(tables, models, colors):
    cmap = plt.colormaps["YlOrRd"].copy()
    cmap.set_bad("#eeeeee")
    subcategories = tables["subcategories"].copy()
    subcategories["Subcategory"] = subcategories["Subcategory"].astype(str)
    names = sorted(
        subcategories["Subcategory"].unique(),
        key=lambda name: tuple(map(int, name.split("."))),
    )
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(4 * len(models), 13),
        layout="constrained",
        squeeze=False,
    )
    for ax, model in zip(axes.flat, models):
        group = subcategories[subcategories["Model"] == model]
        values = group.pivot(
            index="Subcategory", columns="Level", values="HRR (%)"
        ).reindex(index=names, columns=LEVELS)
        image = ax.imshow(
            np.ma.masked_invalid(values.to_numpy()),
            vmin=0,
            vmax=100,
            cmap=cmap,
            aspect="auto",
        )
        for y, name in enumerate(names):
            for x, level in enumerate(LEVELS):
                row = group[
                    (group["Subcategory"] == name) & (group["Level"] == level)
                ].iloc[0]
                value = row["HRR (%)"]
                ax.text(
                    x,
                    y,
                    row["HRR_count"] if np.isfinite(value) else "N/A",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if np.isfinite(value) and value >= 65 else "black",
                )
        ax.set(
            xticks=range(len(LEVELS)),
            xticklabels=LEVELS,
            yticks=range(len(names)),
            yticklabels=names,
            title=LABELS.get(model, model),
        )
        ax.tick_params(axis="x", labelrotation=30)
    fig.colorbar(image, ax=list(axes.flat), label="HRR (%)", shrink=0.5)
    fig.suptitle("Subcategory inspection — labels show harmful / valid counts")
    return fig


def make_plots(tables, output_dir="plots", *, show=True):
    """Show five figures and save PNG/PDF copies. No model calls are made."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models = sorted(tables["overall"]["Model"].unique())
    palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = {model: palette[i % len(palette)] for i, model in enumerate(models)}
    plots = [
        ("01_model_comparison", plot_model_comparison),
        ("02_domain_hrr", plot_domains),
        ("03_paired_changes", plot_paired_changes),
        ("04_l3_outcomes_and_steps", plot_l3),
        ("05_subcategory_counts", plot_subcategories),
    ]
    files = []
    for name, draw in plots:
        fig = draw(tables, models, colors)
        try:
            for extension in ("png", "pdf"):
                path = output_dir / f"{name}.{extension}"
                fig.savefig(path, dpi=200, bbox_inches="tight")
                files.append(path)
            if show:
                plt.show()
        finally:
            plt.close(fig)
    return files

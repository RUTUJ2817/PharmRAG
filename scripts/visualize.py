"""
PharmRAG — Publication-Grade Visualization Suite
Generates 8 research-quality dark-theme figures for the benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Dark research theme ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "grid.alpha": 0.6,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "legend.facecolor": "#161b22",
    "legend.edgecolor": "#30363d",
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "#0d1117",
})

PALETTE = {
    "ungrounded": "#f85149",
    "naive_rag": "#d29922",
    "hybrid_rag": "#3fb950",
}
MODE_LABELS = {
    "ungrounded": "Ungrounded",
    "naive_rag": "Naive RAG",
    "hybrid_rag": "Hybrid RAG",
}
MODEL_COLORS = {
    "Llama 3.3 70B": "#58a6ff",
    "Llama 4 Scout": "#bc8cff",
    "Qwen 3 32B": "#f778ba",
}


def load_data():
    df = pd.read_csv(os.path.join("data", "results", "scores.csv"))
    return df


def fig1_safety_ablation(df):
    """Bar chart: Mean safety score by mode, grouped by difficulty level."""
    fig, ax = plt.subplots(figsize=(10, 6))
    levels = sorted(df["level"].unique())
    modes = ["ungrounded", "naive_rag", "hybrid_rag"]
    x = np.arange(len(levels))
    w = 0.25

    for i, mode in enumerate(modes):
        vals = [df[(df["level"] == lv) & (df["mode"] == mode)]["total"].mean() for lv in levels]
        bars = ax.bar(x + i * w, vals, w, color=PALETTE[mode], label=MODE_LABELS[mode],
                      edgecolor="#30363d", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="#c9d1d9")

    ax.set_xlabel("Query Difficulty Level")
    ax.set_ylabel("Mean Safety Score (0–8)")
    ax.set_title("Safety Score Ablation: Ungrounded → Naive RAG → Hybrid RAG")
    ax.set_xticks(x + w)
    ax.set_xticklabels(["L1: Factual", "L2: Multi-hop", "L3: Adversarial"])
    ax.legend()
    ax.set_ylim(0, 9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(os.path.join("figures", "fig1_safety_ablation.png"))
    plt.close()
    print("  [OK] fig1_safety_ablation.png")


def fig2_hallucination_waterfall(df):
    """Waterfall chart showing hallucination rate reduction across modes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    modes = ["ungrounded", "naive_rag", "hybrid_rag"]
    rates = []
    for mode in modes:
        sub = df[df["mode"] == mode]
        rate = sub["hallucination"].mean() * 100
        rates.append(rate)

    colors = [PALETTE[m] for m in modes]
    bars = ax.bar(range(len(modes)), rates, color=colors, edgecolor="#30363d", width=0.5)

    # Add connecting arrows
    for i in range(len(rates) - 1):
        delta = rates[i] - rates[i + 1]
        if delta > 0:
            ax.annotate(f"−{delta:.0f}%", xy=(i + 0.5, (rates[i] + rates[i+1]) / 2),
                        fontsize=11, color="#3fb950", ha="center", fontweight="bold")

    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold", color="#c9d1d9")

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_LABELS[m] for m in modes])
    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_title("Hallucination Reduction Waterfall")
    ax.set_ylim(0, max(rates) + 15)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(os.path.join("figures", "fig2_hallucination_waterfall.png"))
    plt.close()
    print("  [OK] fig2_hallucination_waterfall.png")


def fig3_error_heatmap(df):
    """Heatmap: error rate by model × mode."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = list(df["model"].unique())
    modes = ["ungrounded", "naive_rag", "hybrid_rag"]
    matrix = np.zeros((len(models), len(modes)))

    for i, model in enumerate(models):
        for j, mode in enumerate(modes):
            sub = df[(df["model"] == model) & (df["mode"] == mode)]
            matrix[i, j] = sub["error"].mean() * 100

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_LABELS[m] for m in modes])
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)

    for i in range(len(models)):
        for j in range(len(modes)):
            color = "white" if matrix[i, j] > 50 else "#0d1117"
            ax.text(j, i, f"{matrix[i,j]:.0f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_title("Error Rate Heatmap: Model × Retrieval Strategy")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Error Rate (%)", color="#c9d1d9")
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color="#8b949e")
    fig.savefig(os.path.join("figures", "fig3_error_heatmap.png"))
    plt.close()
    print("  [OK] fig3_error_heatmap.png")


def fig4_taxonomy_distribution(df):
    """Stacked bar: error taxonomy breakdown by mode."""
    fig, ax = plt.subplots(figsize=(10, 6))
    modes = ["ungrounded", "naive_rag", "hybrid_rag"]
    tax_types = ["Hallucination", "Sycophancy", "Omission", "Retrieval_Failure", "None"]
    tax_colors = {
        "Hallucination": "#f85149",
        "Sycophancy": "#d29922",
        "Omission": "#58a6ff",
        "Retrieval_Failure": "#bc8cff",
        "None": "#3fb950",
    }

    bottoms = np.zeros(len(modes))
    for tax in tax_types:
        vals = []
        for mode in modes:
            sub = df[df["mode"] == mode]
            count = (sub["taxonomy"] == tax).sum()
            vals.append(count / len(sub) * 100 if len(sub) > 0 else 0)
        ax.bar(range(len(modes)), vals, bottom=bottoms, color=tax_colors[tax],
               label=tax, edgecolor="#30363d", linewidth=0.5, width=0.5)
        bottoms += vals

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels([MODE_LABELS[m] for m in modes])
    ax.set_ylabel("Proportion (%)")
    ax.set_title("Error Taxonomy Distribution by Retrieval Strategy")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(os.path.join("figures", "fig4_taxonomy_distribution.png"))
    plt.close()
    print("  [OK] fig4_taxonomy_distribution.png")


def fig5_radar_per_model(df):
    """Radar chart: D1–D4 scores per model (hybrid_rag mode only)."""
    dims = ["D1", "D2", "D3", "D4"]
    dim_labels = ["Accuracy", "Completeness", "Safety", "Grounding"]
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor("#161b22")
    fig.patch.set_facecolor("#0d1117")

    for model, color in MODEL_COLORS.items():
        sub = df[(df["model"] == model) & (df["mode"] == "hybrid_rag")]
        if sub.empty:
            continue
        vals = [sub[d].mean() for d in dims]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", color=color, linewidth=2, label=model)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels)
    ax.set_ylim(0, 2.1)
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticklabels(["0.5", "1.0", "1.5", "2.0"], color="#8b949e", size=8)
    ax.set_title("Per-Model Safety Profile (Hybrid RAG)", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0))
    ax.grid(color="#30363d")
    fig.savefig(os.path.join("figures", "fig5_radar_per_model.png"))
    plt.close()
    print("  [OK] fig5_radar_per_model.png")


def fig6_adversarial_resistance(df):
    """Grouped bar: adversarial query performance by model and mode."""
    fig, ax = plt.subplots(figsize=(10, 6))
    adv = df[df["category"] == "adversarial"]
    models = list(adv["model"].unique())
    modes = ["ungrounded", "naive_rag", "hybrid_rag"]
    x = np.arange(len(models))
    w = 0.25

    for i, mode in enumerate(modes):
        vals = [adv[(adv["model"] == m) & (adv["mode"] == mode)]["total"].mean() for m in models]
        ax.bar(x + i * w, vals, w, color=PALETTE[mode], label=MODE_LABELS[mode],
               edgecolor="#30363d", linewidth=0.5)

    ax.set_xticks(x + w)
    ax.set_xticklabels(models, rotation=15)
    ax.set_ylabel("Mean Safety Score")
    ax.set_title("Adversarial Sycophancy Resistance by Model")
    ax.legend()
    ax.set_ylim(0, 9)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(os.path.join("figures", "fig6_adversarial_resistance.png"))
    plt.close()
    print("  [OK] fig6_adversarial_resistance.png")


def fig7_nli_consistency(df):
    """Bar chart: NLI consistency label distribution for grounded modes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    grounded = df[df["mode"].isin(["naive_rag", "hybrid_rag"])]
    modes = ["naive_rag", "hybrid_rag"]
    labels_nli = ["Entailment", "Neutral", "Contradiction"]
    colors_nli = {"Entailment": "#3fb950", "Neutral": "#8b949e", "Contradiction": "#f85149"}
    x = np.arange(len(labels_nli))
    w = 0.3

    for i, mode in enumerate(modes):
        sub = grounded[grounded["mode"] == mode]
        vals = [(sub["nli_label"] == lb).sum() / len(sub) * 100 if len(sub) > 0 else 0 for lb in labels_nli]
        ax.bar(x + i * w, vals, w, color=[colors_nli[lb] for lb in labels_nli],
               alpha=0.5 + 0.5 * i, edgecolor="#30363d", linewidth=0.5,
               label=MODE_LABELS[mode])

    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(labels_nli)
    ax.set_ylabel("Proportion (%)")
    ax.set_title("NLI Factual Consistency: Naive RAG vs Hybrid RAG")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(os.path.join("figures", "fig7_nli_consistency.png"))
    plt.close()
    print("  [OK] fig7_nli_consistency.png")


def fig8_category_breakdown(df):
    """Grouped bar: error rate by query category and mode."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cats = sorted(df["category"].unique())
    modes = ["ungrounded", "naive_rag", "hybrid_rag"]
    x = np.arange(len(cats))
    w = 0.25

    for i, mode in enumerate(modes):
        vals = [df[(df["category"] == c) & (df["mode"] == mode)]["error"].mean() * 100 for c in cats]
        ax.bar(x + i * w, vals, w, color=PALETTE[mode], label=MODE_LABELS[mode],
               edgecolor="#30363d", linewidth=0.5)

    ax.set_xticks(x + w)
    ax.set_xticklabels([c.replace("_", " ").title() for c in cats], rotation=15)
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Error Rate by Clinical Query Category")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.savefig(os.path.join("figures", "fig8_category_breakdown.png"))
    plt.close()
    print("  [OK] fig8_category_breakdown.png")


def print_metrics(df):
    print("\n" + "=" * 70)
    print("  P H A R M R A G  -  B E N C H M A R K  R E S U L T S")
    print("=" * 70)

    for mode in ["ungrounded", "naive_rag", "hybrid_rag"]:
        sub = df[df["mode"] == mode]
        hal = sub["hallucination"].mean() * 100
        err = sub["error"].mean() * 100
        score = sub["total"].mean()
        print(f"\n  {MODE_LABELS[mode]:>12s}  |  Score: {score:.1f}/8  |  Hallucination: {hal:.0f}%  |  Error: {err:.0f}%")

    print(f"\n{'-' * 70}")
    print("  Per-Model Breakdown (Hybrid RAG):")
    for model in df["model"].unique():
        sub = df[(df["model"] == model) & (df["mode"] == "hybrid_rag")]
        if sub.empty:
            continue
        print(f"    {model:>20s}  |  Score: {sub['total'].mean():.1f}  |  Hal: {sub['hallucination'].mean()*100:.0f}%  |  Err: {sub['error'].mean()*100:.0f}%")

    print(f"\n{'-' * 70}")
    h_un = df[df["mode"] == "ungrounded"]["hallucination"].mean() * 100
    h_hr = df[df["mode"] == "hybrid_rag"]["hallucination"].mean() * 100
    print(f"  HEADLINE: Hybrid RAG reduced hallucinations from {h_un:.0f}% to {h_hr:.0f}%")
    print("=" * 70)


def main():
    os.makedirs("figures", exist_ok=True)
    df = load_data()
    print("Generating 8 publication figures…\n")
    fig1_safety_ablation(df)
    fig2_hallucination_waterfall(df)
    fig3_error_heatmap(df)
    fig4_taxonomy_distribution(df)
    fig5_radar_per_model(df)
    fig6_adversarial_resistance(df)
    fig7_nli_consistency(df)
    fig8_category_breakdown(df)
    print_metrics(df)


if __name__ == "__main__":
    main()

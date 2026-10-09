"""Visualização: comparação de modelos e contradições."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(report, save_path="modelcheck_report.png"):
    """Barras lado a lado por modelo + contradições em vermelho."""
    table = report["comparison_table"]
    normalized = table.div(table.max(axis=0), axis=1).fillna(0)

    contradicted = set()
    for model, items in report.get("contradictions", {}).items():
        for c in items:
            contradicted.add((model, c["feature"]))

    features = list(normalized.index)
    models = list(normalized.columns)
    n_models = len(models)
    y = np.arange(len(features))
    height = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(features))))
    palette = plt.cm.tab10.colors

    for i, model in enumerate(models):
        values = normalized[model].values
        offsets = y + (i - (n_models - 1) / 2) * height
        colors = [
            "#d62728" if (model, f) in contradicted else palette[i % 10]
            for f in features
        ]
        bars = ax.barh(offsets, values, height=height, color=colors,
                       label=model, edgecolor="white", linewidth=0.5)
        for bar, f in zip(bars, features):
            if (model, f) in contradicted:
                ax.text(bar.get_width() + 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        "✗ contradição", va="center", fontsize=8,
                        color="#d62728", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel("Importância SHAP (normalizada)")
    ax.set_title("modelcheck — comparação de modelos\n(vermelho = contradiz regra de negócio)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

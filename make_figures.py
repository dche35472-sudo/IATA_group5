from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT  = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

def load_metrics(rel_path: str) -> dict:
    return json.loads((ROOT / rel_path).read_text())

# BoW baseline vs Slot model
def fig_text_representation():
    base   = load_metrics("baseline_unified_metrics.json")["test"]
    struct = load_metrics("results_TextRepresentation/metrics.json")

    metrics = ["mean_slot_accuracy", "attribute_accuracy", "relation_accuracy"]
    labels  = ["Mean slot acc.", "Attribute acc.", "Relation acc."]

    bow_vals  = [base[m]   * 100 for m in metrics]
    slot_vals = [struct[m] * 100 for m in metrics]

    x = np.arange(len(metrics))
    w = 0.38

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    b1 = ax.bar(x - w/2, bow_vals,  w, label="Bag-of-words baseline",
                color="#a3b8c7", edgecolor="#2b3a4a")
    b2 = ax.bar(x + w/2, slot_vals, w, label="Structured slot output",
                color="#3d6f8c", edgecolor="#1c2c3a")

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.7,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, max(slot_vals + bow_vals) * 1.25)
    ax.set_title("Text representation: structured slot output vs bag-of-words")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_text_representation.png", bbox_inches="tight")
    plt.close(fig)

# CNN architecture comparison (under slot output)

def fig_architecture_comparison():
    base = ROOT / "CNN_architecture_variant"
    runs = {
        "SimpleCNN":  load_metrics("CNN_architecture_variant/slot_simple_CNN_outputs/metrics.json")["test"],
        "DeeperCNN":  load_metrics("CNN_architecture_variant/slot_deeper_CNN_outputs/metrics.json")["test"],
        "ResNet18":   load_metrics("CNN_architecture_variant/slot_ResNet_outputs/metrics.json")["test"],
    }

    metrics = ["mean_slot_accuracy", "attribute_accuracy", "relation_accuracy"]
    labels  = ["Mean slot acc.", "Attribute acc.", "Relation acc."]

    x = np.arange(len(metrics))
    w = 0.27
    colors  = ["#a3b8c7", "#3d6f8c", "#1c2c3a"]

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (name, m) in enumerate(runs.items()):
        vals = [m[k] * 100 for k in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=name,
                      color=colors[i], edgecolor="#1c2c3a")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("CNN architecture (under fixed slot-structured output)")
    ax.set_ylim(0, 60)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_architecture_comparison.png", bbox_inches="tight")
    plt.close(fig)

# Per-slot accuracy heatmap 

def fig_per_slot_heatmap():
    m = load_metrics("results_TextRepresentation/metrics.json")
    head_order = [
        "anchor_size", "anchor_color", "anchor_shape",
        "rel1",
        "target1_size", "target1_color", "target1_shape",
        "rel2",
        "target2_size", "target2_color", "target2_shape",
    ]
    vals = [m[f"acc_{k}"] * 100 for k in head_order]

    fig, ax = plt.subplots(figsize=(6.4, 1.7))
    img = ax.imshow(np.array(vals)[None, :], cmap="Blues", vmin=20, vmax=60,
                    aspect="auto")
    for i, v in enumerate(vals):
        ax.text(i, 0, f"{v:.0f}", ha="center", va="center",
                color="white" if v > 45 else "black", fontsize=10)
    ax.set_yticks([])
    ax.set_xticks(range(len(head_order)))
    ax.set_xticklabels([h.replace("_", "\n") for h in head_order],
                       rotation=0, fontsize=9)
    ax.set_title("Per-slot test accuracy (controlled slot model)")
    fig.colorbar(img, ax=ax, fraction=0.025, pad=0.02, label="Acc (%)")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_per_slot_heatmap.png", bbox_inches="tight")
    plt.close(fig)

# Figure 4 : Relation confusion matrix 

def fig_relation_confusion():
    cm = pd.read_csv(ROOT / "results_TextRepresentation/test_rel1_confusion.csv",
                     index_col=0)
    # Order rows/cols consistently
    order = ["above", "below", "left of", "right of", "overlapping"]
    cm = cm.reindex(index=order, columns=order, fill_value=0)
    norm = cm.div(cm.sum(axis=1), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    img = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=0.5)
    for i in range(len(order)):
        for j in range(len(order)):
            v = norm.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.3 else "black", fontsize=9)
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_yticklabels(order)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Gold")
    ax.set_title("Relation 1 confusion matrix (row-normalised)")
    fig.colorbar(img, ax=ax, fraction=0.045, pad=0.04, label="P(pred|gold)")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_relation_confusion.png", bbox_inches="tight")
    plt.close(fig)


# Performance degradation by object count

def fig_object_count_degradation():
    df = pd.read_csv(ROOT / "CNN_architecture_variant/slot_structure_cnn_architecture_analysis_outputs/by_object_count.csv")

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for model, color, marker in [
        ("SimpleCNN", "#a3b8c7", "o"),
        ("DeeperCNN", "#3d6f8c", "s"),
        ("ResNet18",  "#1c2c3a", "^"),
    ]:
        sub = df[df["model"] == model].sort_values("object_count")
        ax.plot(sub["object_count"], sub["slot_ave_accuracy"] * 100,
                marker=marker, color=color, label=f"{model} (slot avg)",
                linewidth=2)
        ax.plot(sub["object_count"], sub["relation_slot_accuracy"] * 100,
                marker=marker, color=color, linestyle="--", alpha=0.7,
                label=f"{model} (relation)")

    ax.set_xticks([3, 4, 5])
    ax.set_xlabel("Number of objects in scene")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Performance vs. scene complexity")
    ax.legend(loc="lower left", ncol=2, fontsize=8, frameon=False)
    ax.grid(linestyle=":", alpha=0.5)
    ax.set_ylim(20, 55)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_object_count.png", bbox_inches="tight")
    plt.close(fig)


# Same vs mixed-relation pair accuracy

def fig_pair_type():
    df = pd.read_csv(ROOT / "CNN_architecture_variant/slot_structure_cnn_architecture_analysis_outputs/by_relation_pair_type.csv")

    models = ["SimpleCNN", "DeeperCNN", "ResNet18"]
    pair_types = ["same_relation", "mixed_relation"]
    pair_labels = ["Same-relation pair", "Mixed-relation pair"]

    x = np.arange(len(pair_types))
    w = 0.27
    colors = ["#a3b8c7", "#3d6f8c", "#1c2c3a"]

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for i, model in enumerate(models):
        vals = []
        for pt in pair_types:
            row = df[(df["model"] == model) & (df["relation_pair_type"] == pt)].iloc[0]
            vals.append(row["relation_pair_accuracy"] * 100)
        bars = ax.bar(x + (i - 1) * w, vals, w, label=model,
                      color=colors[i], edgecolor="#1c2c3a")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                    f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels)
    ax.set_ylabel("Both relations correct (%)")
    ax.set_ylim(0, 35)
    ax.set_title("Relation-pair accuracy: same-pair vs mixed-pair")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_pair_type.png", bbox_inches="tight")
    plt.close(fig)


#  summary 

def fig_headline():
    base   = load_metrics("baseline_unified_metrics.json")["test"]
    struct = load_metrics("results_TextRepresentation/metrics.json")
    arch_runs = {
        "SimpleCNN":  load_metrics("CNN_architecture_variant/slot_simple_CNN_outputs/metrics.json")["test"],
        "DeeperCNN":  load_metrics("CNN_architecture_variant/slot_deeper_CNN_outputs/metrics.json")["test"],
        "ResNet18":   load_metrics("CNN_architecture_variant/slot_ResNet_outputs/metrics.json")["test"],
    }

    rows = [
        ("BoW baseline",       base["mean_slot_accuracy"], base["relation_accuracy"]),
        ("Slot CNN (4-layer)", struct["mean_slot_accuracy"], struct["relation_accuracy"]),
        ("Slot SimpleCNN",  arch_runs["SimpleCNN"]["mean_slot_accuracy"],  arch_runs["SimpleCNN"]["relation_accuracy"]),
        ("Slot DeeperCNN",  arch_runs["DeeperCNN"]["mean_slot_accuracy"],  arch_runs["DeeperCNN"]["relation_accuracy"]),
        ("Slot ResNet18",   arch_runs["ResNet18"]["mean_slot_accuracy"],   arch_runs["ResNet18"]["relation_accuracy"]),
    ]

    llm_metrics_path = ROOT / "llm_comparison/run_200/metrics.json"
    if llm_metrics_path.exists():
        llm = json.loads(llm_metrics_path.read_text())
        rows.append((f"Gemini ({llm.get('model','?')})",
                     llm["mean_slot_accuracy"], llm["relation_accuracy"]))

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    names    = [r[0] for r in rows]
    slot_v   = [r[1] * 100 for r in rows]
    rel_v    = [r[2] * 100 for r in rows]
    x = np.arange(len(names))
    w = 0.4
    ax.bar(x - w/2, slot_v, w, label="Mean slot acc.",
           color="#3d6f8c", edgecolor="#1c2c3a")
    ax.bar(x + w/2, rel_v,  w, label="Relation acc.",
           color="#a3b8c7", edgecolor="#1c2c3a")
    for i, v in enumerate(slot_v):
        ax.text(i - w/2, v + 0.7, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    for i, v in enumerate(rel_v):
        ax.text(i + w/2, v + 0.7, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, max(slot_v + rel_v) * 1.25)
    ax.set_title("Headline comparison: all systems")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig7_headline.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_text_representation()
    print("[ok] fig1_text_representation.png")
    fig_architecture_comparison()
    print("[ok] fig2_architecture_comparison.png")
    fig_per_slot_heatmap()
    print("[ok] fig3_per_slot_heatmap.png")
    fig_relation_confusion()
    print("[ok] fig4_relation_confusion.png")
    fig_object_count_degradation()
    print("[ok] fig5_object_count.png")
    fig_pair_type()
    print("[ok] fig6_pair_type.png")
    fig_headline()
    print("[ok] fig7_headline.png")

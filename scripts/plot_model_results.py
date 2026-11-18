"""Report figures for model comparison and optimisation (owner: Person 2).

Reads only CSV/JSON files written by train_compare.py, bayes_opt.py and
finalize_model.py, so every figure is traceable to recorded results.
No training curves are drawn for models without an epoch-wise history.

Usage:  python scripts/plot_model_results.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import load_config, results_dir


def main() -> None:
    cfg = load_config()
    m_dir, bo_dir = results_dir(cfg, "models"), results_dir(cfg, "bayes_opt")
    fig_dir = results_dir(cfg, "models", "figures")
    comp = pd.read_csv(m_dir / "comparison.csv")
    folds = pd.read_csv(m_dir / "cv_folds.csv")
    order = comp.sort_values("cv_top1_mean", ascending=False).model.tolist()
    x = np.arange(len(order))
    c = comp.set_index("model").loc[order]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, c.cv_macro_f1_mean, yerr=c.cv_macro_f1_std, capsize=4)
    for i, m in enumerate(order):
        ax.scatter(np.full((folds.model == m).sum(), i), folds.loc[folds.model == m, "macro_f1"], color="k", s=8)
    ax.set(xticks=x, xticklabels=order, ylabel="macro F1", title="Cross-validation macro F1 (GroupKFold by trip; dots = folds)")
    ax.set_ylim(max(0, c.cv_macro_f1_mean.min() - 0.1), 1)
    plt.xticks(rotation=30); fig.tight_layout(); fig.savefig(fig_dir / "cv_macro_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.38
    ax.bar(x - w / 2, c.st_top1_accuracy, w, label="Top-1 accuracy")
    ax.bar(x + w / 2, c.st_macro_f1, w, label="macro F1")
    ax.set(xticks=x, xticklabels=order, title="Student-test performance", ylim=(0.5, 1))
    ax.legend(); plt.xticks(rotation=30); fig.tight_layout()
    fig.savefig(fig_dir / "student_test_top1_f1.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - w / 2, c.cv_log_loss_mean, w, label="CV")
    ax.bar(x + w / 2, c.st_log_loss, w, label="student-test")
    ax.set(xticks=x, xticklabels=order, ylabel="log loss (lower is better)", title="Log loss")
    ax.legend(); plt.xticks(rotation=30); fig.tight_layout(); fig.savefig(fig_dir / "log_loss.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(c.st_predict_ms_per_1k_obs, c.st_top1_accuracy)
    for m in order:
        ax.annotate(m, (c.loc[m, "st_predict_ms_per_1k_obs"], c.loc[m, "st_top1_accuracy"]))
    ax.set(xscale="log", xlabel="inference ms per 1000 observations (log)", ylabel="student-test Top-1",
           title="Accuracy vs inference cost")
    fig.tight_layout(); fig.savefig(fig_dir / "accuracy_vs_speed.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, c.st_route_continuity)
    ax.set(xticks=x, xticklabels=order, ylabel="fraction of connected consecutive arcs",
           title="Classifier-only route continuity (student-test)", ylim=(0, 1))
    plt.xticks(rotation=30); fig.tight_layout(); fig.savefig(fig_dir / "route_continuity.png", dpi=130); plt.close(fig)

    cms = json.loads((m_dir / "confusion_matrices.json").read_text())
    n = len(cms)
    fig, axs = plt.subplots(1, n, figsize=(3.2 * n, 3.2))
    for a, (m, cm) in zip(np.atleast_1d(axs), cms.items()):
        cm = np.array(cm)
        a.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=8)
        a.set(title=m, xticks=[0, 1], yticks=[0, 1], xlabel="pred", ylabel="true")
    fig.tight_layout(); fig.savefig(fig_dir / "confusion_matrices.png", dpi=130); plt.close(fig)

    hist = json.loads((m_dir / "mlp_history.json").read_text())
    for m, h in hist.items():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(h["train_loss"], label="training loss (log loss)")
        ax.set(xlabel="epoch", ylabel="loss", title=f"{m}: epoch-wise history")
        if h["val_accuracy"]:
            ax2 = ax.twinx()
            ax2.plot(h["val_accuracy"], color="C1", label="internal validation accuracy")
            ax2.set_ylabel("validation accuracy")
            ax2.legend(loc="center right")
        ax.legend(loc="upper right"); fig.tight_layout(); fig.savefig(fig_dir / f"history_{m}.png", dpi=130)
        plt.close(fig)

    tvu = bo_dir / "tuned_vs_untuned.csv"
    if tvu.exists():
        t = pd.read_csv(tvu).set_index("variant")
        metrics = ["cv_top1_mean", "cv_macro_f1_mean", "st_top1_accuracy", "st_macro_f1"]
        fig, ax = plt.subplots(figsize=(8, 4))
        xx = np.arange(len(metrics))
        for k, v in enumerate(t.index):
            ax.bar(xx + (k - 0.5) * 0.38, t.loc[v, metrics], 0.38, label=v)
        ax.set(xticks=xx, xticklabels=metrics, ylim=(0.7, 1), title=f"Tuned vs untuned ({t.model.iloc[0]})")
        ax.legend(); fig.tight_layout(); fig.savefig(fig_dir / "tuned_vs_untuned.png", dpi=130); plt.close(fig)
    print(f"figures written to {fig_dir}")


if __name__ == "__main__":
    main()

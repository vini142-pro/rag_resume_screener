# utils/visualizer.py
# ─────────────────────────────────────────────────────────────────────
#  Visualization Module — 4 charts for RAG Resume Screener
# ─────────────────────────────────────────────────────────────────────

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

os.makedirs("outputs", exist_ok=True)

PALETTE   = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
BG_COLOR  = "#F8F9FA"
GRID_COLOR = "#E0E0E0"


def _style_ax(ax, title: str):
    ax.set_facecolor(BG_COLOR)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)


# ── Chart 1: Semantic Similarity Rankings ─────────────────────────────
def plot_similarity_rankings(all_results: dict):
    """Horizontal bar chart of cosine similarity scores per role."""
    n_roles = len(all_results)
    fig, axes = plt.subplots(1, n_roles, figsize=(7 * n_roles, 6))
    if n_roles == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG_COLOR)

    for ax, (role, results) in zip(axes, all_results.items()):
        candidates = [r["candidate"] for r in results]
        scores     = [r["similarity_score"] * 100 for r in results]
        colors     = [PALETTE[0] if i == 0 else PALETTE[1] if s >= 60
                      else PALETTE[4] for i, s in enumerate(scores)]

        bars = ax.barh(candidates[::-1], scores[::-1], color=colors[::-1], height=0.6)
        for bar, score in zip(bars, scores[::-1]):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}%", va="center", fontsize=9, fontweight="bold")

        ax.set_xlim(0, 105)
        ax.axvline(60, color="red", linestyle="--", linewidth=1, alpha=0.6, label="60% threshold")
        _style_ax(ax, f"🎯 {role}")
        ax.set_xlabel("Semantic Similarity Score (%)")
        ax.legend(fontsize=8)

    plt.suptitle("RAG Resume Screener — Semantic Similarity Rankings",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = "outputs/01_similarity_rankings.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Saved: {path}")


# ── Chart 2: Embedding Similarity Heatmap ────────────────────────────
def plot_similarity_heatmap(all_results: dict, all_candidates: list[str]):
    """Heatmap of similarity scores: roles × candidates."""
    roles = list(all_results.keys())
    matrix = np.zeros((len(roles), len(all_candidates)))

    cand_index = {c: i for i, c in enumerate(all_candidates)}
    for r_idx, (role, results) in enumerate(all_results.items()):
        for r in results:
            c_idx = cand_index.get(r["candidate"])
            if c_idx is not None:
                matrix[r_idx][c_idx] = r["similarity_score"] * 100

    fig, ax = plt.subplots(figsize=(max(10, len(all_candidates) * 1.1), 4))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    sns.heatmap(
        matrix, ax=ax,
        xticklabels=all_candidates,
        yticklabels=roles,
        annot=True, fmt=".1f",
        cmap="YlGnBu",
        vmin=0, vmax=100,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Similarity (%)"}
    )
    ax.set_title("Semantic Similarity Heatmap — All Roles × All Candidates",
                 fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    path = "outputs/02_similarity_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Saved: {path}")


# ── Chart 3: Skill Match Breakdown ───────────────────────────────────
def plot_skill_breakdown(all_results: dict, jd_skills: dict):
    """Grouped bar: matched vs missing skills per top candidate per role."""
    fig, axes = plt.subplots(1, len(all_results),
                             figsize=(6 * len(all_results), 5))
    if len(all_results) == 1:
        axes = [axes]
    fig.patch.set_facecolor(BG_COLOR)

    for ax, (role, results) in zip(axes, all_results.items()):
        top5      = results[:5]
        names     = [r["candidate"] for r in top5]
        matched   = [len(set(r.get("skills", [])) & set(jd_skills.get(role, [])))
                     for r in top5]
        missing   = [len(set(jd_skills.get(role, [])) - set(r.get("skills", [])))
                     for r in top5]

        x = np.arange(len(names))
        w = 0.35
        ax.bar(x - w/2, matched, w, label="Matched Skills", color=PALETTE[1])
        ax.bar(x + w/2, missing, w, label="Missing Skills", color=PALETTE[2])
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("# Skills")
        _style_ax(ax, f"Skill Gap — {role}")
        ax.legend(fontsize=8)
        ax.grid(axis="y", color=GRID_COLOR)
        ax.grid(axis="x", visible=False)

    plt.suptitle("RAG Resume Screener — Skill Match & Gap Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = "outputs/03_skill_breakdown.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Saved: {path}")


# ── Chart 4: RAG Pipeline Flow Diagram ───────────────────────────────
def plot_rag_pipeline_diagram():
    """Visual diagram explaining the RAG pipeline flow."""
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor("#1A1A2E")
    ax.set_facecolor("#1A1A2E")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    steps = [
        (1.2,  2.0, "📄 Resumes\n(Input)",          "#4C72B0"),
        (3.5,  2.0, "🧠 HuggingFace\nEmbedder",      "#8172B2"),
        (5.8,  2.0, "📦 FAISS\nVector Store",         "#55A868"),
        (8.1,  2.0, "🔍 Semantic\nRetrieval",         "#CCB974"),
        (10.4, 2.0, "✍️ LLM\nGenerator",              "#C44E52"),
        (12.7, 2.0, "📋 Recruiter\nReport",           "#4C72B0"),
    ]

    for x, y, label, color in steps:
        rect = mpatches.FancyBboxPatch(
            (x - 0.9, y - 0.7), 1.8, 1.4,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9
        )
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center",
                fontsize=9, fontweight="bold", color="white")

    # Arrows
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + 0.9
        x2 = steps[i + 1][0] - 0.9
        y  = steps[i][1]
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="white", lw=1.8))

    # JD input arrow to retrieval
    ax.text(8.1, 3.5, "📝 Job Description", ha="center", va="center",
            fontsize=9, color="white", fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="#CCB974", alpha=0.7))
    ax.annotate("", xy=(8.1, 2.7), xytext=(8.1, 3.3),
                arrowprops=dict(arrowstyle="->", color="#CCB974", lw=1.8))

    ax.set_title("RAG Resume Screener — Pipeline Architecture",
                 fontsize=14, fontweight="bold", color="white", pad=15)

    plt.tight_layout()
    path = "outputs/04_rag_pipeline.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  📊  Saved: {path}")

"""Result charts for the P2 thesis, generated only from Thesis_Paper/data (derived from raw runs)."""

from __future__ import annotations

import json
import math

import numpy as np
from figstyle import (
    AXIS, BLUE, DATA, GRAY, GRID, INK, INK2, ORANGE, SEQ_BLUE, SURFACE, clean_axes, plt, save,
)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

R = json.loads((DATA / "results_summary.json").read_text(encoding="utf-8"))
RUNS = R["runs"]
BLUES = LinearSegmentedColormap.from_list("seqblue", SEQ_BLUE)
METHOD_COLOR = {"rl_lora": BLUE, "hlora_rl": ORANGE}
METHOD_NAME = {"rl_lora": "Standard RL-LoRA", "hlora_rl": "HLoRA-RL (direct transfer)"}
TASK_NAME = {"gsm8k": "GSM8K", "arc_challenge": "ARC-Challenge", "mbpp": "MBPP", "constraints": "Constraints"}


def series(key):
    return json.loads((DATA / f"series_{key}.json").read_text(encoding="utf-8"))


def rolling(values, window):
    v = np.asarray(values, dtype=float)
    out = np.full_like(v, np.nan)
    for i in range(len(v)):
        lo = max(0, i - window + 1)
        out[i] = np.nanmean(v[lo:i + 1])
    return out


def wilson(k, n, z=1.959963984540054):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


# --------------------------------------------------------------------------------------
def fig_phase2_matrices():
    keys = ["p05_order1_100", "p15_order1_100", "p15_order1_500", "p15_order2_500"]
    titles = ["(a) 0.5B, 100 updates/task\nT1 = GSM8K, T2 = ARC-Challenge", "(b) 1.5B, 100 updates/task\nT1 = GSM8K, T2 = ARC-Challenge",
              "(c) 1.5B, 500 updates/task\nT1 = GSM8K, T2 = ARC-Challenge", "(d) 1.5B, 500 updates/task\nT1 = MBPP, T2 = GSM8K"]
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.1))
    fig.subplots_adjust(wspace=0.35)
    for ax, key, title in zip(axes, keys, titles):
        m = RUNS[key]["matrix"]
        tasks, rows = m["tasks"], m["rows"]
        grid = np.array([[m["values"][r][t] if m["values"][r][t] is not None else np.nan for t in tasks] for r in rows])
        ax.imshow(np.nan_to_num(grid, nan=0), cmap=BLUES, vmin=0, vmax=1)
        for i, r in enumerate(rows):
            for j, t in enumerate(tasks):
                v = m["values"][r][t]
                if v is None:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#f4f4f2", ec=SURFACE, lw=2))
                    ax.text(j, i, "not yet\nlearned", ha="center", va="center", fontsize=6.5, color=GRAY)
                else:
                    ax.text(j, i, f"{100 * v:.1f}%", ha="center", va="center", fontsize=8.5,
                            color=SURFACE if v > 0.55 else INK)
        ax.set_xticks(range(len(tasks)), ["eval T1", "eval T2"], fontsize=7)
        ax.set_yticks(range(len(rows)), ["after T1", "after T2"], fontsize=7)
        ax.set_title(title, loc="left", fontsize=8)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        f = RUNS[key]["continual"]["average_forgetting"]
        ax.text(0.0, -0.2, f"average forgetting {100 * f:+.2f} pp", transform=ax.transAxes, fontsize=7, color=INK2)
    return save(fig, "fig_6_1_phase2_performance_matrices")


# --------------------------------------------------------------------------------------
def fig_continuation_rule():
    keys = ["p05_order1_100", "p15_order1_100", "p15_order1_500", "p15_order2_500", "arc_base_100", "arc_base_150"]
    labels = ["0.5B GSM8K->ARC (100)", "1.5B GSM8K->ARC (100)", "1.5B GSM8K->ARC (500)",
              "1.5B MBPP->GSM8K (500)", "1.5B ARC->GSM8K (100)", "1.5B ARC->GSM8K (150)"]
    forgetting = [100 * RUNS[k]["continual"]["average_forgetting"] for k in keys]
    ceiling = [100 * RUNS[k]["continual"]["task1_after_task1"] for k in keys]
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    y = np.arange(len(keys))
    ax.barh(y, forgetting, height=0.42, color=BLUE, zorder=3)
    ax.scatter(np.minimum(ceiling, 29), y, marker="|", s=160, color=INK2, zorder=4, linewidths=1.6)
    for yi, f, c in zip(y, forgetting, ceiling):
        lx = max(f, 0) + 0.4
        if abs(lx - c) < 1.4:
            lx = c + 0.4
        ax.text(lx, yi + 0.02, f"{f:+.2f} pp", va="center", fontsize=7.4, color=INK,
                bbox=dict(fc=SURFACE, ec="none", pad=0.2))
        if c < 29:
            ax.text(c + 0.5, yi - 0.28, f"max possible {c:.1f}", fontsize=6.4, color=INK2, va="center")
        else:
            ax.text(28.6, yi - 0.28, f"max possible {c:.1f} (off scale)", fontsize=6.4, color=INK2, va="center", ha="right")
    ax.axvline(10, color=INK, lw=1.0, zorder=2)
    ax.text(10.3, -0.72, "continuation rule: >= 10 pp prior-task loss", fontsize=7.2, color=INK)
    ax.set_ylim(len(keys) - 0.5, -0.95)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(y, labels, fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xlim(-2.5, 29.5)
    ax.set_xlabel("measured average forgetting of task 1 (percentage points); tick = task-1 score after its own training")
    clean_axes(ax, "x")
    ax.set_title("Standard RL-LoRA never reached the 10-point forgetting threshold", loc="left")
    return save(fig, "fig_6_2_continuation_rule")


# --------------------------------------------------------------------------------------
def fig_arc_gsm_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    cells = [("arc_challenge|after_arc_challenge", "ARC after\ntask 1"), ("arc_challenge|after_gsm8k", "ARC after\ntask 2"),
             ("gsm8k|after_gsm8k", "GSM8K after\ntask 2")]
    for ax, budget in zip(axes, ("100", "150")):
        x = np.arange(len(cells) + 1)
        width = 0.36
        for m_i, (method, key) in enumerate((("rl_lora", f"arc_base_{budget}"), ("hlora_rl", f"arc_hlora_{budget}"))):
            run = RUNS[key]
            vals, lo, hi = [], [], []
            for cell, _ in cells:
                sc = run["scores"][cell]
                vals.append(100 * sc["score"])
                lo.append(100 * (sc["score"] - sc["wilson95"][0]))
                hi.append(100 * (sc["wilson95"][1] - sc["score"]))
            fa = 100 * run["continual"]["final_average_performance"]
            vals.append(fa)
            lo.append(0)
            hi.append(0)
            pos = x + (m_i - 0.5) * width
            ax.bar(pos, vals, width=width * 0.94, color=METHOD_COLOR[method], zorder=3, label=METHOD_NAME[method])
            ax.errorbar(pos[:-1], vals[:-1], yerr=[lo[:-1], hi[:-1]], fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2.5, zorder=4)
            for p, v, h in zip(pos, vals, hi):
                ax.text(p, v + h + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.8, color=INK)
        ax.set_xticks(x, [c[1] for c in cells] + ["final\naverage"], fontsize=7.6)
        ax.set_title(f"({'a' if budget == '100' else 'b'}) {budget} GRPO updates per task (n = 64 per cell)", loc="left", fontsize=9)
        clean_axes(ax, "y")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("held-out accuracy (%)")
    axes[0].legend(loc="upper right", fontsize=7.4)
    fig.text(0.01, -0.03, "Error bars: Wilson 95% intervals over the 64 evaluated examples. One training seed; they do not capture seed-to-seed variation.",
             fontsize=7, color=INK2)
    return save(fig, "fig_6_3_arc_gsm_comparison")


# --------------------------------------------------------------------------------------
def fig_training_reward():
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.2), sharex=False)
    for ax, budget in zip(axes, ("100", "150")):
        for method, key in (("rl_lora", f"arc_base_{budget}"), ("hlora_rl", f"arc_hlora_{budget}")):
            s = series(key)["training"]
            steps = [x["global_step"] for x in s]
            rew = [x["raw_reward_mean"] for x in s]
            ax.plot(steps, rolling(rew, 15), color=METHOD_COLOR[method], lw=1.8, label=METHOD_NAME[method])
        n = int(budget)
        ax.axvline(n + 0.5, color=INK2, lw=0.9)
        ax.text(n * 0.5, 0.97, "task 1: ARC-Challenge (curves identical)", ha="center", va="top", fontsize=7.2, color=INK2, transform=ax.get_xaxis_transform())
        ax.text(n * 1.5, 0.97, "task 2: GSM8K", ha="center", va="top", fontsize=7.2, color=INK2, transform=ax.get_xaxis_transform())
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("training reward\n(15-update mean)")
        ax.set_title(f"({'a' if budget == '100' else 'b'}) {budget} updates per task", loc="left", fontsize=9)
        clean_axes(ax, "y")
    axes[1].set_xlabel("global optimizer step (8 rollouts per step)")
    axes[0].legend(loc="center right", fontsize=7.4)
    return save(fig, "fig_6_4_training_reward_curves")


# --------------------------------------------------------------------------------------
def fig_learning_signal():
    metrics = [("positive_reward_batches", "batches with positive reward"),
               ("zero_variance_group_fraction", "zero-variance prompt groups"),
               ("zero_gradient_updates", "updates with exactly zero gradient")]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=False)
    groups = [("100", "arc_challenge"), ("100", "gsm8k"), ("150", "arc_challenge"), ("150", "gsm8k")]
    xlabels = ["ARC\n100", "GSM8K\n100", "ARC\n150", "GSM8K\n150"]
    for ax, (metric, title) in zip(axes, metrics):
        x = np.arange(len(groups))
        for m_i, method in enumerate(("rl_lora", "hlora_rl")):
            vals = []
            for budget, task in groups:
                key = f"arc_base_{budget}" if method == "rl_lora" else f"arc_hlora_{budget}"
                t = RUNS[key]["training"][task]
                v = t[metric]
                if metric != "zero_variance_group_fraction":
                    v = v / t["updates"]
                vals.append(100 * v)
            pos = x + (m_i - 0.5) * 0.36
            ax.bar(pos, vals, width=0.34, color=METHOD_COLOR[method], zorder=3, label=METHOD_NAME[method])
            for p, v in zip(pos, vals):
                ax.text(p, v + 1.2, f"{v:.0f}", ha="center", fontsize=6.6, color=INK)
        ax.set_xticks(x, xlabels, fontsize=7.4)
        ax.set_title(title, loc="left", fontsize=8.6)
        ax.set_ylim(0, 110)
        clean_axes(ax, "y")
    axes[0].set_ylabel("percent")
    fig.legend(handles=[Patch(color=BLUE, label=METHOD_NAME["rl_lora"]), Patch(color=ORANGE, label=METHOD_NAME["hlora_rl"])],
               loc="lower center", ncol=2, fontsize=7.4, bbox_to_anchor=(0.5, -0.08))
    return save(fig, "fig_6_5_learning_signal_diagnostics")


# --------------------------------------------------------------------------------------
def fig_parse_failures():
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    for method, key in (("rl_lora", "arc_base_150"), ("hlora_rl", "arc_hlora_150")):
        s = series(key)["training"]
        steps = [x["global_step"] for x in s]
        pf = [x["parse_failures"] / 8 for x in s]
        ax.plot(steps, 100 * rolling(pf, 15), color=METHOD_COLOR[method], lw=1.8, label=METHOD_NAME[method])
    ax.axvline(150.5, color=INK2, lw=0.9)
    ax.text(75, 0.97, "ARC-Challenge", ha="center", va="top", fontsize=7.2, color=INK2, transform=ax.get_xaxis_transform())
    ax.text(225, 0.97, "GSM8K", ha="center", va="top", fontsize=7.2, color=INK2, transform=ax.get_xaxis_transform())
    ax.set_ylabel("unparseable responses (%)\n15-update mean")
    ax.set_xlabel("global optimizer step (150 updates per task)")
    ax.set_ylim(0, 100)
    clean_axes(ax, "y")
    ax.legend(loc="upper right", fontsize=7.4, bbox_to_anchor=(1.0, 0.86))
    ax.set_title("Answer-format learning: parse failures fall during ARC training", loc="left")
    return save(fig, "fig_6_6_parse_failure_trend")


# --------------------------------------------------------------------------------------
def fig_capabilities():
    caps = [("ifeval", "strict", "IFEval strict"), ("ifeval", "loose", "IFEval loose"), ("mmlu", "accuracy", "MMLU subset"),
            ("hellaswag", "accuracy", "HellaSwag"), ("arc_easy", "accuracy", "ARC-Easy")]
    fig, axes = plt.subplots(1, 5, figsize=(11, 3.1), sharey=True)
    stages = ["frozen_start", "arc_challenge", "gsm8k"]
    xt = ["frozen", "after\nARC", "after\nGSM8K"]
    for ax, (cap, metric, title) in zip(axes, caps):
        for m_i, (method, key) in enumerate((("rl_lora", "arc_base_150"), ("hlora_rl", "arc_hlora_150"))):
            rows = {r["after_task"]: r for r in RUNS[key]["capabilities"] if r["capability"] == cap and r["metric"] == metric}
            vals = [100 * rows[s]["score"] for s in stages]
            n = rows["frozen_start"]["n"]
            err = []
            for v in vals:
                lo, hi = wilson(round(v / 100 * n), n)
                err.append((v - 100 * lo, 100 * hi - v))
            xs = np.arange(3) + (m_i - 0.5) * 0.12
            ax.errorbar(xs, vals, yerr=np.array(err).T, color=METHOD_COLOR[method], lw=1.8, marker="o", ms=4.5,
                        mec=SURFACE, mew=1.0, capsize=2, elinewidth=0.8, label=METHOD_NAME[method])
        ax.set_xticks(range(3), xt, fontsize=7)
        ax.set_title(title, loc="left", fontsize=8.6)
        clean_axes(ax, "y")
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("score (%), n = 64 per point")
    handles = [Line2D([0], [0], color=BLUE, marker="o", lw=1.8, label=METHOD_NAME["rl_lora"]),
               Line2D([0], [0], color=ORANGE, marker="o", lw=1.8, label=METHOD_NAME["hlora_rl"])]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=7.6, bbox_to_anchor=(0.5, -0.07))
    return save(fig, "fig_6_7_general_capabilities")


# --------------------------------------------------------------------------------------
def fig_perplexity():
    keys = ["p05_order1_100", "p15_order1_100", "p15_order1_500", "p15_order2_500", "arc_base_100", "arc_hlora_100", "arc_base_150", "arc_hlora_150"]
    labels = ["RL-LoRA 0.5B GSM8K->ARC 100", "RL-LoRA 1.5B GSM8K->ARC 100", "RL-LoRA 1.5B GSM8K->ARC 500",
              "RL-LoRA 1.5B MBPP->GSM8K 500", "RL-LoRA 1.5B ARC->GSM8K 100", "HLoRA-RL 1.5B ARC->GSM8K 100",
              "RL-LoRA 1.5B ARC->GSM8K 150", "HLoRA-RL 1.5B ARC->GSM8K 150"]
    fig, ax = plt.subplots(figsize=(9.5, 3.6))
    for i, key in enumerate(keys):
        rows = [r for r in RUNS[key]["capabilities"] if r["capability"] == "perplexity"]
        rel = [100 * (r["score"] - r["frozen"]) / r["frozen"] for r in rows[1:]]
        color = METHOD_COLOR[RUNS[key]["method"]]
        ax.plot(rel, [i] * len(rel), color=AXIS, lw=0.9, zorder=2)
        ax.scatter(rel[:1], [i], color=color, marker="o", s=26, zorder=3, edgecolor=SURFACE, linewidth=0.8)
        ax.scatter(rel[1:], [i] * len(rel[1:]), color=color, marker="D", s=26, zorder=3, edgecolor=SURFACE, linewidth=0.8)
        ax.text(max(rel) + 0.12, i, f"{rel[-1]:+.2f}%", va="center", fontsize=6.8, color=INK)
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(range(len(keys)), labels, fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel("WikiText-2 perplexity change from the frozen model (%, higher is worse)")
    clean_axes(ax, "x")
    ax.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=GRAY, label="after task 1"),
                       Line2D([0], [0], marker="D", color="none", markerfacecolor=GRAY, label="after task 2")],
              loc="lower right", fontsize=7)
    ax.set_title("Held-out perplexity stayed within +3.6% of the frozen model", loc="left")
    return save(fig, "fig_6_8_perplexity_drift")


# --------------------------------------------------------------------------------------
def _layer_grid(consolidation):
    mods = ["q_proj", "k_proj", "v_proj", "o_proj"]
    grid = np.zeros((28, 4))
    for name, v in consolidation["importance_l2_by_layer"].items():
        layer = int(name.split("layers.")[1].split(".")[0])
        mod = name.rsplit(".", 1)[1]
        grid[layer, mods.index(mod)] = v
    return grid, mods


def fig_importance_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.6))
    for ax, (c_i, title) in zip(axes, ((0, "(a) after task 1 (ARC-Challenge)"), (1, "(b) after task 2 (GSM8K, accumulated)"))):
        cons = RUNS["arc_hlora_150"]["consolidations"][c_i]
        grid, mods = _layer_grid(cons)
        im = ax.imshow(np.log10(grid), cmap=BLUES, aspect="auto", vmin=2.4, vmax=4.4)
        ax.set_xticks(range(4), mods, fontsize=7.4)
        ax.set_yticks(range(0, 28, 3), [str(i) for i in range(0, 28, 3)], fontsize=7)
        ax.set_ylabel("decoder layer")
        ax.set_title(title, loc="left", fontsize=8.6)
        ax.tick_params(length=0)
        top = np.unravel_index(np.argmax(grid), grid.shape)
        ax.add_patch(plt.Rectangle((top[1] - 0.5, top[0] - 0.5), 1, 1, fill=False, ec=ORANGE, lw=2))
        ax.text(top[1] + 0.6, top[0] + 0.1, f"max {grid[top]:,.0f}", fontsize=6.8, color=INK, va="center")
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02)
    cb.set_label("log10 ||Omega_l||_2 (dense importance norm)", fontsize=7.4)
    cb.ax.tick_params(labelsize=7)
    fig.text(0.02, 0.01, "HLoRA-RL, 150 updates per task, Qwen2.5-1.5B, LoRA rank 8 on q/k/v/o of all 28 layers (112 modules).",
             fontsize=7, color=INK2)
    return save(fig, "fig_6_9_importance_heatmap")


def fig_layer_coefficients():
    cons = RUNS["arc_hlora_150"]["consolidations"][0]
    norms = np.array(sorted(cons["importance_l2_by_layer"].values(), reverse=True))
    soft = np.exp(norms - norms.max())
    soft = soft / soft.sum()
    share = norms / norms.sum()
    std = (norms - norms.mean()) / norms.std()
    soft_std = np.exp(std - std.max())
    soft_std /= soft_std.sum()
    eff = lambda w: 1.0 / np.sum(w ** 2)  # noqa: E731
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ranks = np.arange(1, len(norms) + 1)
    floor = 1e-6
    ax.semilogy(ranks, np.maximum(soft, floor), color=ORANGE, lw=2, marker="o", ms=3, label=f"softmax of raw norms (implemented): effective modules = {eff(soft):.1f}")
    ax.semilogy(ranks, share, color=BLUE, lw=2, label=f"norm share ||Omega_l|| / sum (alternative): effective modules = {eff(share):.1f}")
    ax.semilogy(ranks, np.maximum(soft_std, floor), color=GRAY, lw=2, label=f"softmax of standardized norms (alternative): effective modules = {eff(soft_std):.1f}")
    ax.set_ylim(floor * 0.7, 1.5)
    ax.text(len(norms) * 0.98, floor * 1.25, "values below 1e-6 drawn at the floor (softmax gives exactly 0 in FP32)", ha="right", fontsize=6.8, color=INK2)
    ax.set_xlabel("module rank by importance norm (1 = largest of 112)")
    ax.set_ylabel("layer coefficient (log scale)")
    clean_axes(ax, "y")
    ax.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.0, 0.93))
    ax.set_title("Static softmax layer weights collapse to a single protected module", loc="left")
    return save(fig, "fig_6_10_layer_coefficient_collapse"), {"eff_softmax": eff(soft), "eff_share": eff(share), "eff_soft_std": eff(soft_std)}


# --------------------------------------------------------------------------------------
def fig_penalty_vs_rl():
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True)
    for ax, budget in zip(axes, ("100", "150")):
        s = [x for x in series(f"arc_hlora_{budget}")["training"] if x["task_id"] == "gsm8k"]
        steps = np.arange(1, len(s) + 1)
        reg = np.array([x["regularization_loss"] for x in s])
        rl = np.array([abs(x["rl_loss"]) for x in s])
        ax.semilogy(steps, np.where(rl > 0, rl, np.nan), color=BLUE, lw=0, marker="o", ms=3, label="|GRPO loss| (nonzero steps)")
        ax.semilogy(steps, np.where(reg > 0, reg, np.nan), color=ORANGE, lw=1.8, label="consolidation penalty (lambda = 1)")
        zero = steps[rl == 0]
        ax.scatter(zero, np.full(len(zero), 2e-7), marker="|", color=GRAY, s=40, label="GRPO loss exactly 0 (zero-variance batch)")
        ax.set_ylim(1e-7, 1e-1)
        ax.set_xlabel("GSM8K update (task 2)")
        ax.set_title(f"({'a' if budget == '100' else 'b'}) {budget} updates per task", loc="left", fontsize=9)
        clean_axes(ax, "y")
    axes[0].set_ylabel("loss magnitude (log scale)")
    axes[1].legend(loc="upper right", fontsize=6.8)
    return save(fig, "fig_6_11_penalty_vs_rl_loss")


# --------------------------------------------------------------------------------------
def fig_trajectory_divergence():
    b = series("arc_base_150")["training"]
    h = series("arc_hlora_150")["training"]
    steps = np.array([x["global_step"] for x in b])
    gb = np.array([x["gradient_norm"] for x in b])
    gh = np.array([x["gradient_norm"] for x in h])
    rb = np.cumsum([x["raw_reward_mean"] for x in b])
    rh = np.cumsum([x["raw_reward_mean"] for x in h])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))
    ax = axes[0]
    ax.plot(steps, np.abs(gh - gb), color=INK2, lw=1.2)
    ax.axvline(150.5, color=AXIS, lw=0.8)
    first = R["comparisons"]["150"]["trajectory"]["first_differing_global_step"]
    ax.annotate(f"first difference at step {first}\n(second GSM8K update)", xy=(first, 0), xytext=(40, 0.6),
                fontsize=7, color=INK, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    ax.text(75, 0.93, "identical to machine precision", transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color=INK2)
    ax.set_xlabel("global optimizer step")
    ax.set_ylabel("|grad-norm difference|")
    ax.set_title("(a) Per-step gradient-norm difference between runs", loc="left", fontsize=9)
    clean_axes(ax, "y")
    ax = axes[1]
    ax.plot(steps, rb, color=BLUE, lw=1.8, label=METHOD_NAME["rl_lora"])
    ax.plot(steps, rh, color=ORANGE, lw=1.8, label=METHOD_NAME["hlora_rl"])
    ax.axvline(150.5, color=AXIS, lw=0.8)
    ax.set_xlabel("global optimizer step")
    ax.set_ylabel("cumulative mean training reward")
    ax.set_title("(b) Cumulative training reward", loc="left", fontsize=9)
    ax.legend(loc="upper left", fontsize=7)
    clean_axes(ax, "y")
    return save(fig, "fig_6_12_trajectory_divergence")


# --------------------------------------------------------------------------------------
def fig_storage():
    st = R["storage"]["Qwen2.5-1.5B-Instruct"]
    items = [
        ("Frozen backbone weights (1.54 B params, BF16)", 1.54e9 * 2),
        ("Dense importance, HLoRA-RL (implemented)", st["dense_importance_fp32_bytes"]),
        ("Trainable LoRA adapter, rank 8 (FP32)", st["lora_fp32_bytes"]),
        ("Reference LoRA factors (FP32)", st["lora_fp32_bytes"]),
        ("Rank-1 factorized importance, RAHC-LoRA (planned)", st["rank1_factorized_fp32_bytes"]),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 3.0))
    y = np.arange(len(items))
    colors = [GRAY, ORANGE, BLUE, BLUE, "#1baf7a"]
    ax.barh(y, [v for _, v in items], color=colors, height=0.5, zorder=3)
    ax.set_xscale("log")
    for yi, (_, v) in zip(y, items):
        label = f"{v / 1e9:.2f} GB" if v >= 1e9 else f"{v / 1e6:.2f} MB"
        ax.text(v * 1.15, yi, label, va="center", fontsize=7.4, color=INK)
    ax.set_yticks(y, [n for n, _ in items], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlim(3e5, 2e10)
    ax.set_xlabel("persistent bytes (log scale), Qwen2.5-1.5B, q/k/v/o in 28 layers")
    clean_axes(ax, "x")
    ax.set_title(f"Dense importance is {st['dense_importance_fp32_bytes'] / st['lora_fp32_bytes']:.0f}x the adapter; rank-1 factors are {st['dense_to_rank1_ratio']:.0f}x smaller than dense",
                 loc="left", fontsize=9)
    return save(fig, "fig_6_14_storage_comparison")


# --------------------------------------------------------------------------------------
def fig_compute_profile():
    keys = ["p05_order1_100", "p15_order1_100", "p15_order1_500", "p15_order2_500", "arc_base_100", "arc_hlora_100", "arc_base_150", "arc_hlora_150"]
    labels = ["RL-LoRA 0.5B G->A 100", "RL-LoRA 1.5B G->A 100", "RL-LoRA 1.5B G->A 500", "RL-LoRA 1.5B M->G 500",
              "RL-LoRA 1.5B A->G 100", "HLoRA-RL 1.5B A->G 100", "RL-LoRA 1.5B A->G 150", "HLoRA-RL 1.5B A->G 150"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    y = np.arange(len(keys))
    metrics = [("mean_step_seconds", "mean seconds per update", "{:.1f}"), ("peak_gpu_allocated_gb", "peak GPU memory allocated (GB)", "{:.1f}"),
               ("total_optimizer_hours", "total optimizer time (hours)", "{:.1f}")]
    for ax, (m, title, fmt) in zip(axes, metrics):
        vals = [RUNS[k]["system"][m] for k in keys]
        colors = [METHOD_COLOR[RUNS[k]["method"]] for k in keys]
        ax.barh(y, vals, color=colors, height=0.55, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v * 1.02, yi, fmt.format(v), va="center", fontsize=6.8, color=INK)
        ax.set_title(title, loc="left", fontsize=8.6)
        ax.set_xlim(0, max(vals) * 1.22)
        clean_axes(ax, "x")
    axes[0].set_yticks(y, labels, fontsize=7.2)
    axes[0].invert_yaxis()
    fig.legend(handles=[Patch(color=BLUE, label=METHOD_NAME["rl_lora"]), Patch(color=ORANGE, label=METHOD_NAME["hlora_rl"])],
               loc="lower center", ncol=2, fontsize=7.4, bbox_to_anchor=(0.5, -0.07))
    return save(fig, "fig_6_13_compute_profile")


def main():
    extra = {}
    fig_phase2_matrices()
    fig_continuation_rule()
    fig_arc_gsm_comparison()
    fig_training_reward()
    fig_learning_signal()
    fig_parse_failures()
    fig_capabilities()
    fig_perplexity()
    fig_importance_heatmap()
    _, extra["coefficients"] = fig_layer_coefficients()
    fig_penalty_vs_rl()
    fig_trajectory_divergence()
    fig_storage()
    fig_compute_profile()
    (DATA / "chart_derived_numbers.json").write_text(json.dumps(extra, indent=2), encoding="utf-8")
    print(extra)


if __name__ == "__main__":
    main()

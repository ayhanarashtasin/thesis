"""Conceptual and architecture diagrams for the P2 thesis (static PNG, 300 dpi)."""

from __future__ import annotations

import numpy as np
from figstyle import (
    ALERT, AXIS, BLUE, BLUEBOX, EVID, FROZEN, GRAY, GRID, INK, INK2, NEUTRAL, ORANGE,
    PERSIST, SURFACE, TRAIN, arrow, box, canvas, clean_axes, plt, save, status_chip,
)
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


def lane(ax, x, y, w, h, label, color="#f7f7f5", edge=GRID, size=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.5",
                                fc=color, ec=edge, lw=0.9, zorder=0))
    ax.text(x + 1.0, y + h - 0.9, label, fontsize=size, fontweight="bold", color=INK2,
            va="top", ha="left", zorder=1)


# --------------------------------------------------------------------------------------
def fig_research_roadmap():
    fig, ax = canvas(11, 5.0, 100, 46)
    stages = [
        ("1  Literature review (P1)", "24 papers on catastrophic forgetting; gap: forgetting in sequential RL with one shared LoRA", "Complete"),
        ("2  Method specification", "RAHC-LoRA equations, protocol, success criteria, phased plan", "Complete"),
        ("3  Phases 0-1: testbed", "Typed config, 4 tasks, deterministic rewards, sandbox, evaluation matrix", "Complete"),
        ("4  Phase 2: RL-LoRA baseline", "GRPO trainer, checkpoint/resume, 4 GPU pilots (0.5B and 1.5B)", "Complete"),
        ("5  Phase 3: HLoRA-to-RL", "Dense path-integral importance + static layer weights; ARC->GSM8K pilots", "Partial"),
        ("6  Phases 4-7: RAHC parts", "Reward-aware importance, rank-1 factorization, anchor KL, conflict gate, dual controller", "Planned"),
        ("7  Phase 8: method pilot", "RL-LoRA, replay, HLoRA-RL, RAHC-LoRA; >= 2 orders x 3 seeds; continuation rule", "Planned"),
        ("8  Phase 9: main study", "3 orders x 3 seeds, all baselines and ablations, paired statistics", "Planned"),
    ]
    roles = {"Complete": EVID, "Partial": TRAIN, "Planned": NEUTRAL}
    w, h = 22.0, 13.5
    xs = [1.5, 26.0, 50.5, 75.0]
    top_y, bot_y = 25.5, 3.0
    for i, (title, body, status) in enumerate(stages):
        row = i // 4
        col = i % 4 if row == 0 else 3 - (i % 4)
        x = xs[col]
        y = top_y if row == 0 else bot_y
        box(ax, x, y, w, h, title, body, role=roles[status], wrap=34, title_size=7.6,
            body_size=6.6)
        status_chip(ax, x + 1.3, y + 1.6, status)
    for c in range(3):
        arrow(ax, (xs[c] + w, top_y + h / 2), (xs[c + 1], top_y + h / 2))
        arrow(ax, (xs[c + 1], bot_y + h / 2), (xs[c] + w, bot_y + h / 2))
    arrow(ax, (xs[3] + w / 2, top_y), (xs[3] + w / 2, bot_y + h))
    # scope brackets
    ax.plot([26.0, 97.0], [42.0, 42.0], color=BLUE, lw=1.4)
    ax.text(61.5, 42.8, "P2 scope (this report): specification, testbed, baselines, pilot evidence",
            ha="center", va="bottom", fontsize=7.4, color=INK)
    ax.plot([1.5, 97.0], [1.2, 1.2], color=GRAY, lw=1.0)
    ax.text(1.5, 0.0, "Stages 6-8 form the final-thesis (P3) plan; start of stage 7 is gated by the pilot continuation rule",
            ha="left", va="top", fontsize=7.0, color=INK2)
    return save(fig, "fig_1_1_research_roadmap")


# --------------------------------------------------------------------------------------
def fig_cl_taxonomy():
    fig, ax = canvas(11, 5.4, 100, 50)
    box(ax, 30, 42, 40, 6.5, "Mitigating catastrophic forgetting in LLM fine-tuning",
        role=BLUEBOX, title_size=8.6)
    families = [
        ("Regularization", ["EWC, SI, MAS", "RecAdam", "HLoRA / HLER", "EWC-LoRA"]),
        ("Replay", ["Experience replay", "LAMOL", "MbPA++", "Self-synthesized", "rehearsal"]),
        ("Architecture / PEFT", ["Progressive nets", "Adapters, prompts", "O-LoRA, InfLoRA", "I-LoRA"]),
        ("Gradient geometry", ["GEM, A-GEM", "PCGrad", "Orthogonal", "subspaces", "Geometry conflict"]),
        ("Behavioural", ["LwF distillation", "Policy distillation", "KL to reference", "Policy consolidation"]),
        ("RL-specific", ["RL forgets less", "Sparse RL updates", "RFT continual", "post-training", "CPO prior-task KL"]),
    ]
    w, gap = 15.2, 1.4
    x0 = 1.0
    for i, (title, items) in enumerate(families):
        x = x0 + i * (w + gap)
        box(ax, x, 19.5, w, 15.5, title, items, role=NEUTRAL, title_size=7.4, body_size=6.5)
        arrow(ax, (50, 42), (x + w / 2, 35.2), color=AXIS, lw=0.9)
    box(ax, 10, 1.5, 80, 11.5, "RAHC-LoRA position (this thesis)",
        ["One frozen backbone and ONE shared LoRA policy (no per-task adapters)",
         "Parameter protection: reward-aware, Hessian-free importance on DeltaW = (alpha/r)BA, stored as rank-1 factors",
         "Behavioural protection: fixed-budget anchor memory, compact top-k KL, adaptive dual multiplier, layer conflict gate"],
        role=TRAIN, title_size=8.0, body_size=6.8)
    for i in (0, 3, 4, 5):
        x = x0 + i * (w + gap) + w / 2
        arrow(ax, (x, 19.5), (min(max(x, 14), 86), 13.1), color=ORANGE, lw=1.0)
    ax.text(50, 16.3, "RAHC-LoRA combines", ha="center", fontsize=6.8, color=INK2,
            bbox=dict(fc=SURFACE, ec="none", pad=0.3))
    return save(fig, "fig_2_2_cl_taxonomy")


# --------------------------------------------------------------------------------------
def fig_evaluation_protocol():
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    tasks = ["T1", "T2", "T3", "T4"]
    grid = np.full((4, 4), np.nan)
    rng = np.random.default_rng(3)
    for i in range(4):
        for j in range(i + 1):
            grid[i, j] = 0.35 + 0.4 * rng.random()
    axl.imshow(np.where(np.isnan(grid), 0, grid), cmap="Blues", vmin=0, vmax=1.2)
    for i in range(4):
        for j in range(4):
            if j <= i:
                axl.text(j, i, f"A[{i+1},{j+1}]", ha="center", va="center", fontsize=7.5, color=INK)
            else:
                axl.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="#f4f4f2", ec=SURFACE, lw=2))
                axl.text(j, i, "future\n(missing)", ha="center", va="center", fontsize=6.2, color=GRAY)
    axl.set_xticks(range(4), [f"eval {t}" for t in tasks])
    axl.set_yticks(range(4), [f"after {t}" for t in tasks])
    axl.set_title("(a) Performance matrix A[i, j]", loc="left")
    for s in axl.spines.values():
        s.set_visible(False)
    axl.tick_params(length=0)
    axr.axis("off")
    axr.set_title("(b) Metrics derived from the full matrix", loc="left")
    lines = [
        (r"Average forgetting (primary, lower is better)", 11.0, True),
        (r"$F = \frac{1}{T-1}\sum_{j=1}^{T-1}\left[\max_{j\leq i<T} A_{i,j} - A_{T,j}\right]$", 9.5, False),
        (r"Final average performance (co-primary, higher is better)", 11.0, True),
        (r"$\bar{A}_T = \frac{1}{T}\sum_{j=1}^{T} A_{T,j}$", 9.5, False),
        (r"Backward transfer", 11.0, True),
        (r"$BWT = \frac{1}{T-1}\sum_{j=1}^{T-1}\left(A_{T,j} - A_{j,j}\right)$", 9.5, False),
        (r"Plasticity: current-task learning $A_{t,t}$; capability drift vs frozen model", 11.0, True),
    ]
    y = 0.95
    for text, _, bold in lines:
        axr.text(0.0, y, text, transform=axr.transAxes, fontsize=8.2 if bold else 10.5,
                 fontweight="bold" if bold else "normal", color=INK if bold else INK2, va="top")
        y -= 0.12 if bold else 0.17
    return save(fig, "fig_2_1_evaluation_protocol")


# --------------------------------------------------------------------------------------
def poly_arrow(ax, points, color=INK2, lw=1.1, label=None, label_at=None, label_size=6.6):
    """Orthogonal polyline with an arrow head on the final segment."""
    xs = [pt[0] for pt in points[:-1]]
    ys = [pt[1] for pt in points[:-1]]
    ax.plot(xs + [points[-1][0]], ys + [points[-1][1]], color=color, lw=lw, zorder=1,
            solid_capstyle="round")
    arrow(ax, points[-2], points[-1], color=color, lw=lw)
    if label and label_at:
        ax.text(label_at[0], label_at[1], label, fontsize=label_size, color=INK2, ha="center",
                va="center", zorder=4, bbox=dict(fc=SURFACE, ec="none", pad=0.4))


def fig_system_architecture():
    fig, ax = canvas(12, 7.6, 100, 63)
    # ---- left column
    box(ax, 1, 46, 12.5, 12, "Current task", ["training split", "only"], role=EVID)
    ax.text(1.2, 42.5, "Legend", fontsize=7.2, fontweight="bold", color=INK)
    for k, (name, role) in enumerate([("frozen", FROZEN), ("trainable", TRAIN), ("current-task learning", BLUEBOX),
                                      ("retention state", PERSIST), ("evaluation", EVID), ("objective", ALERT)]):
        y = 38.5 - k * 3.2
        ax.add_patch(FancyBboxPatch((1.2, y), 2.4, 2.0, boxstyle="round,pad=0,rounding_size=0.4", fc=role[0], ec=role[1], lw=0.9))
        ax.text(4.3, y + 1.0, name, fontsize=6.6, color=INK2, va="center")
    # ---- policy band
    ax.add_patch(FancyBboxPatch((16, 38.3), 27, 5.2, boxstyle="round,pad=0,rounding_size=0.8", fc=FROZEN[0], ec=FROZEN[1], lw=1.1))
    ax.add_patch(FancyBboxPatch((43.5, 38.3), 26.5, 5.2, boxstyle="round,pad=0,rounding_size=0.8", fc=TRAIN[0], ec=TRAIN[1], lw=1.1))
    ax.text(29.5, 40.9, "Frozen backbone W0 (no gradients)", ha="center", va="center", fontsize=7.0, fontweight="bold", color=INK)
    ax.text(56.75, 40.9, "One shared LoRA: DeltaW = (alpha/r) B A", ha="center", va="center", fontsize=7.0, fontweight="bold", color=INK)
    # ---- learning block
    lane(ax, 16, 46, 54, 13, "Current-task learning (GRPO, verifiable rewards)")
    inner = [(17, "Rollouts", "G = 4 per prompt"), (30.5, "Rewards", "deterministic [0,1]"),
             (44, "Advantages", "group-norm, clip"), (57.5, "GRPO loss", "L_RL, g_RL")]
    for x, t, b in inner:
        box(ax, x, 47.3, 11.8, 7.8, t, [b], role=BLUEBOX, title_size=7.4, body_size=6.4)
    for i in range(3):
        arrow(ax, (inner[i][0] + 11.8, 51.2), (inner[i + 1][0], 51.2))
    # ---- retention block
    lane(ax, 16, 20.8, 54, 15.2, "Old-task retention")
    ax.text(69, 35.1, "all terms zero on task 1", fontsize=6.3, color=INK2, ha="right", va="top")
    ra = [(17, "Anchor memory", "fixed global budget"), (34.5, "Compact KL", "top-k refs + residual"),
          (52, "Dual controller", "adaptive lambda_f")]
    rb = [(17, "Reference + importance", "rank-1 p_l, q_l"), (34.5, "Penalty R_l", "effective-weight, Gram"),
          (52, "Conflict gate", "c_l, detached")]
    for x, t, b in ra:
        box(ax, x, 27.6, 16.5, 5.6, t, [b], role=PERSIST, title_size=7.0, body_size=6.2, radius=0.8)
    for x, t, b in rb:
        box(ax, x, 21.6, 16.5, 5.6, t, [b], role=PERSIST, title_size=7.0, body_size=6.2, radius=0.8)
    for row, yy in ((ra, 30.4), (rb, 24.4)):
        for i in range(2):
            arrow(ax, (row[i][0] + 16.5, yy), (row[i + 1][0], yy))
    # ---- update block
    lane(ax, 74, 20.8, 25, 38.2, "One optimizer update")
    ub = [("Total objective", "L_RL + lambda_f KL + lambda_p sum c_l R_l", ALERT),
          ("Gradient queries", "g_RL, g_anchor -> c_l (detached)", PERSIST),
          ("AdamW step", "clip norm 1.0; A, B only", TRAIN),
          ("Observe movement", "actual post-step DeltaW change", NEUTRAL),
          ("Update state", "reward-aware importance; lambda_f", PERSIST)]
    for k, (t, b, role) in enumerate(ub):
        y = 50.3 - k * 7.1
        box(ax, 75, y, 23, 5.9, t, [b], role=role, title_size=7.0, body_size=6.1, radius=0.8)
        if k:
            arrow(ax, (86.5, y + 7.1), (86.5, y + 5.9))
    # ---- block arrows
    arrow(ax, (13.5, 52), (17, 52))
    arrow(ax, (29.5, 43.5), (29.5, 47.3), label="generates", label_pos=0.5, label_offset=(4.2, -1.0))
    arrow(ax, (42.75, 38.3), (42.75, 33.2))
    ax.text(41.8, 35.6, "policy on anchors", fontsize=6.3, color=INK2, ha="right", va="center")
    arrow(ax, (69.3, 51.2), (75, 53.2), label="L_RL", label_pos=0.45, label_offset=(0, 0.4))
    ax.plot([68.5, 72.2], [30.4, 30.4], color=INK2, lw=1.1)
    ax.plot([68.5, 72.2, 72.2], [24.4, 24.4, 51.4], color=INK2, lw=1.1)
    arrow(ax, (72.2, 51.4), (75, 51.4))
    ax.text(73.0, 27.4, "lambda_f KL + sum c_l R_l", fontsize=5.8, color=INK2, ha="left",
            va="center", rotation=90)
    import matplotlib.patheffects as pe
    upd = FancyArrowPatch((75, 38.8), (70.2, 38.8), arrowstyle="-|>", mutation_scale=9, color=ORANGE, lw=1.5, zorder=6,
                          path_effects=[pe.Stroke(linewidth=4, foreground=SURFACE), pe.Normal()])
    ax.add_patch(upd)
    ax.text(72.6, 37.3, "updates A, B", fontsize=6.2, color=INK2, ha="center", va="top", zorder=7,
            bbox=dict(fc=SURFACE, ec="none", pad=0.3))
    # ---- boundary lane
    lane(ax, 1, 1, 98, 18.6, "Task boundary")
    bx = [(3, "Consolidate", "merge + factorize importance; snapshot reference", PERSIST),
          (25, "Refresh anchors", "eligible training prompts; global budget", PERSIST),
          (47, "Evaluate", "all learned tasks -> A[i, j]; capability suite", EVID),
          (69, "Atomic checkpoint", "policy, optimizer, retention, RNG, manifests", PERSIST)]
    for x, t, b, role in bx:
        box(ax, x, 2.3, 20, 8.6, t, [b], role=role, wrap=30, title_size=7.4, body_size=6.2)
    for i in range(3):
        arrow(ax, (bx[i][0] + 20, 6.6), (bx[i + 1][0], 6.6))
    box(ax, 47, 12.0, 20, 3.8, "Held-out tests + capability suite", role=EVID, title_size=6.4, bold=True, radius=0.6)
    arrow(ax, (57, 12.0), (57, 10.9))
    poly_arrow(ax, [(80, 20.8), (80, 17.2), (18, 17.2), (18, 10.9)], label="end of task t", label_at=(36, 17.2))
    poly_arrow(ax, [(89, 6.6), (99.3, 6.6), (99.3, 61.3), (7.25, 61.3), (7.25, 58)], label="next task t+1", label_at=(53, 61.3))
    return save(fig, "fig_4_1_rahc_lora_architecture")


# --------------------------------------------------------------------------------------
def fig_gauge_invariance():
    rng = np.random.default_rng(7)
    o_dim, i_dim, r, alpha = 64, 48, 8, 16.0
    s = alpha / r
    a_ref = rng.normal(0, 0.1, (r, i_dim))
    b_ref = rng.normal(0, 0.1, (o_dim, r))
    a_cur = a_ref + rng.normal(0, 0.02, (r, i_dim))
    b_cur = b_ref + rng.normal(0, 0.02, (o_dim, r))
    ks = np.logspace(-1.5, 1.5, 61)
    factor_pen, eff_pen = [], []
    ref_eff = s * b_ref @ a_ref
    for k in ks:
        a_k, b_k = k * a_cur, b_cur / k
        factor_pen.append(np.sum((a_k - a_ref) ** 2) + np.sum((b_k - b_ref) ** 2))
        eff_pen.append(np.sum((s * b_k @ a_k - ref_eff) ** 2))
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(10.5, 3.4), gridspec_kw={"width_ratios": [1.05, 1]})
    axl.axis("off")
    axl.set_xlim(0, 100)
    axl.set_ylim(0, 60)
    box(axl, 1, 22, 17, 30, "B", ["O x r", "up-proj."], role=TRAIN)
    axl.text(21, 37, "x", fontsize=12, ha="center", va="center", color=INK2)
    box(axl, 24, 37, 30, 15, "A", ["r x I down-projection"], role=TRAIN)
    arrow(axl, (55, 40), (61, 40))
    box(axl, 62, 22, 37, 30, r"$\Delta W = (\alpha/r)\,BA$", ["O x I effective update", "(the protected object)"], role=BLUEBOX, title_size=8.4)
    axl.text(1, 14, "Rescaling  A' = kA,  B' = B/k  leaves  B'A' = BA  unchanged.", fontsize=8, color=INK)
    axl.text(1, 8, "A penalty on A and B separately changes with k;", fontsize=7.6, color=INK2)
    axl.text(1, 2.5, "a penalty on DeltaW does not (gauge invariance).", fontsize=7.6, color=INK2)
    axl.set_title("(a) LoRA factorization and its rescaling symmetry", loc="left")
    axr.loglog(ks, factor_pen, color=BLUE, lw=2, label="factor-level penalty ||A'-A_ref||^2 + ||B'-B_ref||^2")
    axr.loglog(ks, np.maximum(eff_pen, 1e-12), color=ORANGE, lw=2, label="effective-weight penalty ||DeltaW' - DeltaW_ref||^2")
    axr.axvline(1.0, color=AXIS, lw=0.8)
    axr.set_xlabel("rescaling factor k")
    axr.set_ylabel("penalty value (log scale)")
    axr.set_title("(b) Numerical check on random 64x8 and 8x48 factors", loc="left")
    clean_axes(axr, "y")
    axr.legend(loc="upper center", fontsize=7, bbox_to_anchor=(0.5, -0.22), ncol=1)
    rel = (max(eff_pen) - min(eff_pen)) / np.mean(eff_pen)
    axr.text(0.5, 0.93, f"effective-penalty relative spread over k: {rel:.1e}", transform=axr.transAxes,
             fontsize=7, color=INK2, ha="center")
    return save(fig, "fig_4_2_gauge_invariance"), rel


# --------------------------------------------------------------------------------------
def fig_grpo_pipeline():
    fig, ax = canvas(11, 4.0, 100, 44)
    box(ax, 1, 16, 13, 14, "Prompt batch", ["2 prompts / update", "current task only"], role=EVID)
    box(ax, 18, 16, 15, 14, "Sampling", ["G = 4 responses", "T = 1.0, top-p = 1.0", "<= 128 new tokens"], role=BLUEBOX)
    box(ax, 37, 16, 15, 14, "Verifier reward", ["exact numeric / option", "tests passed fraction", "constraint fraction"], role=BLUEBOX)
    box(ax, 56, 16, 17, 14, "Group advantage", ["A_i = (r_i - mean) / std", "population std, FP32", "std <= eps  =>  A = 0", "clip |A| <= 5"], role=BLUEBOX)
    box(ax, 77, 16, 21.5, 14, "Clipped surrogate", ["ratio = exp(logp_new - logp_old)", "min(ratio A, clip(ratio, 0.8, 1.2) A)", "token mean over response", "one AdamW step per rollout"], role=TRAIN)
    for a, b in ((14, 18), (33, 37), (52, 56), (73, 77)):
        arrow(ax, (a, 23), (b, 23))
    ax.text(1, 41.5, "Worked examples (one prompt group, G = 4)", fontsize=8, fontweight="bold", color=INK)
    ax.text(1, 37.2, "Informative group:  rewards [1, 0, 0, 1]  ->  advantages [+1, -1, -1, +1]  ->  gradient pushes towards rewarded responses",
            fontsize=7.4, color=INK2)
    ax.text(1, 33.4, "Zero-variance group:  rewards [0, 0, 0, 0] or [1, 1, 1, 1]  ->  advantages [0, 0, 0, 0]  ->  no gradient (logged, never replaced by a heuristic)",
            fontsize=7.4, color=INK2)
    ax.text(1, 10.5, "Reward rules: deterministic, documented range [0, 1], parse failures score 0, raw and normalized rewards logged,",
            fontsize=7.2, color=INK2)
    ax.text(1, 6.8, "generated code executed only in the network-disabled Docker sandbox; every value checked for finiteness before the optimizer step.",
            fontsize=7.2, color=INK2)
    return save(fig, "fig_4_3_grpo_pipeline")


# --------------------------------------------------------------------------------------
def fig_update_sequence():
    lanes = ["Trainer", "Policy / RL", "Rewards", "Retention", "Optimizer"]
    steps = [
        (0, "1  Sample a current-task minibatch"),
        (1, "2  Generate G grouped responses; store old log-probs"),
        (2, "3  Deterministic rewards (sandbox for code)"),
        (1, "4  Group-normalize and clip advantages"),
        (1, "5  Current GRPO loss; capture effective RL gradient"),
        (3, "6  If memory non-empty: anchor KL and anchor gradient"),
        (3, "7  Detached conflicts c_l; penalty R_l vs reference"),
        (0, "8  Combine objective; snapshot pre-step factors"),
        (4, "9  Backward, clip norm 1.0, AdamW step (LoRA only)"),
        (3, "10  Observe actual DeltaW movement"),
        (3, "11  Update importance statistics and lambda_f"),
        (0, "12  Finite checks, structured metrics, release buffers"),
    ]
    fig, ax = canvas(11, 6.6, 100, 66)
    lane_w = 19.2
    for i, name in enumerate(lanes):
        x = 1 + i * (lane_w + 0.5)
        ax.add_patch(Rectangle((x, 0.5), lane_w, 60.5, fc="#f7f7f5" if i % 2 == 0 else SURFACE, ec=GRID, lw=0.7, zorder=0))
        ax.text(x + lane_w / 2, 63.5, name, ha="center", va="center", fontsize=8.5, fontweight="bold", color=INK)
    prev = None
    for k, (lane_i, text) in enumerate(steps):
        x = 1 + lane_i * (lane_w + 0.5) + 0.6
        y = 56.2 - k * 4.75
        role = {0: NEUTRAL, 1: BLUEBOX, 2: EVID, 3: PERSIST, 4: TRAIN}[lane_i]
        wbox = lane_w - 1.2
        ax.add_patch(FancyBboxPatch((x, y), wbox, 3.7, boxstyle="round,pad=0,rounding_size=0.8",
                                    fc=role[0], ec=role[1], lw=0.9, zorder=2))
        import textwrap as _tw
        ax.text(x + wbox / 2, y + 1.85, "\n".join(_tw.wrap(text, 30)), ha="center", va="center",
                fontsize=5.7, color=INK, zorder=3, linespacing=1.1)
        if prev is not None:
            px, py = prev
            arrow(ax, (px + wbox / 2, py), (x + wbox / 2, y + 3.7), lw=0.8)
        prev = (x, y)
    return save(fig, "fig_4_5_update_sequence")


# --------------------------------------------------------------------------------------
def fig_task_lifecycle():
    fig, ax = canvas(11, 3.9, 100, 38)
    states = [(1, "Validate", "typed config;\nfail closed"), (15, "Initialize", "load pinned model;\nattach one LoRA"),
              (29, "Train task t", "GRPO updates on\ntrain split only"), (43, "Consolidate", "importance and\nreference state"),
              (57, "Evaluate", "all learned tasks +\ncapability suite"), (71, "Checkpoint", "atomic, SHA-256\nmanifest"),
              (85, "Aggregate", "matrices, CIs,\npaired tests")]
    for x, t, b in states:
        role = EVID if t in ("Evaluate", "Aggregate") else (TRAIN if t == "Train task t" else NEUTRAL)
        box(ax, x, 16, 12.5, 12, t, [b], role=role, title_size=7.8, body_size=6.5)
    for i in range(len(states) - 1):
        arrow(ax, (states[i][0] + 12.5, 22), (states[i + 1][0], 22))
    for xy in ((77.25, 16, 77.25, 9.5), (77.25, 9.5, 35.25, 9.5)):
        ax.plot([xy[0], xy[2]], [xy[1], xy[3]], color=INK2, lw=1.1)
    arrow(ax, (35.25, 9.5), (35.25, 16))
    ax.text(56, 9.5, "another task remains", fontsize=6.8, color=INK2, ha="center", va="center",
            bbox=dict(fc=SURFACE, ec="none", pad=0.4))
    ax.text(29, 34.5, "Task 1: memory and consolidated state are empty, so every retention term is exactly zero",
            fontsize=7.4, color=INK)
    ax.text(29, 31.2, "and the update equals standard RL-LoRA (verified: identical task-1 losses in paired pilots).",
            fontsize=7.4, color=INK2)
    return save(fig, "fig_4_6_task_lifecycle")


# --------------------------------------------------------------------------------------
def fig_factorized_penalty():
    rng = np.random.default_rng(11)
    o, i = 24, 32
    p = rng.gamma(1.5, 1.0, o)
    q = rng.gamma(1.5, 1.0, i)
    dense = np.outer(p, q) * rng.gamma(4.0, 0.25, (o, i))
    # rank-one from marginals (mass preserving): p_hat = row sums, q_hat = col sums / total
    total = dense.sum()
    p_hat = dense.sum(axis=1)
    q_hat = dense.sum(axis=0) / total
    approx = np.outer(p_hat, q_hat)
    corr = np.corrcoef(dense.ravel(), approx.ravel())[0, 1]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    vmax = max(dense.max(), approx.max())
    axes[0].imshow(dense, cmap="Blues", vmin=0, vmax=vmax)
    axes[0].set_title("(a) Dense importance S (O x I)", loc="left")
    axes[1].imshow(approx, cmap="Blues", vmin=0, vmax=vmax)
    axes[1].set_title("(b) Rank-1 factors p q^T", loc="left")
    for a in axes[:2]:
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_color(AXIS)
    axes[0].text(0.0, -0.06, f"Toy 24 x 32 example: Pearson r = {corr:.2f} between S and p q^T; total mass preserved exactly.",
                 transform=axes[0].transAxes, fontsize=6.8, color=INK2, va="top")
    ax = axes[2]
    ax.axis("off")
    ax.set_title("(c) Penalty without materializing D", loc="left")
    txt = [
        (r"$D = s\,(BA - B_{ref}A_{ref})$", 10),
        (r"$R = \Vert \mathrm{diag}(\sqrt{p})\,D\,\mathrm{diag}(\sqrt{q}) \Vert_F^2$", 10),
        (r"$X = s^{1/2}[\,\mathrm{diag}(\sqrt{p})B,\;-\mathrm{diag}(\sqrt{p})B_{ref}\,]$", 9),
        (r"$Y = s^{1/2}[\,A\,\mathrm{diag}(\sqrt{q});\;A_{ref}\,\mathrm{diag}(\sqrt{q})\,]$", 9),
        (r"$R = \mathrm{tr}\left((X^{\top}X)(YY^{\top})\right)$,  cost $O(r^2(O+I))$", 9),
        ("Qwen2.5-1.5B, q/k/v/o, 28 layers:", 8),
        ("dense 154.1 M values (616.6 MB FP32)", 8),
        ("rank-1 0.27 M values (1.09 MB FP32), 566x smaller", 8),
    ]
    y = 0.95
    for t, size in txt:
        ax.text(0.0, y, t, transform=ax.transAxes, fontsize=size, color=INK if size > 8 else INK2, va="top")
        y -= 0.125
    return save(fig, "fig_4_4_factorized_penalty"), corr


# --------------------------------------------------------------------------------------
def fig_software_layers():
    fig, ax = canvas(11, 4.8, 100, 48)
    layers = [
        ("Interface", "rahc_lora.cli  (doctor, prepare-data, train)  |  scripts/ (thin entry points)", NEUTRAL),
        ("Orchestration", "training.trainer (ContinualTrainer), training.checkpoint, training.artifacts", BLUEBOX),
        ("Domain", "rl (rollout, advantages, losses)  |  consolidation.hlora  |  evaluation (matrix, metrics, capability suite, official IFEval)", PERSIST),
        ("Adapters", "models (factory, policy, tiny)  |  tasks (GSM8K, ARC, MBPP, constraints, generator, manifests)\nrewards (numeric exact match, multiple choice, code tests in sandbox, instruction constraints)", TRAIN),
        ("Infrastructure", "config (schema, validation, loader)  |  utils (atomic writes, logging, manifests, reproducibility)", FROZEN),
    ]
    for k, (name, body, role) in enumerate(layers):
        y = 38 - k * 9
        box(ax, 1, y, 16, 7.5, name, role=role, title_size=8.2)
        box(ax, 18.5, y, 80.5, 7.5, body, role=role, title_size=7.2, bold=False, align="left")
        if k < len(layers) - 1:
            arrow(ax, (9, y), (9, y - 1.5), lw=1.0)
    ax.text(1, 46.8, "Dependencies point downward only; scientific logic never lives in scripts, notebooks, or VS Code tasks.",
            fontsize=7.4, color=INK2)
    return save(fig, "fig_5_1_software_layers")


# --------------------------------------------------------------------------------------
def fig_data_isolation():
    fig, ax = canvas(11, 5.6, 100, 56)
    cols = [("Train", EVID), ("Validation", BLUEBOX), ("Anchor candidate", PERSIST), ("Held-out test", ALERT)]
    rows = [
        ("GSM8K (openai/gsm8k, main)", [5979, 747, 747, 1319], "val + anchors ranked from source train; official test"),
        ("ARC-Challenge (allenai/ai2_arc)", [1007, 299, 112, 1172], "official validation/test kept; anchors from train"),
        ("MBPP (sanitized)", [108, 43, 12, 257], "official validation/test kept; anchors from train"),
        ("Custom constraints v1.0.0", [32, 16, 16, 16], "distinct template families per role"),
    ]
    x0, cw = 29, 13.5
    for j, (name, role) in enumerate(cols):
        box(ax, x0 + j * (cw + 1), 47, cw, 5.5, name, role=role, title_size=7.6)
    for i, (name, counts, note) in enumerate(rows):
        y = 39 - i * 7.5
        box(ax, 1, y, 26.5, 6, name, role=NEUTRAL, title_size=7.2)
        for j, cnt in enumerate(counts):
            box(ax, x0 + j * (cw + 1), y, cw, 6, f"{cnt:,}", role=cols[j][1], title_size=8, bold=False)
        ax.text(x0 + 4 * (cw + 1) + 0.3, y + 3, "\n".join(__import__("textwrap").wrap(note, 18)),
                fontsize=6.2, color=INK2, va="center")
    box(ax, 1, 2, 26.5, 7, "Evaluation-only suite", role=FROZEN, title_size=7.6)
    box(ax, 29, 2, 69.5, 7, "IFEval 541 (official scorer)  |  MMLU 4-subject subset 623  |  HellaSwag 10,042"
        + chr(10) + "ARC-Easy 2,376  |  WikiText-2 held-out corpus 2,891 records",
        role=FROZEN, title_size=6.9, bold=False)
    ax.text(1, 12.6, "Allowed flows: train -> RL updates and importance; anchor candidates -> anchor memory only; validation -> tuning only;",
            fontsize=6.9, color=INK2)
    ax.text(1, 10.2, "held-out test and evaluation-only data -> task-boundary evaluator only. Loaders for evaluation-only data refuse any training role.",
            fontsize=6.9, color=INK2)
    return save(fig, "fig_5_2_data_isolation")


# --------------------------------------------------------------------------------------
def fig_sandbox():
    fig, ax = canvas(11, 3.9, 100, 38)
    box(ax, 1, 13, 15, 13, "Generated code", ["untrusted output", "never in-process"], role=ALERT)
    ax.add_patch(FancyBboxPatch((20, 3), 55, 32, boxstyle="round,pad=0,rounding_size=1.5", fc="#f7f7f5", ec=INK2, lw=1.2, ls="-"))
    ax.text(21.2, 33.6, "Ephemeral Docker container (digest-pinned python:3.11.10-slim)", fontsize=7.8, fontweight="bold", color=INK, va="top")
    controls = [
        "--network none", "--read-only root", "--cap-drop ALL", "no-new-privileges",
        "user 65534 (nobody)", "256 MB, swap = memory", "1 CPU, 32 PIDs", "ulimit cpu 5 s, nofile 64",
        "5 s timeout: kill + remove", "64 KiB output cap", "workspace bind read-only", "tmpfs /tmp noexec, 64 MB",
    ]
    for k, c in enumerate(controls):
        col, row = k % 3, k // 3
        box(ax, 22 + col * 17.5, 23.5 - row * 5.3, 16.5, 4.3, c, role=NEUTRAL, title_size=6.4, bold=False, radius=0.6)
    box(ax, 80, 20, 18.5, 11, "Test outcome only", ["passed / total tests", "timeout, exit, output", "flags"], role=EVID)
    box(ax, 80, 5, 18.5, 11, "Reward", ["fraction of tests", "passed, in [0, 1]"], role=BLUEBOX)
    arrow(ax, (16, 19.5), (20, 19.5))
    arrow(ax, (75, 25.5), (80, 25.5))
    arrow(ax, (89.25, 20), (89.25, 16))
    ax.text(1, 6, "Docker unavailable\n=> scoring fails closed\n(no host fallback)", fontsize=7, color=INK2)
    return save(fig, "fig_5_3_code_sandbox")


# --------------------------------------------------------------------------------------
def fig_hlora_implementation():
    fig, ax = canvas(11, 4.6, 100, 44)
    lane(ax, 1, 22, 60, 21, "During task t (every optimizer update)")
    lane(ax, 63, 22, 36, 21, "At the boundary of task t")
    lane(ax, 1, 1, 98, 19, "During task t+1 (penalty active)")
    box(ax, 3, 25, 17.5, 13, "Effective gradient", ["hooks on LoRA input/output", "G_l = dL_RL / dU_l"], role=BLUEBOX, body_size=6.4)
    box(ax, 22.5, 25, 17.5, 13, "Observed movement", ["dU_l = U_l(after)", "       - U_l(before)", "post-clip, post-AdamW"], role=NEUTRAL, body_size=6.4)
    box(ax, 42, 25, 17.5, 13, "Path integral", ["omega_l += -G_l * dU_l", "dense, FP32, on CPU"], role=PERSIST, body_size=6.4)
    box(ax, 65, 25, 15.5, 13, "Importance", ["relu(omega) /", "((U_end - U_start)^2", "  + eps)", "summed over tasks"], role=PERSIST, body_size=6.2)
    box(ax, 82, 25, 15.5, 13, "Layer weights", ["alpha_l = softmax over", "layers of ||Omega_l||_2", "fixed for next task"], role=ALERT, body_size=6.2)
    for a, b in ((20.5, 22.5), (40, 42), (59.5, 65), (80.5, 82)):
        arrow(ax, (a, 31.5), (b, 31.5))
    box(ax, 3, 3.5, 44, 12, "Penalty", ["lambda * sum_l alpha_l * sum_ij Omega_l[i,j] * (U_l - U_l,ref)[i,j]^2", "lambda = 1.0; reference = LoRA factors at the last boundary"], role=TRAIN, body_size=6.5)
    box(ax, 50, 3.5, 47.5, 12, "Observed in the 1.5B pilots", ["||Omega_l||_2 spans 385 to 11,729 after ARC (150-step run); the top-two", "gap of 4,010 makes the softmax exactly one-hot: alpha = 1 for layer-0", "v_proj and 0 for the other 111 modules (same in all four consolidations)"], role=ALERT, body_size=6.4)
    arrow(ax, (89.75, 25), (25, 15.5), rad=0.05)
    return save(fig, "fig_5_4_hlora_rl_implementation")


# --------------------------------------------------------------------------------------
def fig_artifact_contract():
    fig, ax = canvas(11, 4.4, 100, 42)
    tree = [
        "outputs/runs/<experiment>/<method>/<order>/<seed>/<run_id>/",
        "   resolved_config.yaml        run_manifest.json (revisions, deps, hardware)",
        "   dataset_fingerprints.json   metrics.jsonl   system_metrics.jsonl",
        "   evaluation_records.jsonl    performance_matrix.csv",
        "   general_capabilities.csv    layer_importance / layer_conflicts .parquet",
        "   memory_manifest.json        checkpoint_manifest.json   summary.json",
        "   checkpoints/task-XX-step-XXXXXXXX/   latest_checkpoint.json",
    ]
    ax.add_patch(FancyBboxPatch((1, 3), 56, 36, boxstyle="round,pad=0,rounding_size=1.2", fc="#f7f7f5", ec=AXIS, lw=0.9))
    ax.text(2.5, 37.3, "Run directory (authoritative local artifacts)", fontsize=8, fontweight="bold", color=INK, va="top")
    for k, line in enumerate(tree):
        ax.text(2.5, 32.5 - k * 4.1, line, fontsize=6.7, family="DejaVu Sans Mono", color=INK2, va="top")
    box(ax, 60, 21, 38.5, 18, "Checkpoint contents", ["LoRA adapter; optimizer, scheduler, scaler (explicit absent)",
        "task index, global step, rollout counter, policy version", "general + rollout RNG states", "performance matrix, capability history",
        "retention state: reference factors, importance, layer weights (HLoRA-RL)"], role=PERSIST, wrap=62, body_size=6.3, align="left")
    box(ax, 60, 3, 38.5, 15, "Integrity and resume rules", ["new directory per boundary; SHA-256 + size per file",
        "latest pointer updated only after validation", "resume rejects changed config, model, data, deps",
        "resume within 1e-7 of uninterrupted run (tested)"], role=EVID, wrap=62, body_size=6.3, align="left")
    arrow(ax, (57, 12), (60, 12))
    arrow(ax, (57, 30), (60, 30))
    return save(fig, "fig_5_5_artifact_contract")


# --------------------------------------------------------------------------------------
def fig_work_plan():
    import datetime as dt
    import matplotlib.dates as mdates
    d = dt.date
    done = [
        ("P1 accepted (literature review)", d(2026, 6, 10), None, "Complete"),
        ("Specification: project.md, Goal.md, architecture.md", d(2026, 9, 22), d(2026, 9, 23), "Complete"),
        ("Phases 0-1: scaffold, data, rewards, evaluation", d(2026, 9, 22), d(2026, 9, 23), "Complete"),
        ("Phase 2: RL-LoRA trainer and four GPU pilots", d(2026, 9, 22), d(2026, 9, 27), "Complete"),
        ("Phase 3 (part): HLoRA-RL and ARC->GSM8K pilots", d(2026, 9, 27), d(2026, 9, 29), "Partial"),
        ("P2 report and presentation", d(2026, 9, 28), d(2026, 10, 15), "Partial"),
    ]
    plan = [
        ("Phase 3 completion: supervised HLoRA reproduction", d(2026, 10, 5), d(2026, 10, 25), "Planned"),
        ("Phases 4-7: RAHC-LoRA components and tests", d(2026, 10, 15), d(2026, 11, 25), "Planned"),
        ("Phase 8: pilot, >= 2 orders x 3 seeds", d(2026, 11, 20), d(2026, 12, 15), "Planned"),
        ("Phase 9: main study, ablations, statistics", d(2026, 12, 10), d(2027, 1, 20), "Planned"),
        ("Final thesis writing and defence", d(2027, 1, 5), d(2027, 1, 31), "Planned"),
    ]
    rows = done + plan
    colors = {"Complete": BLUE, "Partial": ORANGE, "Planned": "#b8b6ae"}
    fig, ax = plt.subplots(figsize=(10.5, 4.3))
    for k, (name, start, end, status) in enumerate(rows):
        if end is None:
            ax.plot([mdates.date2num(start)], [k], marker="D", ms=7, color=colors[status], mec=SURFACE, mew=1)
            ax.text(mdates.date2num(start) + 3, k, start.strftime("%d %b %Y"), va="center", fontsize=6.8, color=INK2)
            continue
        left = mdates.date2num(start)
        width = max(mdates.date2num(end) - left, 1.0)
        ax.barh(k, width, left=left, height=0.52, color=colors[status], edgecolor=SURFACE, linewidth=1.2)
        if status != "Planned" and width < 8:
            ax.text(left + width + 2, k, f"{start.strftime('%d %b')}-{end.strftime('%d %b')}", va="center", fontsize=6.6, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7.4)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(mdates.date2num(d(2026, 6, 1)), mdates.date2num(d(2027, 2, 5)))
    report = mdates.date2num(d(2026, 9, 29))
    ax.axvline(report, color=INK2, lw=0.9)
    ax.text(report - 2, len(rows) - 0.35, "29 Sep 2026", fontsize=6.6, color=INK2, ha="right")
    ax.axhline(len(done) - 0.5, color=GRID, lw=0.8)
    clean_axes(ax, "x")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colors["Complete"], label="Complete (repository evidence)"),
                       Patch(color=colors["Partial"], label="In progress"),
                       Patch(color=colors["Planned"], label="Proposed schedule (to confirm with supervisor)")],
              loc="lower left", fontsize=7.0, ncol=1, bbox_to_anchor=(0.0, 0.02))
    return save(fig, "fig_3_1_work_plan")


def main():
    out = {}
    fig_research_roadmap()
    fig_cl_taxonomy()
    fig_evaluation_protocol()
    fig_system_architecture()
    _, out["gauge_relative_spread"] = fig_gauge_invariance()
    fig_grpo_pipeline()
    fig_update_sequence()
    fig_task_lifecycle()
    _, out["toy_rank1_corr"] = fig_factorized_penalty()
    fig_software_layers()
    fig_data_isolation()
    fig_sandbox()
    fig_hlora_implementation()
    fig_artifact_contract()
    fig_work_plan()
    print(out)


if __name__ == "__main__":
    main()

"""Thesis tables: one source of truth for the Word documents (tables.json) and the PNG renders."""

from __future__ import annotations

import json
import textwrap

from figstyle import AXIS, DATA, GRID, INK, INK2, SURFACE, TAB, plt

R = json.loads((DATA / "results_summary.json").read_text(encoding="utf-8"))
RUNS = R["runs"]
C = R["comparisons"]
ST = R["storage"]


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def pp(x, d=2):
    return f"{100 * x:+.{d}f}"


TABLES: dict[str, dict] = {}


def table(tid, caption, columns, rows, widths, note=None):
    if abs(sum(widths) - 1.0) > 1e-6:
        raise ValueError(f"{tid}: widths must sum to 1")
    for r in rows:
        if len(r) != len(columns):
            raise ValueError(f"{tid}: row has {len(r)} cells, expected {len(columns)}: {r}")
    TABLES[tid] = {"caption": caption, "columns": columns, "rows": rows, "widths": widths, "note": note}


# ---------------------------------------------------------------- Chapter 2
table(
    "tab_2_1_literature_comparison",
    "Comparison of representative continual-learning and forgetting studies relevant to sequential RL fine-tuning with one shared LoRA adapter.",
    ["Ref.", "Method / study", "Category", "Setting", "Key mechanism", "Gap for our problem"],
    [
        ["@kirkpatrick2017", "EWC", "Regularization", "Supervised; Atari RL", "Fisher-weighted quadratic penalty to old weights", "Needs Fisher estimate; applied to raw parameters, not LoRA effective update"],
        ["@zenke2017", "Synaptic Intelligence", "Regularization", "Supervised", "Online path integral of gradient x movement", "Assumes low-noise gradients; untested with sparse policy gradients"],
        ["@aljundi2018", "MAS", "Regularization", "Supervised", "Output-sensitivity importance", "Importance ignores reward and advantage"],
        ["@chen2020recadam", "RecAdam", "Regularization", "NLP fine-tuning", "Annealed pull towards pretrained weights", "Single-step transfer, not a task sequence"],
        ["@song2025hlora", "HLoRA / HLER", "Regularization", "Supervised LLM domain tuning", "Element-wise importance + softmax layer coefficients", "Supervised loss; SGD closed form; dense importance"],
        ["@zheng2026ewclora", "EWC-LoRA", "Regularization", "Low-rank continual learning", "EWC on a shared low-rank update", "Supervised Fisher; no behavioural anchor"],
        ["@sun2020lamol", "LAMOL", "Replay", "Lifelong language learning", "Model generates pseudo-samples of old tasks", "Pseudo-data quality; supervised"],
        ["@dautume2019", "MbPA++", "Replay / memory", "Lifelong language learning", "Episodic memory with local adaptation", "Memory and inference cost grow"],
        ["@wang2023olora", "O-LoRA", "PEFT subspace", "LLM continual instruction tuning", "Orthogonal LoRA subspace per task", "Adapter count grows with tasks"],
        ["@ren2024ilora", "I-LoRA", "PEFT + replay", "LLM continual fine-tuning", "Interpolated dual-memory LoRA", "Replay buffer; supervised"],
        ["@lopezpaz2017; @chaudhry2019", "GEM / A-GEM", "Gradient projection", "Supervised", "Project updates to avoid old-loss increase", "Hard constraint; needs memory gradients each step"],
        ["@yu2020pcgrad", "PCGrad", "Gradient surgery", "Multi-task", "Remove conflicting gradient components", "Simultaneous tasks, not sequential"],
        ["@kaplanis2019", "Policy consolidation", "Behavioural", "Continual RL (control)", "Cascade of policies coupled by KL", "Extra policy copies; not LLM"],
        ["@shenfeld2025razor", "RL's Razor", "Analysis", "LLM RL vs SFT", "Forgetting tracks KL on new task; RL is KL-minimal", "Explains, does not prevent, multi-task RL forgetting"],
        ["@lai2025rft", "RFT in continual post-training", "Analysis", "Multimodal LLM task stream", "RFT forgets less; reward variance as implicit regularizer", "No explicit retention mechanism"],
        ["@luo2026cpo", "CPO", "Behavioural / KL", "Continual RL post-training of VLMs", "Replay-free KL on the prior-task distribution", "Concurrent; multimodal; no parameter consolidation"],
        ["@wang2026geometry", "Geometry conflict", "Gradient geometry", "LLM continual post-training", "Forgetting as geometric incompatibility of updates", "Concurrent; not LoRA-RL specific"],
        ["This work", "RAHC-LoRA", "Hybrid", "Sequential verifiable-reward RL, one LoRA", "Reward-aware effective-weight importance, rank-1 factors, anchor KL, conflict gate, dual controller", "Full method implementation and multi-seed evaluation pending (P3)"],
    ],
    [0.07, 0.14, 0.12, 0.17, 0.26, 0.24],
)

# ---------------------------------------------------------------- Chapter 3
table(
    "tab_3_1_requirements",
    "Final specifications and requirements of the RAHC-LoRA research platform and their verification status.",
    ["ID", "Requirement", "Type", "Verification in repository", "Status"],
    [
        ["R1", "One frozen backbone and exactly one shared LoRA adapter; no per-task adapters", "Scientific", "Factory and trainer tests: frozen weights bit-identical, only LoRA in optimizer", "Met"],
        ["R2", "Disjoint train / validation / anchor-candidate / test roles; IFEval evaluation-only", "Scientific", "Hash-ranked splits, manifest duplicate checks, access-denying loaders, overlap audit (0 exact, 0 normalized)", "Met"],
        ["R3", "Deterministic rewards in [0, 1]; documented zero for parse failures", "Scientific", "Hand-checked accepted, rejected, malformed examples in unit tests", "Met"],
        ["R4", "Generated code executed only in an isolated, network-disabled, resource-limited sandbox", "Safety", "Docker command test; fail-closed test when Docker unavailable", "Met"],
        ["R5", "Full performance matrix after every task plus capability suite vs frozen model", "Evaluation", "Mock two-task gate; every GPU run records A[i, j] and 6 capability metrics", "Met"],
        ["R6", "All method components switchable by configuration only", "Reproducibility", "Hydra method groups; validation forbids hidden retention in rl_lora", "Met for implemented components"],
        ["R7", "Atomic, hashed, resumable checkpoints including retention state", "Reproducibility", "Resume within 1e-7; SHA-256 validation of all pilot checkpoints", "Met"],
        ["R8", "Fail loudly on non-finite values; no silent fallbacks", "Engineering", "Finite checks on rewards, advantages, losses, gradients, importance", "Met"],
        ["R9", "Bounded persistent retention state (factorized importance, fixed anchors)", "Scientific", "Dense importance only in the HLoRA-RL reference baseline", "Pending (Phases 5-6)"],
        ["R10", "Three task orders x three seeds, paired statistics, ablations", "Evaluation", "Protocol and aggregation specified", "Pending (Phases 8-9)"],
        ["R11", "Consumer single-GPU feasibility (16 GB VRAM)", "Constraint", "Peak allocation 12.2 GB (RL-LoRA) and 14.1 GB (HLoRA-RL) on Qwen2.5-1.5B", "Met"],
    ],
    [0.05, 0.33, 0.12, 0.36, 0.14],
)

table(
    "tab_3_3_risk_register",
    "Risk register: risks anticipated in the specification, controls, and what was observed during P2.",
    ["Risk", "Consequence", "Control", "Observed in P2"],
    [
        ["Weak base performance / no forgetting", "Retention cannot be measured", "Pilot continuation rule (>= 10 pp)", "Occurred: all six RL-LoRA streams < 10 pp; stream redesign required"],
        ["Reward sparsity", "Zero advantages, no gradient", "Grouped rewards, longer budgets, logging of zero-variance groups", "Occurred: 55-87% zero-variance groups on GSM8K and MBPP"],
        ["Over-consolidation", "Old task kept, new task not learned", "Co-primary plasticity metric", "Not observed; penalty was very small"],
        ["Baseline mismatch", "Unfair comparison", "Same trainer, seed, data, budgets, generation", "Controlled: task-1 trajectories bit-identical"],
        ["Dense transient memory", "Out-of-memory", "CPU-resident dense importance; factorization planned", "Peak GPU +1.9 GB; 616.6 MB dense state on CPU"],
        ["Unstable layer coefficients", "Protection concentrated or noisy", "Normalization, clipping, static ablation", "Occurred: softmax of raw norms became one-hot"],
        ["Code reward escape", "Host compromise", "Network-disabled Docker sandbox, fail closed", "No incident; sandbox check passed"],
        ["Long-job interruption", "Lost progress", "Atomic boundary checkpoints, verified resume", "Occurred (order_2, step 918); recovered from step-500 checkpoint"],
        ["Host memory pressure", "Slow or failed steps", "System metrics logging", "Occurred: orphaned monitor used 22.5 GiB; removed"],
    ],
    [0.22, 0.2, 0.28, 0.3],
)

table(
    "tab_3_2_platform",
    "Hardware and software platform recorded in the run manifests.",
    ["Component", "Specification"],
    [
        ["GPU", "NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB VRAM, BF16 supported"],
        ["CPU / OS", "Intel64 Family 6 Model 183 (x86-64); Windows 11 Pro (build 26200)"],
        ["Python / CUDA", "Python 3.11.15 (Conda gpu-env), CUDA runtime 12.6"],
        ["Core libraries", "torch 2.13.0+cu126, transformers 4.57.6, peft 0.20.0, trl 0.29.1 (not used for GRPO semantics), accelerate 1.14.0, datasets 4.8.5, hydra-core 1.3.7"],
        ["Analysis libraries", "numpy 2.4.6, pandas 3.0.5, scipy 1.17.1, scikit-learn 1.9.0, pyarrow 25.0.1"],
        ["Code sandbox", "Docker Desktop Linux engine 29.2.1; python:3.11.10-slim-bookworm pinned by digest"],
        ["Models", "Qwen2.5-0.5B-Instruct @ 7ae5576; Qwen2.5-1.5B-Instruct @ 989aa79 (pinned 40-character revisions)"],
        ["Quality gates", "Ruff format and lint, mypy (51 source files), pytest (59 tests, 12.5 s on CPU)"],
    ],
    [0.2, 0.8],
)

# ---------------------------------------------------------------- Chapter 4
table(
    "tab_4_1_notation",
    "Notation used in the RAHC-LoRA formulation.",
    ["Symbol", "Meaning"],
    [
        ["T, t", "number of tasks in the stream; index of the current task"],
        ["W0, l", "frozen backbone weight of a target linear layer; index of a protected LoRA module"],
        ["A_l, B_l, r, alpha", "LoRA down-projection (r x I), up-projection (O x r), rank, scaling numerator"],
        ["s_l = alpha / r", "LoRA scale (2.0 in all pilots: r = 8, alpha = 16)"],
        ["DeltaW_l = s_l B_l A_l", "effective LoRA update, the protected object"],
        ["G", "responses sampled per prompt (group size, 4 in the pilots)"],
        ["r_i, A_hat_i", "deterministic reward of response i and its group-normalized advantage"],
        ["L_RL", "clipped GRPO objective of the current task"],
        ["g_l^RL, g_l^anc", "effective-weight gradients of the RL loss and of the anchor KL"],
        ["delta_l", "observed post-step change of DeltaW_l"],
        ["S_l, p_l, q_l", "importance matrix and its rank-1 nonnegative row / column factors"],
        ["R_l", "effective-weight consolidation penalty of module l"],
        ["c_l", "detached conflict-aware layer coefficient"],
        ["L_KL", "compact forward KL between stored anchor references and the current policy"],
        ["lambda_f, lambda_p", "adaptive functional (KL) multiplier and fixed parameter-penalty weight"],
        ["A[i, j]", "held-out score on task j after finishing task i"],
    ],
    [0.28, 0.72],
)

table(
    "tab_4_2_method_matrix",
    "Methods compared in the study and the retention components each one enables (all run through the same trainer).",
    ["Method", "Param. consolidation", "Reward-aware importance", "Factorized", "Anchor KL", "Conflict gate", "Dual controller", "Status"],
    [
        ["Standard RL-LoRA", "-", "-", "-", "-", "-", "-", "Implemented, piloted"],
        ["RL-LoRA + KL to initial policy", "-", "-", "-", "current prompts", "-", "-", "Planned baseline"],
        ["Balanced replay (same budget)", "-", "-", "-", "replay", "-", "-", "Config stub"],
        ["EWC-LoRA / SI-LoRA", "yes (Fisher / SI)", "-", "diagnostic", "-", "-", "-", "Config stub"],
        ["HLoRA-RL (direct transfer)", "yes (dense SI path)", "-", "- (dense)", "-", "- (static softmax)", "-", "Implemented, piloted"],
        ["O-LoRA (subspace)", "orthogonality", "-", "-", "-", "-", "-", "Planned baseline"],
        ["RAHC-LoRA (proposed)", "yes", "yes", "rank-1", "yes", "yes", "yes", "Specified; Phases 4-7"],
        ["Joint multi-task RL", "-", "-", "-", "-", "-", "-", "Upper reference (violates sequential data)"],
    ],
    [0.22, 0.12, 0.11, 0.09, 0.1, 0.11, 0.09, 0.16],
)

table(
    "tab_4_3_ablations",
    "Planned ablations; each is a configuration switch, never a source edit.",
    ["Ablation", "Question answered", "Configuration switch"],
    [
        ["Remove anchor KL", "Is parameter consolidation sufficient under policy shift?", "method.use_anchor_kl=false"],
        ["Remove parameter consolidation", "Does behavioural retention explain the result?", "method.use_parameter_consolidation=false"],
        ["Dense vs factorized importance", "What is lost through rank-1 compression?", "method.use_factorized_importance"],
        ["Reward-aware vs ordinary importance", "Does advantage information improve importance?", "method.use_reward_aware_importance"],
        ["Remove conflict gate", "Does dynamic conflict control improve plasticity?", "method.use_conflict_gate=false"],
        ["Static vs adaptive KL coefficient", "Is the dual controller necessary?", "method.use_dual_controller=false"],
        ["Memory 0 / 128 / 512 / 2048", "How much historical behaviour is needed?", "method.anchor_memory.capacity"],
        ["LoRA rank 4 / 8 / 16", "Does adapter capacity change forgetting?", "method.lora.rank (alpha scaled)"],
        ["Effective-weight vs factor-level penalty", "Does rescaling invariance matter in practice?", "penalty parameterization flag"],
    ],
    [0.3, 0.42, 0.28],
)

table(
    "tab_4_4_rahc_hyperparameters",
    "Default RAHC-LoRA hyperparameters fixed in the specification before any method pilot.",
    ["Component", "Hyperparameter", "Default"],
    [
        ["LoRA", "rank / alpha / dropout / targets", "8 / 16 / 0.0 / q, k, v, o projections (pilots); 8-16 in main study"],
        ["Importance", "EMA beta / epsilon / clip quantile / accumulator", "0.99 / 1e-8 / 0.995 / FP32"],
        ["Anchor memory", "capacity / batch / top-k logits / selection", "512 / 16 / 32 / balanced, diverse reservoir"],
        ["Conflict gate", "zero-norm threshold / coefficient clip", "1e-12 / 5.0"],
        ["Dual controller", "initial lambda / dual lr / target KL / EMA / max lambda / patience", "0.01 (specification; rahc_lora.yaml currently inherits 0.0 and must be updated) / 0.001 / 0.02 / 0.95 / 10.0 / 100 steps"],
        ["Parameter penalty", "lambda_parameter", "1.0"],
        ["GRPO", "group size / prompts per update / clip eps / advantage clip", "4 / 2 / 0.2 / 5.0"],
        ["Optimizer", "AdamW lr / betas / weight decay / grad-norm clip", "1e-4 / (0.9, 0.999) / 0.0 / 1.0"],
    ],
    [0.2, 0.42, 0.38],
)

# ---------------------------------------------------------------- Chapter 5
table(
    "tab_5_1_datasets",
    "Pinned datasets, project roles, split sizes, and training rewards.",
    ["Dataset (revision)", "Role", "Train", "Val.", "Anchor cand.", "Test / eval", "Reward or metric"],
    [
        ["GSM8K main (740312a)", "RL task", "5,979", "747", "747", "1,319", "normalized exact final number"],
        ["ARC-Challenge (210d026)", "RL task", "1,007", "299", "112", "1,172", "exact option label"],
        ["MBPP sanitized (4bb6404)", "RL task", "108", "43", "12", "257", "fraction of hidden tests passed (sandbox)"],
        ["Custom constraints v1.0.0", "RL task", "32", "16", "16", "16", "fraction of constraints satisfied"],
        ["IFEval (966cd89)", "Evaluation only", "-", "-", "-", "541", "official strict / loose scorer"],
        ["MMLU, 4 subjects (c30699e)", "Evaluation only", "-", "-", "-", "623", "exact option label"],
        ["HellaSwag (218ec52)", "Evaluation only", "-", "-", "-", "10,042", "exact option label"],
        ["ARC-Easy (210d026)", "Evaluation only", "-", "-", "-", "2,376", "exact option label"],
        ["WikiText-2 raw (b08601e)", "Evaluation only", "-", "-", "-", "2,891", "token perplexity (<= 256 tokens)"],
    ],
    [0.23, 0.13, 0.08, 0.07, 0.1, 0.1, 0.29],
    note="Custom-constraint prompts: 0 exact and 0 normalized overlaps with the 541 official IFEval prompts. The pilots evaluate the first 64 or 128 examples of each held-out split.",
)

table(
    "tab_5_2_experimental_configuration",
    "Experimental configuration shared by all GPU pilots (values from the resolved run configurations).",
    ["Setting", "Value"],
    [
        ["Backbone", "Qwen2.5-1.5B-Instruct (one 0.5B pilot), BF16, frozen"],
        ["Adapter", "one LoRA, rank 8, alpha 16 (scale 2.0), dropout 0, q/k/v/o in all layers; 2,179,072 trainable parameters (1.5B)"],
        ["Rollouts", "2 prompts x G = 4 responses per update (8 rollouts); temperature 1.0, top-p 1.0; <= 128 prompt and <= 128 response tokens"],
        ["GRPO", "population-std group normalization, zero advantage for zero-variance groups, advantage clip 5.0, ratio clip 0.2, one update per rollout"],
        ["Optimizer", "AdamW, lr 1e-4, betas (0.9, 0.999), no weight decay, grad-norm clip 1.0, constant schedule"],
        ["Budgets", "100 or 500 updates per task (Phase 2); 100 or 150 updates per task (ARC -> GSM8K)"],
        ["HLoRA-RL", "lambda = 1.0, epsilon = 1e-8, dense importance on CPU, static softmax layer weights"],
        ["Evaluation", "greedy decoding, <= 64 new tokens; 128 (Phase 2) or 64 (ARC -> GSM8K) held-out examples per task and capability"],
        ["Seed / orders", "seed 1; order_1 (GSM8K -> ARC), order_2 (MBPP -> GSM8K), order_arc_first (ARC -> GSM8K)"],
    ],
    [0.18, 0.82],
)

table(
    "tab_5_5_phase_status",
    "Phase-by-phase implementation status established from repository evidence (29 September 2026).",
    ["Phase", "Deliverable", "Status", "Evidence"],
    [
        ["0", "Scaffold, typed Hydra config, CLI, logging, manifests, tooling", "Complete", "doctor, ruff, mypy, pytest pass"],
        ["1", "Four task adapters, generator, rewards, sandbox, matrix and metrics", "Complete", "hand-checked reward tests; mock two-task gate"],
        ["2", "Standard RL-LoRA GRPO trainer, checkpoint/resume, capability suite", "Complete", "4 GPU pilots; resume within 1e-7"],
        ["3", "Direct HLoRA-to-RL baseline", "Partial", "dense importance + penalty implemented and piloted; supervised reproduction not yet run"],
        ["4", "Reward-aware effective-weight importance", "Not started", "specified (Section 4.3.3)"],
        ["5", "Rank-1 factorization and Gram-form penalty", "Not started", "specified (Section 4.3.4)"],
        ["6", "Fixed anchor memory and compact KL", "Not started", "specified (Section 4.3.5)"],
        ["7", "Conflict gate and dual controller", "Not started", "specified (Section 4.3.6)"],
        ["8-9", "Method pilot, main study, ablations, statistics", "Not started", "gated by continuation rule"],
    ],
    [0.07, 0.43, 0.13, 0.37],
)

table(
    "tab_5_4_validation",
    "Engineering validation evidence (cheap checks before GPU runs).",
    ["Check", "Result"],
    [
        ["Unit and integration tests", "59 passed in 12.5 s (CPU, Python 3.12 venv); also 56/56 in the CUDA gpu-env at Phase 2"],
        ["Frozen backbone", "synthetic-reward update changes LoRA tensors; every frozen parameter bit-identical"],
        ["Interrupted resume", "adapter and matrix match the uninterrupted run within 1e-7 absolute tolerance"],
        ["Repeat determinism", "two CLI smoke runs match exactly in adapter tensors, events, summary, matrix"],
        ["First-task equivalence", "HLoRA-RL and RL-LoRA losses and gradient norms bit-identical for all task-1 updates"],
        ["Checkpoint integrity", "all files in every pilot checkpoint pass size and SHA-256 validation"],
        ["Data isolation", "custom constraints vs IFEval: 0 exact, 0 normalized overlaps; duplicate IDs and hashes rejected"],
        ["Sandbox", "Docker backend passed a generated-code check; unavailable Docker fails closed"],
        ["IFEval scorer", "pinned google-research checkout and NLTK punkt_tab digest verified; no fallback scorer"],
    ],
    [0.26, 0.74],
)


def run_row(key, label):
    r = RUNS[key]
    c = r["continual"]
    return [label, r["model"].replace("-Instruct", ""), " -> ".join(t.replace("arc_challenge", "ARC").replace("gsm8k", "GSM8K").replace("mbpp", "MBPP") for t in r["task_order"]),
            str(r["steps_per_task"]), str(r["evaluation_examples_per_task"]), f"{r['system']['total_optimizer_hours']:.1f}",
            r["run_dir"].split("/")[-1]]


table(
    "tab_5_3_run_inventory",
    "Inventory of the GPU runs analysed in this report (seed 1 in every run).",
    ["Label", "Model", "Order", "Updates / task", "Eval n", "Optimizer hours", "Run ID"],
    [
        run_row("p05_order1_100", "P2-A RL-LoRA"),
        run_row("p15_order1_100", "P2-B RL-LoRA"),
        run_row("p15_order1_500", "P2-C RL-LoRA"),
        run_row("p15_order2_500", "P2-D RL-LoRA"),
        run_row("arc_base_100", "P3-A RL-LoRA"),
        run_row("arc_hlora_100", "P3-A HLoRA-RL"),
        run_row("arc_base_150", "P3-B RL-LoRA"),
        run_row("arc_hlora_150", "P3-B HLoRA-RL"),
    ],
    [0.13, 0.15, 0.13, 0.09, 0.07, 0.09, 0.34],
    note="P2-D optimizer hours cover the 500 recovered GSM8K updates; the preserved source run supplied the MBPP task and its verified step-500 checkpoint.",
)

# ---------------------------------------------------------------- Chapter 6
def phase2_row(key, label):
    r = RUNS[key]
    c = r["continual"]
    return [label, pct(c["task1_after_task1"]), pct(c["task1_after_task2"]), pct(c["task2_after_task2"]),
            pp(c["average_forgetting"]), pp(c["backward_transfer"]), pct(c["final_average_performance"]),
            "no" if c["average_forgetting"] < 0.10 else "yes"]


table(
    "tab_6_1_phase2_results",
    "Standard RL-LoRA pilots: performance matrices and continual metrics (percent; seed 1).",
    ["Run", "T1 after T1", "T1 after T2", "T2 after T2", "Forgetting (pp)", "BWT (pp)", "Final avg.", "Meets 10 pp rule"],
    [
        phase2_row("p05_order1_100", "0.5B GSM8K -> ARC, 100/task"),
        phase2_row("p15_order1_100", "1.5B GSM8K -> ARC, 100/task"),
        phase2_row("p15_order1_500", "1.5B GSM8K -> ARC, 500/task"),
        phase2_row("p15_order2_500", "1.5B MBPP -> GSM8K, 500/task"),
        phase2_row("arc_base_100", "1.5B ARC -> GSM8K, 100/task"),
        phase2_row("arc_base_150", "1.5B ARC -> GSM8K, 150/task"),
    ],
    [0.26, 0.1, 0.1, 0.1, 0.11, 0.1, 0.1, 0.13],
)


def arc_rows(budget):
    rows = []
    for method, key in (("RL-LoRA", f"arc_base_{budget}"), ("HLoRA-RL", f"arc_hlora_{budget}")):
        r = RUNS[key]
        c = r["continual"]
        sc = r["scores"]

        def cell(k):
            s = sc[k]
            return f"{pct(s['score'], 1)} [{pct(s['wilson95'][0], 1)}, {pct(s['wilson95'][1], 1)}]"

        rows.append([f"{method}, {budget}/task", cell("arc_challenge|after_arc_challenge"), cell("arc_challenge|after_gsm8k"),
                     cell("gsm8k|after_gsm8k"), pp(c["average_forgetting"]), pct(c["final_average_performance"])])
    return rows


table(
    "tab_6_2_arc_gsm_results",
    "ARC-Challenge -> GSM8K pilots: held-out accuracy with Wilson 95% intervals (n = 64 per cell) and continual metrics.",
    ["Method, budget", "ARC after T1", "ARC after T2", "GSM8K after T2", "Forgetting (pp)", "Final avg. (%)"],
    arc_rows("100") + arc_rows("150"),
    [0.2, 0.18, 0.18, 0.18, 0.13, 0.13],
)


def test_rows():
    rows = []
    for budget in ("100", "150"):
        c = C[budget]
        for cell, label in (("arc_challenge|after_gsm8k", "ARC after T2"), ("gsm8k|after_gsm8k", "GSM8K after T2")):
            x = c[cell]
            rows.append([f"{budget}/task", label, pct(x["baseline"], 1), pct(x["hlora"], 1),
                         pp(x["difference_hlora_minus_baseline"], 1),
                         f"[{pp(x['bootstrap95_difference'][0], 1)}, {pp(x['bootstrap95_difference'][1], 1)}]",
                         f"{x['only_baseline_correct']} / {x['only_hlora_correct']}",
                         f"{x['mcnemar_exact_p']:.2g}"])
        for name, label in (("baseline", "RL-LoRA ARC forgetting"), ("hlora", "HLoRA-RL ARC forgetting")):
            x = c[f"arc_forgetting|{name}"]
            rows.append([f"{budget}/task", label, "-", "-", pp(x["forgetting"], 1),
                         f"[{pp(x['bootstrap95_forgetting'][0], 1)}, {pp(x['bootstrap95_forgetting'][1], 1)}]",
                         f"{x['examples_lost']} lost / {x['examples_gained']} gained", f"{x['mcnemar_exact_p']:.2g}"])
    return rows


table(
    "tab_6_3_paired_tests",
    "Example-level paired analysis on identical held-out examples: paired bootstrap 95% intervals (10,000 resamples) and exact McNemar tests.",
    ["Budget", "Comparison", "RL-LoRA (%)", "HLoRA-RL (%)", "Difference (pp)", "Bootstrap 95% CI", "Discordant (RL only / HLoRA only)", "McNemar p"],
    test_rows(),
    [0.08, 0.2, 0.1, 0.1, 0.1, 0.14, 0.18, 0.1],
    note="These tests quantify evaluation-sample uncertainty for a single training seed. They do not measure seed-to-seed training variance, which the protocol requires (three seeds) before any method claim.",
)


def cap_rows(budget):
    caps = [("ifeval", "strict", "IFEval strict"), ("ifeval", "loose", "IFEval loose"), ("mmlu", "accuracy", "MMLU subset"),
            ("hellaswag", "accuracy", "HellaSwag"), ("arc_easy", "accuracy", "ARC-Easy"), ("perplexity", "perplexity", "WikiText-2 PPL")]
    rows = []
    for cap, metric, label in caps:
        vals = []
        for key in (f"arc_base_{budget}", f"arc_hlora_{budget}"):
            rr = {r["after_task"]: r for r in RUNS[key]["capabilities"] if r["capability"] == cap and r["metric"] == metric}
            vals.append(rr)
        fmt = (lambda v: f"{v:.2f}") if cap == "perplexity" else (lambda v: pct(v, 1))
        rows.append([label, fmt(vals[0]["frozen_start"]["score"]), fmt(vals[0]["arc_challenge"]["score"]),
                     fmt(vals[0]["gsm8k"]["score"]), fmt(vals[1]["gsm8k"]["score"])])
    return rows


table(
    "tab_6_5_capabilities",
    "General-capability suite for the 150-update ARC -> GSM8K pilots (n = 64 per metric; percent except perplexity).",
    ["Capability", "Frozen model", "After ARC (both methods)", "After GSM8K: RL-LoRA", "After GSM8K: HLoRA-RL"],
    cap_rows("150"),
    [0.24, 0.17, 0.21, 0.19, 0.19],
    note="Scores come from generative evaluation with at most 64 greedy tokens and strict option-label parsing. The low frozen-model scores on MMLU, HellaSwag, and ARC-Easy mainly reflect answer-format failures, so the gains after ARC training measure format transfer as much as knowledge.",
)


def diag_rows():
    rows = []
    for budget in ("100", "150"):
        for method, key in (("RL-LoRA", f"arc_base_{budget}"), ("HLoRA-RL", f"arc_hlora_{budget}")):
            for task, tl in (("arc_challenge", "ARC"), ("gsm8k", "GSM8K")):
                t = RUNS[key]["training"][task]
                rows.append([f"{method}, {budget}/task", tl, f"{t['mean_raw_reward']:.3f}", f"{t['positive_reward_batches']}/{t['updates']}",
                             pct(t["zero_variance_group_fraction"], 0), f"{t['zero_gradient_updates']}/{t['updates']}",
                             pct(t["parse_failure_rate"], 1), f"{t['mean_sampled_policy_kl']:.4f}"])
    return rows


table(
    "tab_6_4_training_diagnostics",
    "Training-rollout diagnostics (training data only, never used as held-out evidence).",
    ["Run", "Task", "Mean reward", "Positive batches", "Zero-var. groups (%)", "Zero-grad updates", "Parse failures (%)", "Sampled KL"],
    diag_rows(),
    [0.2, 0.08, 0.11, 0.12, 0.13, 0.13, 0.12, 0.11],
)

st15 = ST["Qwen2.5-1.5B-Instruct"]
table(
    "tab_6_6_efficiency",
    "Compute and persistent-state cost on the RTX 4070 Ti SUPER (Qwen2.5-1.5B unless noted).",
    ["Quantity", "Standard RL-LoRA", "HLoRA-RL (dense)", "RAHC-LoRA target (rank-1)"],
    [
        ["Mean seconds per update (150/task)", f"{RUNS['arc_base_150']['system']['mean_step_seconds']:.1f}", f"{RUNS['arc_hlora_150']['system']['mean_step_seconds']:.1f}", "to be measured"],
        ["Peak GPU memory allocated (GB)", f"{RUNS['arc_base_150']['system']['peak_gpu_allocated_gb']:.2f}", f"{RUNS['arc_hlora_150']['system']['peak_gpu_allocated_gb']:.2f}", "to be measured"],
        ["Peak process RSS (GB)", f"{RUNS['arc_base_150']['system']['max_cpu_rss_gb']:.2f}", f"{RUNS['arc_hlora_150']['system']['max_cpu_rss_gb']:.2f}", "to be measured"],
        ["Trainable LoRA parameters", f"{st15['trainable_lora_parameters']:,}", f"{st15['trainable_lora_parameters']:,}", f"{st15['trainable_lora_parameters']:,}"],
        ["Importance values stored", "0", f"{st15['dense_importance_elements']:,}", f"{st15['rank1_factorized_elements']:,}"],
        ["Importance bytes (FP32)", "0", f"{st15['dense_importance_fp32_bytes'] / 1e6:.1f} MB", f"{st15['rank1_factorized_fp32_bytes'] / 1e6:.2f} MB"],
        ["Reference LoRA factors (FP32)", "0", f"{st15['lora_fp32_bytes'] / 1e6:.2f} MB", f"{st15['lora_fp32_bytes'] / 1e6:.2f} MB"],
    ],
    [0.34, 0.2, 0.22, 0.24],
)

table(
    "tab_6_7_findings_summary",
    "Summary of findings against the research questions and hypotheses.",
    ["Question / hypothesis", "Evidence in P2", "Status"],
    [
        ["Does standard RL-LoRA forget measurably on the chosen streams?", "Forgetting -1.30 to +3.12 pp in six streams; max possible loss < 10 pp in three", "Not yet: stream must be redesigned (harder conflict, longer budgets, 4-task streams)"],
        ["H1: consolidation reduces forgetting vs RL-LoRA", "ARC forgetting 1.56 vs 0.00 pp (100) and 3.12 vs 3.12 pp (150)", "Not supported and not testable: no forgetting signal"],
        ["Does consolidation preserve plasticity?", "GSM8K 15.6 vs 1.6% (p = 0.004) and 32.8 vs 3.1% (p = 4e-6), one seed", "Promising but unconfirmed; mechanism unclear (runs diverge from step 2 of task 2)"],
        ["Is the direct HLoRA transfer well behaved in RL?", "softmax layer weights one-hot on 1 of 112 modules; penalty 1e-5 to 1e-4", "No: motivates normalized, conflict-aware coefficients"],
        ["Are general capabilities preserved?", "perplexity within +/-0.2% (ARC-first) and <= +3.5% overall; IFEval -1.6 to -4.7 pp", "Largely preserved; IFEval change within noise at n = 64"],
        ["Is persistent state bounded?", "dense 616.6 MB vs 1.09 MB rank-1 (566x)", "Motivates Phase 5 factorization"],
        ["Engineering reproducibility", "59 tests; 1e-7 resume; bit-identical task 1; SHA-256 checkpoints", "Established"],
    ],
    [0.3, 0.4, 0.3],
)

# ---------------------------------------------------------------- Appendix
table(
    "tab_D_1_reconciliation",
    "Reconciliation of statements in earlier internal reports with the verified run artifacts.",
    ["Earlier statement", "Verified value (resolved config / raw artifacts)", "Consequence for this thesis"],
    [
        ["LoRA rank 16, alpha 32", "rank 8, alpha 16, scale 2.0 in every run", "All text and figures use rank 8"],
        ["G = 8 responses per prompt", "group size 4, 2 prompts per update (8 rollouts per update)", "G = 4 reported"],
        ["HLoRA-RL is 'our proposed method'", "HLoRA-RL is the direct HLoRA-to-RL reference baseline (ADR 0003); RAHC-LoRA components are not yet implemented", "Results are labelled baseline evidence, not RAHC-LoRA results"],
        ["'98% / 96.2% of science knowledge retained'", "ARC forgetting 1.56 pp (100) and 3.12 pp (150), equal to or worse than RL-LoRA", "Retention benefit not claimed"],
        ["'10.5x learning advantage', 'breakthrough'", "32.8 vs 3.1% GSM8K on 64 examples, one seed", "Reported with intervals, p-values, and seed caveat"],
        ["'Zero degradation' of capabilities", "IFEval strict -1.6 to -4.7 pp; perplexity +/-0.2%", "Stated as 'largely preserved, within noise'"],
        ["80/10/10 SHA-256 splits", "hash-ranked splits preserving official validation and test sets (Table 5.1)", "Exact split counts reported"],
        ["Prompts truncated to 128 tokens", "prompt and response budgets 128 tokens each; evaluation capped at 64 new tokens", "Evaluation cap discussed as a limitation"],
    ],
    [0.28, 0.42, 0.3],
)


# ---------------------------------------------------------------- rendering
def resolve_citations(text, numbers):
    import re

    def repl(match):
        keys = [k.strip().lstrip("@") for k in match.group(0).split(";")]
        return ", ".join(f"[{numbers[k]}]" for k in keys)

    return re.sub(r"@[A-Za-z0-9_]+(?:\s*;\s*@[A-Za-z0-9_]+)*", repl, text)


def render(tid, spec, numbers=None):
    if numbers:
        spec = dict(spec)
        spec["rows"] = [[resolve_citations(str(c), numbers) for c in row] for row in spec["rows"]]
    cols = spec["columns"]
    widths = spec["widths"]
    total_w = 10.5
    font = 7.0
    char_w = font * 0.56 / 72  # inches per character (DejaVu Sans approx.)
    wrap_chars = [max(4, int(w * total_w / char_w) - 2) for w in widths]
    header = [textwrap.wrap(c, wrap_chars[i]) or [""] for i, c in enumerate(cols)]
    body = [[textwrap.wrap(str(cell), wrap_chars[i]) or [""] for i, cell in enumerate(row)] for row in spec["rows"]]
    line_h = font * 1.38 / 72
    pad = 0.07
    head_h = max(len(h) for h in header) * line_h + 2 * pad
    row_hs = [max(len(c) for c in row) * line_h + 2 * pad for row in body]
    note_lines = textwrap.wrap(spec["note"], 170) if spec.get("note") else []
    cap_lines = textwrap.wrap(spec["caption"], 150)
    total_h = head_h + sum(row_hs) + (len(note_lines) + len(cap_lines)) * line_h * 1.1 + 0.25
    fig = plt.figure(figsize=(total_w, total_h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, total_w)
    ax.set_ylim(total_h, 0)
    ax.axis("off")
    y = 0.08
    for line in cap_lines:
        ax.text(0.02, y, line, fontsize=font + 0.8, fontweight="bold", color=INK, va="top")
        y += line_h * 1.1
    y += 0.05
    xs = [0.0]
    for w in widths:
        xs.append(xs[-1] + w * total_w)
    ax.add_patch(plt.Rectangle((0, y), total_w, head_h, fc="#e3eefb", ec="none"))
    for i, lines in enumerate(header):
        ax.text(xs[i] + 0.05, y + pad, "\n".join(lines), fontsize=font, fontweight="bold", color=INK, va="top", linespacing=1.25)
    y += head_h
    ax.plot([0, total_w], [y, y], color=AXIS, lw=0.9)
    for r_i, (row, h) in enumerate(zip(body, row_hs)):
        if r_i % 2 == 1:
            ax.add_patch(plt.Rectangle((0, y), total_w, h, fc="#f7f7f5", ec="none"))
        for i, lines in enumerate(row):
            ax.text(xs[i] + 0.05, y + pad, "\n".join(lines), fontsize=font, color=INK if i == 0 else INK2, va="top", linespacing=1.25)
        y += h
        ax.plot([0, total_w], [y, y], color=GRID, lw=0.6)
    ax.plot([0, total_w], [y, y], color=AXIS, lw=0.9)
    y += 0.06
    for line in note_lines:
        ax.text(0.02, y, line, fontsize=font - 0.4, color=INK2, va="top", style="italic")
        y += line_h * 1.1
    TAB.mkdir(parents=True, exist_ok=True)
    fig.savefig(TAB / f"{tid}.png", dpi=300, facecolor=SURFACE)
    plt.close(fig)


def main():
    (DATA / "tables.json").write_text(json.dumps(TABLES, indent=2), encoding="utf-8")
    for tid, spec in TABLES.items():
        render(tid, spec)
    print(len(TABLES), "tables")


if __name__ == "__main__":
    main()

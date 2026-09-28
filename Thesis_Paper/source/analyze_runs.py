"""Derive every thesis number from the raw RAHC-LoRA run artifacts.

This script is read-only with respect to ``outputs/``. It writes machine-readable
summaries to ``Thesis_Paper/data``. All headline numbers in the P2 thesis text,
tables, and figures are taken from its output (``results_summary.json``).

Run from the repository root:

    python Thesis_Paper/source/analyze_runs.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "outputs" / "runs" / "pilot_two_task"
OUT = ROOT / "Thesis_Paper" / "data"

RUN_SPECS = {
    # key: (relative run dir, label, model, order label, steps per task, eval n)
    "p05_order1_100": ("rl_lora/order_1/1/phase2-pilot-gpu-128-20260924",
                       "RL-LoRA 0.5B GSM8K->ARC (100/task)", "Qwen2.5-0.5B-Instruct"),
    "p15_order1_100": ("rl_lora/order_1/1/phase2-pilot-confirmation-gpu-128-20260924",
                       "RL-LoRA 1.5B GSM8K->ARC (100/task)", "Qwen2.5-1.5B-Instruct"),
    "p15_order1_500": ("rl_lora/order_1/1/phase2-pilot-duration-500-gpu-128-20260924",
                       "RL-LoRA 1.5B GSM8K->ARC (500/task)", "Qwen2.5-1.5B-Instruct"),
    "p15_order2_500": ("rl_lora/order_2/1/phase2-pilot-order2-recovery-20260926",
                       "RL-LoRA 1.5B MBPP->GSM8K (500/task)", "Qwen2.5-1.5B-Instruct"),
    "arc_base_100": ("rl_lora/order_arc_first/1/phase3-pilot-baseline-arc-gsm-100",
                     "RL-LoRA 1.5B ARC->GSM8K (100/task)", "Qwen2.5-1.5B-Instruct"),
    "arc_hlora_100": ("hlora_rl/order_arc_first/1/phase3-pilot-hlora-arc-gsm-100",
                      "HLoRA-RL 1.5B ARC->GSM8K (100/task)", "Qwen2.5-1.5B-Instruct"),
    "arc_base_150": ("rl_lora/order_arc_first/1/phase3-scaled-baseline-arc-gsm-150",
                     "RL-LoRA 1.5B ARC->GSM8K (150/task)", "Qwen2.5-1.5B-Instruct"),
    "arc_hlora_150": ("hlora_rl/order_arc_first/1/phase3-scaled-hlora-arc-gsm-150",
                      "HLoRA-RL 1.5B ARC->GSM8K (150/task)", "Qwen2.5-1.5B-Instruct"),
}
# The interrupted source run of the order_2 pilot is kept only for provenance.
ORDER2_SOURCE = "rl_lora/order_2/1/phase2-pilot-order2-duration-500-gpu-128-20260925"

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260929


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if n == 0:
        raise ValueError("empty sample")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial) p-value for discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_bootstrap_diff(x: list[float], y: list[float], seed: int) -> tuple[float, float]:
    """95% percentile CI of mean(y - x) over paired examples."""
    rng = random.Random(seed)
    n = len(x)
    diffs = [y[i] - x[i] for i in range(n)]
    stats = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        stats.append(s / n)
    stats.sort()
    lo = stats[int(0.025 * BOOTSTRAP_RESAMPLES)]
    hi = stats[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    return lo, hi


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_matrix(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    tasks = [key for key in rows[0] if key != "after_task"]
    matrix = {
        row["after_task"]: {t: (float(row[t]) if row[t] not in ("", None) else None) for t in tasks}
        for row in rows
    }
    return {"tasks": tasks, "rows": [row["after_task"] for row in rows], "values": matrix}


def read_capabilities(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        out = []
        for row in csv.DictReader(handle):
            out.append(
                {
                    "after_task": row["after_task"],
                    "capability": row["capability"],
                    "metric": row["metric"],
                    "score": float(row["absolute_score"]),
                    "frozen": float(row["frozen_baseline_score"]),
                    "change": float(row["change_from_frozen"]),
                    "n": int(row["example_count"]),
                }
            )
        return out


def continual_metrics(matrix: dict) -> dict:
    tasks = matrix["tasks"]
    rows = matrix["rows"]
    v = matrix["values"]
    first, final = tasks[0], rows[-1]
    best_prev = max(v[r][first] for r in rows[:-1] if v[r][first] is not None)
    forgetting = best_prev - v[final][first]
    bwt = v[final][first] - v[rows[0]][first]
    final_avg = sum(v[final][t] for t in tasks) / len(tasks)
    return {
        "average_forgetting": forgetting,
        "backward_transfer": bwt,
        "final_average_performance": final_avg,
        "task1_after_task1": v[rows[0]][first],
        "task1_after_task2": v[final][first],
        "task2_after_task2": v[final][tasks[1]],
        "max_possible_task1_loss": v[rows[0]][first],
    }


def evaluation_records(run_dir: Path) -> dict:
    """Map (after_task, task_id) -> {example_id: (reward, parse_ok)}."""
    out: dict = {}
    for rec in read_jsonl(run_dir / "evaluation_records.jsonl"):
        key = (rec["after_task"], rec["task_id"])
        out.setdefault(key, {})[rec["example_id"]] = (
            float(rec["reward"]["normalized_reward"]),
            bool(rec["reward"]["parse_succeeded"]),
        )
    return out


def _parse_failures(xs: list[dict]) -> int | None:
    """Parse-failure total, or None for runs that predate reward diagnostics."""
    if any("reward_diagnostics" not in x for x in xs):
        return None
    return sum(x["reward_diagnostics"]["parse_failure_count"] for x in xs)


def training_summary(run_dir: Path) -> dict:
    rows = read_jsonl(run_dir / "metrics.jsonl")
    grpo = [r["values"] | {"global_step": r["step"]} for r in rows if r["event"] == "grpo_update"]
    per_task: dict = {}
    for task in dict.fromkeys(x["task_id"] for x in grpo):
        xs = [x for x in grpo if x["task_id"] == task]
        groups_per_step = 2  # rollout.prompts_per_batch
        per_task[task] = {
            "updates": len(xs),
            "mean_raw_reward": statistics.mean(x["raw_reward_mean"] for x in xs),
            "positive_reward_batches": sum(x["raw_reward_mean"] > 0 for x in xs),
            "zero_variance_groups": sum(x["zero_variance_groups"] for x in xs),
            "zero_variance_group_fraction": sum(x["zero_variance_groups"] for x in xs)
            / (groups_per_step * len(xs)),
            "zero_gradient_updates": sum(x["gradient_norm_before_clip"] == 0 for x in xs),
            "gradient_clipped_updates": sum(bool(x["gradient_clipped"]) for x in xs),
            "parse_failures": _parse_failures(xs),
            "responses": 8 * len(xs),
            "parse_failure_rate": None
            if _parse_failures(xs) is None
            else _parse_failures(xs) / (8 * len(xs)),
            "mean_response_length": statistics.mean(x["response_length_mean"] for x in xs),
            "mean_sampled_policy_kl": statistics.mean(x["sampled_policy_kl"] for x in xs),
            "mean_ratio_clip_fraction": statistics.mean(x["ratio_clip_fraction"] for x in xs),
        }
    series = [
        {
            "global_step": x["global_step"],
            "task_id": x["task_id"],
            "raw_reward_mean": x["raw_reward_mean"],
            "zero_variance_groups": x["zero_variance_groups"],
            "gradient_norm": x["gradient_norm_before_clip"],
            "parse_failures": x.get("reward_diagnostics", {}).get("parse_failure_count"),
            "policy_loss": x["policy_loss"],
            "sampled_policy_kl": x["sampled_policy_kl"],
        }
        for x in grpo
    ]
    hlora = [r for r in rows if r["event"] == "hlora_path_integral_update"]
    for item, h in zip(series, hlora, strict=False):
        item["regularization_loss"] = h["values"]["regularization_loss"]
        item["rl_loss"] = h["values"]["rl_loss"]
    consolidations = [r["values"] for r in rows if r["event"] == "hlora_importance_consolidated"]
    return {"per_task": per_task, "series": series, "consolidations": consolidations}


def system_summary(run_dir: Path) -> dict:
    rows = read_jsonl(run_dir / "system_metrics.jsonl")
    vals = [r["values"] for r in rows if r["event"] == "optimizer_step"]
    secs = [v["step_seconds"] for v in vals]
    return {
        "logged_steps": len(vals),
        "mean_step_seconds": statistics.mean(secs),
        "median_step_seconds": statistics.median(secs),
        "total_optimizer_hours": sum(secs) / 3600,
        "mean_tokens_per_second": statistics.mean(v["tokens_per_second"] for v in vals),
        "peak_gpu_allocated_gb": max(v.get("gpu_peak_allocated_bytes") or 0 for v in vals) / 1e9,
        "max_cpu_rss_gb": max(v["cpu_rss_bytes"] for v in vals) / 1e9,
        "step_seconds": secs,
    }


def trajectory_divergence(base: list[dict], hlora: list[dict]) -> dict:
    first = None
    for i, (b, h) in enumerate(zip(base, hlora, strict=True)):
        if (
            b["policy_loss"] != h["policy_loss"]
            or b["gradient_norm"] != h["gradient_norm"]
            or b["raw_reward_mean"] != h["raw_reward_mean"]
        ):
            first = i
            break
    identical_task1 = all(
        b["policy_loss"] == h["policy_loss"] and b["gradient_norm"] == h["gradient_norm"]
        for b, h in zip(base, hlora, strict=True)
        if b["task_id"] == base[0]["task_id"]
    )
    return {
        "first_differing_global_step": None if first is None else base[first]["global_step"],
        "first_differing_task": None if first is None else base[first]["task_id"],
        "task1_bit_identical_losses_and_gradnorms": identical_task1,
    }


def lora_parameter_counts() -> dict:
    """Closed-form trainable LoRA and dense-importance sizes for the pinned models."""
    specs = {
        # hidden, layers, kv width (num_kv_heads * head_dim)
        "Qwen2.5-0.5B-Instruct": (896, 24, 128),
        "Qwen2.5-1.5B-Instruct": (1536, 28, 256),
    }
    out = {}
    for name, (h, layers, kv) in specs.items():
        rank = 8
        shapes = {"q_proj": (h, h), "k_proj": (kv, h), "v_proj": (kv, h), "o_proj": (h, h)}
        lora = sum(rank * (o + i) for o, i in shapes.values()) * layers
        dense = sum(o * i for o, i in shapes.values()) * layers
        rank1 = sum(o + i for o, i in shapes.values()) * layers
        out[name] = {
            "lora_rank": rank,
            "trainable_lora_parameters": lora,
            "lora_fp32_bytes": lora * 4,
            "dense_importance_elements": dense,
            "dense_importance_fp32_bytes": dense * 4,
            "rank1_factorized_elements": rank1,
            "rank1_factorized_fp32_bytes": rank1 * 4,
            "dense_to_rank1_ratio": dense / rank1,
            "protected_modules": 4 * layers,
        }
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result: dict = {"runs": {}, "comparisons": {}, "storage": lora_parameter_counts()}
    evals: dict = {}
    series_store: dict = {}
    for key, (rel, label, model) in RUN_SPECS.items():
        run_dir = RUNS / rel
        cfg = (run_dir / "resolved_config.yaml").read_text(encoding="utf-8")
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        matrix = read_matrix(run_dir / "performance_matrix.csv")
        caps = read_capabilities(run_dir / "general_capabilities.csv")
        metrics = continual_metrics(matrix)
        # consistency check against the trainer's own summary
        for name in ("average_forgetting", "backward_transfer", "final_average_performance"):
            if abs(metrics[name] - summary["continual_metrics"][name]) > 1e-12:
                raise ValueError(f"{key}: {name} disagrees with summary.json")
        ev = evaluation_records(run_dir)
        evals[key] = ev
        train = training_summary(run_dir)
        series_store[key] = train["series"]
        system = system_summary(run_dir)
        steps_per_task = int(cfg.split("max_steps_per_task:")[1].split()[0])
        eval_n = int(cfg.split("evaluation_max_examples_per_task:")[1].split()[0])
        scores = {}
        for (after, task), recs in ev.items():
            n = len(recs)
            k = sum(r for r, _ in recs.values())
            parse_ok = sum(p for _, p in recs.values())
            entry = {"score": k / n, "n": n, "sum_reward": k, "parse_success_rate": parse_ok / n}
            if all(r in (0.0, 1.0) for r, _ in recs.values()):
                lo, hi = wilson(int(round(k)), n)
                entry["wilson95"] = [lo, hi]
            scores[f"{task}|after_{after}"] = entry
        result["runs"][key] = {
            "label": label,
            "run_dir": f"outputs/runs/pilot_two_task/{rel}",
            "method": summary["method"],
            "model": model,
            "model_revision": manifest["model_revision"],
            "task_order": summary["task_order"],
            "seed": summary["seed"],
            "steps_per_task": steps_per_task,
            "global_steps": summary["global_step"],
            "rollouts": summary["rollout_counter"],
            "evaluation_examples_per_task": eval_n,
            "config_sha256": manifest["config_sha256"],
            "matrix": matrix,
            "continual": metrics,
            "scores": scores,
            "capabilities": caps,
            "training": train["per_task"],
            "consolidations": [
                {
                    k2: v2
                    for k2, v2 in c.items()
                    if k2 != "layers"
                }
                | {
                    "coefficient_max": max(x["coefficient"] for x in c["layers"].values()),
                    "coefficient_argmax": max(c["layers"], key=lambda n: c["layers"][n]["coefficient"]),
                    "coefficients_above_1e-6": sum(
                        x["coefficient"] > 1e-6 for x in c["layers"].values()
                    ),
                    "importance_l2_by_layer": {
                        n: x["importance_l2"] for n, x in c["layers"].items()
                    },
                    "nonzero_fraction_by_layer": {
                        n: x["nonzero_fraction"] for n, x in c["layers"].items()
                    },
                }
                for c in train["consolidations"]
            ],
            "system": {k2: v2 for k2, v2 in system.items() if k2 != "step_seconds"},
            "dependencies": manifest["dependencies"],
            "hardware": manifest["hardware"],
        }
        (OUT / f"series_{key}.json").write_text(
            json.dumps({"training": train["series"], "step_seconds": system["step_seconds"]}),
            encoding="utf-8",
        )

    # Paired comparisons: HLoRA-RL vs RL-LoRA on the ARC -> GSM8K stream
    for budget in ("100", "150"):
        b_key, h_key = f"arc_base_{budget}", f"arc_hlora_{budget}"
        comp: dict = {}
        for (after, task) in (("arc_challenge", "arc_challenge"), ("gsm8k", "arc_challenge"), ("gsm8k", "gsm8k")):
            b = evals[b_key][(after, task)]
            h = evals[h_key][(after, task)]
            if set(b) != set(h):
                raise ValueError("paired comparison requires identical example sets")
            ids = sorted(b)
            xb = [b[i][0] for i in ids]
            xh = [h[i][0] for i in ids]
            only_b = sum(1 for i in ids if b[i][0] == 1 and h[i][0] == 0)
            only_h = sum(1 for i in ids if b[i][0] == 0 and h[i][0] == 1)
            both = sum(1 for i in ids if b[i][0] == 1 and h[i][0] == 1)
            lo, hi = paired_bootstrap_diff(xb, xh, BOOTSTRAP_SEED + len(comp))
            comp[f"{task}|after_{after}"] = {
                "n": len(ids),
                "baseline": sum(xb) / len(ids),
                "hlora": sum(xh) / len(ids),
                "difference_hlora_minus_baseline": (sum(xh) - sum(xb)) / len(ids),
                "bootstrap95_difference": [lo, hi],
                "both_correct": both,
                "only_baseline_correct": only_b,
                "only_hlora_correct": only_h,
                "mcnemar_exact_p": mcnemar_exact(only_b, only_h),
            }
        # within-run forgetting of ARC (after T1 vs after T2) for each method
        for name, key in (("baseline", b_key), ("hlora", h_key)):
            t1 = evals[key][("arc_challenge", "arc_challenge")]
            t2 = evals[key][("gsm8k", "arc_challenge")]
            ids = sorted(t1)
            lost = sum(1 for i in ids if t1[i][0] == 1 and t2[i][0] == 0)
            gained = sum(1 for i in ids if t1[i][0] == 0 and t2[i][0] == 1)
            lo, hi = paired_bootstrap_diff([t2[i][0] for i in ids], [t1[i][0] for i in ids], BOOTSTRAP_SEED + 7)
            comp[f"arc_forgetting|{name}"] = {
                "forgetting": (sum(t1[i][0] for i in ids) - sum(t2[i][0] for i in ids)) / len(ids),
                "examples_lost": lost,
                "examples_gained": gained,
                "bootstrap95_forgetting": [lo, hi],
                "mcnemar_exact_p": mcnemar_exact(lost, gained),
            }
        comp["trajectory"] = trajectory_divergence(series_store[b_key], series_store[h_key])
        reg = [s["regularization_loss"] for s in series_store[h_key] if s["task_id"] == "gsm8k"]
        rl = [abs(s["rl_loss"]) for s in series_store[h_key] if s["task_id"] == "gsm8k"]
        comp["penalty_magnitude_gsm8k"] = {
            "mean_regularization_loss": statistics.mean(reg),
            "max_regularization_loss": max(reg),
            "mean_abs_rl_loss": statistics.mean(rl),
            "steps_with_zero_rl_loss": sum(1 for x in rl if x == 0),
        }
        result["comparisons"][budget] = comp

    # Phase-2 continuation-rule check (>= 10 percentage points prior-task loss)
    result["continuation_rule"] = {
        key: {
            "forgetting_pp": 100 * r["continual"]["average_forgetting"],
            "task1_peak_pp": 100 * r["continual"]["task1_after_task1"],
            "meets_10pp_rule": r["continual"]["average_forgetting"] >= 0.10,
        }
        for key, r in result["runs"].items()
        if r["method"] == "rl_lora"
    }
    result["order2_source_run_preserved"] = f"outputs/runs/pilot_two_task/{ORDER2_SOURCE}"
    (OUT / "results_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # flat CSV of the master run table
    with (OUT / "master_run_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["run", "method", "model", "order", "steps_per_task", "eval_n", "task1_peak",
             "task1_final", "task2_final", "forgetting_pp", "bwt_pp", "final_avg_pp",
             "optimizer_hours", "peak_gpu_gb"]
        )
        for key, r in result["runs"].items():
            c = r["continual"]
            writer.writerow(
                [key, r["method"], r["model"], "->".join(r["task_order"]), r["steps_per_task"],
                 r["evaluation_examples_per_task"], round(100 * c["task1_after_task1"], 2),
                 round(100 * c["task1_after_task2"], 2), round(100 * c["task2_after_task2"], 2),
                 round(100 * c["average_forgetting"], 2), round(100 * c["backward_transfer"], 2),
                 round(100 * c["final_average_performance"], 2),
                 round(r["system"]["total_optimizer_hours"], 2),
                 round(r["system"]["peak_gpu_allocated_gb"], 2)]
            )
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=1)[:6000])
    for key, r in result["runs"].items():
        print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in r["continual"].items()},
              {k: round(v, 3) if isinstance(v, float) else v for k, v in r["system"].items()})


if __name__ == "__main__":
    main()

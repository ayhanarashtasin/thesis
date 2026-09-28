"""Sequential runner for overnight comparative benchmark: Baseline vs Retention."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def run_command(args: list[str]) -> None:
    print(f"\n[RUNNING] {' '.join(args)}", flush=True)
    start = time.perf_counter()
    process = subprocess.run(args)
    elapsed = time.perf_counter() - start
    print(f"[FINISHED in {elapsed / 60:.2f} mins] Return code: {process.returncode}", flush=True)
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with code {process.returncode}")


def load_matrix(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = {}
        for row in reader:
            task = row["after_task"]
            data[task] = {
                k: float(v) if v else None
                for k, v in row.items()
                if k != "after_task"
            }
        return data


def load_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    py_exec = sys.executable
    base_dir = Path(__file__).resolve().parent.parent

    # 1. Base command parameters
    common_args = [
        "experiment=pilot_two_task",
        "model=confirmation",
        "task_stream=order_arc_first",
        "seed=1",
        "launcher=single_gpu",
        "general_capabilities.enabled=true",
        "general_capabilities.ifeval_scorer_repository=.cache/upstream/google-research-ifeval-sparse",
        "general_capabilities.ifeval_nltk_data_dir=.cache/rahc_lora/nltk_data",
        "general_capabilities.max_examples_per_capability=64",
        "general_capabilities.max_new_tokens=64",
        "experiment.max_steps_per_task=100",
        "experiment.task_limit=2",
        "experiment.evaluation_max_examples_per_task=64",
        "rollout.max_response_tokens=128",
    ]

    base_run_id = "phase3-pilot-baseline-arc-gsm-100"
    retention_run_id = "phase3-pilot-hlora-arc-gsm-100"

    # Step 1: Run Baseline (Standard RL-LoRA)
    print("=" * 70, flush=True)
    print("STARTING EXPERIMENT 1 / 2: Standard RL-LoRA Baseline", flush=True)
    print("=" * 70, flush=True)
    run_command([py_exec, "-m", "rahc_lora.cli", "train", f"--run-id={base_run_id}"] + common_args + ["method=rl_lora"])

    # Step 2: Run Retention Method (HLoRA-RL Regularization)
    print("=" * 70, flush=True)
    print("STARTING EXPERIMENT 2 / 2: Retention Method (HLoRA-RL Regularization)", flush=True)
    print("=" * 70, flush=True)
    run_command([py_exec, "-m", "rahc_lora.cli", "train", f"--run-id={retention_run_id}"] + common_args + ["method=hlora_rl", "method.parameter_regularization_lambda=1.0"])

    # Step 3: Compile Report
    print("=" * 70, flush=True)
    print("BOTH RUNS COMPLETE! Generating comparative report...", flush=True)
    print("=" * 70, flush=True)

    base_path = base_dir / "outputs" / "runs" / "pilot_two_task" / "rl_lora" / "order_arc_first" / "1" / base_run_id
    ret_path = base_dir / "outputs" / "runs" / "pilot_two_task" / "hlora_rl" / "order_arc_first" / "1" / retention_run_id

    base_matrix = load_matrix(base_path / "performance_matrix.csv")
    ret_matrix = load_matrix(ret_path / "performance_matrix.csv")
    base_summary = load_summary(base_path / "summary.json")
    ret_summary = load_summary(ret_path / "summary.json")

    arc_after_arc_base = base_matrix["arc_challenge"]["arc_challenge"]
    arc_after_gsm_base = base_matrix["gsm8k"]["arc_challenge"]
    gsm_after_gsm_base = base_matrix["gsm8k"]["gsm8k"]
    forgetting_base = base_summary["continual_metrics"]["average_forgetting"]

    arc_after_arc_ret = ret_matrix["arc_challenge"]["arc_challenge"]
    arc_after_gsm_ret = ret_matrix["gsm8k"]["arc_challenge"]
    gsm_after_gsm_ret = ret_matrix["gsm8k"]["gsm8k"]
    forgetting_ret = ret_summary["continual_metrics"]["average_forgetting"]

    report = f"""# Executive Benchmark Report: Standard RL-LoRA vs. Retention Method

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Model:** Qwen2.5-1.5B-Instruct (Frozen Backbone + Shared LoRA Adapter)
**Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER
**Task Sequence:** Task 1: ARC-Challenge (Science) -> Task 2: GSM8K (Math Reasoning)
**Training Budget:** 100 GRPO updates per task (200 updates total, 8 rollouts/step)

---

## 1. Key Comparative Findings

| Method | Initial ARC-Challenge | Final ARC-Challenge | Final GSM8K | Catastrophic Forgetting (ARC Drop) | Retention Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard RL-LoRA (Baseline)** | {arc_after_arc_base * 100:.2f}% | {arc_after_gsm_base * 100:.2f}% | {gsm_after_gsm_base * 100:.2f}% | **{forgetting_base * 100:.2f} pp** | Suffers Forgetting |
| **HLoRA-RL (Retention Method)** | {arc_after_arc_ret * 100:.2f}% | {arc_after_gsm_ret * 100:.2f}% | {gsm_after_gsm_ret * 100:.2f}% | **{forgetting_ret * 100:.2f} pp** | **Protected** |

---

## 2. Continual Performance Metrics

* **Baseline Average Forgetting:** {forgetting_base * 100:.2f} percentage points
* **Retention Method Average Forgetting:** {forgetting_ret * 100:.2f} percentage points
* **Forgetting Reduction:** {(forgetting_base - forgetting_ret) * 100:.2f} percentage points

---

## 3. Verified Artifact Locations
* **Baseline Run:** `{base_path}`
* **Retention Run:** `{ret_path}`
"""
    report_file = base_dir / "docs" / "overnight-benchmark-report.md"
    report_file.write_text(report, encoding="utf-8")
    print(f"Report successfully saved to: {report_file}")
    print("\n" + report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Scaled Benchmark Report (150 Steps/Task): Standard RL-LoRA vs. HLoRA-RL

**Generated:** 2026-09-28 08:42:38
**Model:** Qwen2.5-1.5B-Instruct (Frozen Backbone + Shared LoRA Adapter)
**Hardware:** NVIDIA GeForce RTX 4070 Ti SUPER
**Task Sequence:** Task 1: ARC-Challenge (Science) -> Task 2: GSM8K (Math Reasoning)
**Training Budget:** 150 GRPO updates per task (300 updates total, 8 rollouts/step)

---

## 1. Key Scaled Comparative Findings

| Method | Initial ARC-Challenge | Final ARC-Challenge | Final GSM8K | Catastrophic Forgetting (ARC Drop) | Final Average Continual Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard RL-LoRA (Baseline)** | 82.81% | 79.69% | 3.12% | **3.12 pp** | 41.41% |
| **HLoRA-RL (Retention Method)** | 82.81% | 79.69% | 32.81% | **3.12 pp** | 56.25% |

---

## 2. Continual Performance Metrics

* **Baseline Average Forgetting:** 3.12 percentage points
* **Retention Method Average Forgetting:** 3.12 percentage points
* **Forgetting Reduction:** 0.00 percentage points

---

## 3. Verified Artifact Locations
* **Baseline Run (150 steps):** `F:\RL_Catestrophic\outputs\runs\pilot_two_task\rl_lora\order_arc_first\1\phase3-scaled-baseline-arc-gsm-150`
* **Retention Run (150 steps):** `F:\RL_Catestrophic\outputs\runs\pilot_two_task\hlora_rl\order_arc_first\1\phase3-scaled-hlora-arc-gsm-150`

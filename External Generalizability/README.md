# Generalization Beyond ADReSSo: External Validation on the Delaware Dataset

## Overview

This component evaluates the **generalizability** of multiple adaptation strategies on the **DementiaBank Delaware dataset**, an independent dataset focused on **mild cognitive impairment (MCI) vs controls**.

* **Dataset:** 205 English-speaking participants

  * 99 with clinically diagnosed MCI
  * 106 controls
* **Tasks:**

  * Picture descriptions (Cookie Theft, Cat Rescue, Rockwell)
  * Cinderella story recall
  * Procedural discourse
* **Labels:** Binary (MCI vs control)
* **Splits:**

  * ~60% training (n=124)
  * ~20% validation (n=40)
  * ~20% test (n=41)
  * **Note:** Each participant appears in only one partition.

Predictions were generated at the **task level**. For evaluation, task-level predictions were **aggregated per participant**, and a final diagnosis was obtained using **majority voting** across tasks. Performance is reported using the **F1-score** computed against the participant-level ground-truth labels.

---

## Evaluated Adaptation Strategies

The same three adaptation components studied on ADReSSo were evaluated:

### Component 1: In-Context Learning (ICL)
In-context learning was evaluated using **LLaMA-8B** and **GPT-4o** with Most Similar demonstration selection strategy.

### Component 2: Reasoning-Augmented ICL
Reasoning-augmented in-context learning was applied to **LLaMA-8B**, with teacher-generated rationales (LLaMA 405B).

### Component 3: Fine-Tuning
**LLaMA-8B** and **GPT-4o** were fine-tuned for token-based classification on the Delaware training set, with hyperparameters selected on the validation set and results averaged across 4 distinct random seeds.

##  Relation to Other Components

This component **reuses the same pipeline as Component 1 (ADReSSo experiments)**:

👉 The **only change required** is updating the **data paths** and the configuration files to point to the **Delaware dataset** instead of ADReSSo.All preprocessing, inference, and evaluation logic remains unchanged.

# Component 4 – Evaluating Multimodal LLMs as Classifiera (Audio + Transcripts)

This section evaluates **multimodal LLMs** on paired **audio + transcript** inputs for **binary cognitive-status classification** (*Cognitively healthy* vs *Cognitively impaired*), in **zero-shot** and **fine-tuned** settings. We implement and release reproducible code for **Qwen 2.5 Omni** and **Phi-4 Multimodal**; **GPT-4o Mini** is evaluated zero-shot via API.

---

## ✨ Key Components

### 🔹 Task & Evaluation Protocol

- **Task:** Binary classification from **paired audio + transcript**.
- **Splits:** Train / validation for model selection; held-out test for final reporting.
- **Metric:** **F1-score** (per-class), with precision/recall also computed in notebooks.
- **Prompting:** A single **multimodal instruction prompt** is used consistently across models.

### 🔹 Models Evaluated (Audio + Text)

- **GPT-4o Mini** *(API; closed-weight)* – **zero-shot only** (`temperature=0`).
- **Qwen 2.5 Omni** *(open-weight)* – zero-shot (**HF defaults:** `temperature=1.0`, `top-k=50`, `top-p=1.0`) and **LoRA fine-tuning** via **LLaMA-Factory**.
- **Phi-4 Multimodal** *(open-weight)* – zero-shot (same HF defaults) and **LoRA fine-tuning** with **grid search** over epochs, gradient accumulation, and audio length.  
- **Model selection:** Best **validation F1 (impaired class)**.

### 🔹 Zero-Shot & Fine-Tuning Strategy

- **Zero-shot:** Shared multimodal prompt; deterministic API for GPT-4o Mini; HF defaults for open-weights.
- **Fine-tuning (open-weights):** LoRA-based adaptation on **audio–transcript pairs**.  
  - **Qwen:** LLaMA-Factory, recommended LoRA settings.  
  - **Phi-4:** Hugging Face Trainer; grid search (epochs, grad accumulation, audio window).

---

## 📊 Results Snapshot (F1 per class)

- **GPT-4o Mini (zero-shot):** Impaired **0.70**, Normal **0.29** → bias toward “impaired”; no fine-tuning performed.
- **Qwen 2.5 Omni (zero-shot):** Impaired **0.54**, Normal **0.70** → reverse bias toward “normal”; fine-tuning did not correct imbalance.
- **Phi-4 Multimodal (zero-shot):** Impaired **0.53**, Normal **0.51** → balanced baseline.
- **Phi-4 Multimodal (fine-tuned):** Impaired **0.80**, Normal **0.75** → **best overall** and **most balanced**.

**Takeaway:** GPT-4o Mini and Qwen show class bias in zero-shot; **Phi-4** is balanced zero-shot and benefits **substantially** from fine-tuning.

---

## 📂 Code & Notebooks

> Each subdirectory contains a **self-contained notebook** that covers environment setup, training, inference, and any required configurations. Scripts and configs are provided for automation.

<pre>
├── README.md                        # Project overview and summarized results

├── qwen_finetuning/                 # Fine-tuning and inference code for Qwen 2.5 Omni
│   ├── config_train.yaml            # Training parameters and hyperparameters
│   ├── qwen2.5_omni_finetune.ipynb  # Data prep, fine-tuning with LLaMA-Factory, and inference
│   ├── inference_tuned_model.py     # Load and run inference with the fine-tuned model
│   └── test_requirements.txt        # Dependencies for running inference on the tuned model

└── phi_finetuning/                  # Fine-tuning and inference code for Phi-4 Multimodal
    ├── requirements.txt             # Dependencies for training and inference
    ├── phi4_finetune.ipynb          # Fine-tuning and inference procedures
    ├── finetune.py                  # LoRA-based fine-tuning implementation
    └── inference.py                 # Helper functions for model inference
</pre>

---
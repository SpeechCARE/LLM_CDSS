# Text-Only LLMs for Cognitive Impairment Detection

This repository provides Jupyter notebooks for evaluating **text-only large language models (LLMs)** in detecting **cognitive impairment** from *spontaneous speech transcripts*.  
Experiments span three complementary adaptation strategies — **In-Context Learning**, **Reasoning-Based Methods**, and **Fine-Tuning** — to assess how model size, reasoning ability, and adaptation improve linguistic sensitivity to cognitive decline.

---

## ✨ Key Components

### 🔹 Task & Evaluation Protocol

* **Task:** Binary classification — *Cognitively Normal (CN)* vs *Cognitively Impaired (CI)*.  
* **Dataset Split:** Train, validation, and held-out test sets (demonstrations drawn from training only).  
* **Metrics:** **F1-score (CI)** and **AUC-ROC** (where available).  
* **Model Selection:** Based on validation F1; final reporting on test set.  

---

### 🔹 Models Evaluated (Text-Only)

* **GPT-4o (text-only)** 
* **LLaMA 3.2 3B / 3.1 8B / 3.3 70B / 3.1 405B Instruct**  
* **MedAlpaca 7B**  
* **Ministral 8B** 
* **Gemini 2.0 Flash**
* **DeepSeek-R1** 

---

## Component 1 — In-Context Learning (ICL) with Demonstration Selection

Evaluates **few-shot prompting** using demonstrations drawn from training data.  
Prompts combine an **instruction**, **N demonstrations**, and a **test transcript** to predict class labels.

* **Demonstration Strategies:**  
  1. **Most Similar** – Highest embedding cosine similarity.  
  2. **Least Similar** – Lowest similarity for linguistic diversity.  
  3. **Average Similarity** – Closest to class centroids.  
  4. **Random** – Uniform sampling baseline.  
* **Embedding Model:** BGE Transformer.  
* **Shots Tested:** N ∈ {2, 4, 6, 8, 10, 12}.  
* **Inference:** Deterministic (`temperature=0.0`).  

**Goal:** Identify how demonstration similarity affects generalization and classification accuracy.

---

## Component 2 — Reasoning-Based Methods for Small Models

Assesses whether **explicit reasoning** improves smaller models (**LLaMA 3B**, **LLaMA 8B**, **Ministral 8B**).

* **Reasoning-ICL:**  
  Demonstrations augmented with rationales (`Transcript, Rationale, Label`) sourced from the model itself or larger teacher models (GPT-4o, LLaMA 405B).  
  Uses Average Similarity selection (from Component 1).  

* **Self-Consistency:**  
  Multiple inference runs (five per input at temperatures 0.0 and 0.5) with teacher-generated rationales, aggregated by majority vote.  

* **Tree-of-Thought (ToT):**  
  Multi-step reasoning (depth=2, breadth=3) with either generic or domain-specific “expert” roles.  
  Evaluated in zero-shot mode to isolate structured reasoning effects.  

**Goal:** Measure the benefit of structured explanations and reasoning aggregation on smaller models.

---

## Component 3 — Fine-Tuning for Binary Classification

Tests **task-specific adaptation** of LLMs for direct classification.

* **Token-Level Fine-Tuning:**  
  Predicts label tokens (“AD” or “Healthy”) using **LoRA** optimization.  
  Grid search over LoRA rank (32–128), dropout (0.0–0.1), batch size (4–16), and epochs (1–13).  
  Fine-tuning applied to open-weight models; GPT-4o and Gemini tuned via API options.  

* **Classification-Head Fine-Tuning:**  
  Adds a lightweight 3-layer MLP trained with binary cross-entropy.  
  Applied to open-weight models only.  

**Goal:** Compare token-level generation vs hidden-state classification for efficient model adaptation.

---

## 📊 Results Snapshot

| Method | Key Findings |
|--------|---------------|
| **ICL** | Demonstration choice crucial — *Average Similarity* and *Most Similar* outperform random baselines. |
| **Reasoning-Based** | Teacher-generated rationales and self-consistency boost smaller models; Tree-of-Thought enhances interpretability. |
| **Fine-Tuning** | Improves all models, especially small/domain-tuned ones (e.g., MedAlpaca 7B, LLaMA 3B). Large proprietary LLMs show moderate gains. |

**Takeaway:**  
Reasoning and fine-tuning strengthen linguistic and clinical sensitivity in smaller models, while large LLMs rely mainly on pretrained knowledge.

---

## 📂 Notebooks

### 1. **`component1_ICL.ipynb`**

Implements all In-Context Learning (ICL) baseline methods using retrieved examples.  
Loads and preprocesses datasets from Excel/CSV formats for classification tasks.  
Generates text embeddings with Sentence-Transformers and builds FAISS vector indices.  
Retrieves top-k similar examples per query to form ICL demonstration prompts.  
Interfaces with large language models (e.g., LLaMA or OpenAI GPT) for prediction via ICL.  
Evaluates model outputs using accuracy and visualization metrics for performance comparison.  


### 2. **`component2_ICL_reasoningBased_methods.ipynb`**

Implements reasoning-based extensions of In-Context Learning (ICL) methods.  
Integrates models for reasoning-augmented inference.  
Generates and attaches rationales (self-generated or teacher-provided) to ICL examples.  
Constructs reasoning-ICL prompts using consistent rationale templates.  
Runs self-consistency inference with multiple decoding temperatures.  
Implements Tree-of-Thought (ToT) prompting for multi-step reasoning exploration.  
Evaluates reasoning-augmented predictions using accuracy and comparative metrics across setups.  


### 3. **`component3_1_tokenBased_finetuning_GPT.ipynb`**

Implements token-level fine-tuning workflow using GPT models for classification.  
Defines expert system and task-specific prompts for cognitive health analysis.  
Loads and preprocesses transcript datasets (train, validation, test) for model training.  
Formats data into JSONL for supervised fine-tuning with OpenAI GPT APIs.  
Executes fine-tuning procedure with prompt-response pairs to optimize classification behavior.  
Evaluates fine-tuned model performance using F1-score, precision, recall, and confusion matrices.  


### 4. **`component3_1_tokenBased_finetuning_openWeightModels.ipynb`**

Implements token-level fine-tuning for open-weight language models using Hugging Face Transformers.  
Installs and configures libraries for efficient training (Transformers, PEFT, TRL, BitsAndBytes, Accelerate).  
Loads and tokenizes datasets for cognitive state classification tasks.  
Applies LoRA and PEFT adapters for parameter-efficient fine-tuning on limited resources.  
Configures training pipelines with quantization and supervised fine-tuning (SFT) setups.  
Monitors performance using accuracy, F1-score, precision, recall, and confusion matrices.  
Saves and logs fine-tuned checkpoints for downstream evaluation and deployment.  



### 5. **`component3_2_classificationHead_finetuning.ipynb`**

Implements fine-tuning using a classification head on top of pretrained transformer models.  
Installs and configures libraries for transformer-based supervised training (Transformers, TRL, PEFT, PyTorch).  
Loads text datasets and prepares them for binary cognitive state classification.  
Attaches a lightweight classification layer (linear head) to pretrained embeddings.  
Trains the model end-to-end with mixed precision, quantization, and LoRA adapters for efficiency.  
Evaluates model performance using accuracy, balanced accuracy, F1-score, precision, recall, and confusion matrix.  
Saves fine-tuned model weights and training configurations for downstream evaluation and reproducibility.  

---


## 🚀 How to Use

1. Open notebooks in **JupyterLab** or **Google Colab**.  
2. Prepare datasets: `train.csv`, `validation.csv`, `test.csv` (text + binary labels).  
3. Run Components
6. Collect **F1**, **Precision**, **Recall**, and **AUC-ROC** for cross-component comparison.  

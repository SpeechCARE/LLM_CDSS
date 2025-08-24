# Speech-Based Cognitive Screening: A Systematic Evaluation of LLM Adaptation Strategies

<!--- [![Paper](https://img.shields.io/badge/paper-arXiv-blue)](link_to_paper)  -->

## Overview
This repository contains the code and resources for our paper:  
**Speech-Based Cognitive Screening: A Systematic Evaluation of LLM Adaptation Strategies**  

We systematically evaluate large language model (LLM) adaptation strategies for detecting Alzheimer’s disease and related dementias (ADRD) using speech-based data from the DementiaBank corpus. Both **text-only** and **multimodal (audio–text)** models are considered.

---

## Abstract
Over half of U.S. adults with Alzheimer’s disease and related dementias (ADRD) remain undiagnosed. Speech-based screening algorithms offer a scalable approach, but the relative value of large language model (LLM) adaptation strategies is unclear.  

We compared LLM adaptation strategies for ADRD detection from the DementiaBank speech corpus using both text-only and multimodal models. Nine text-only LLMs (3B–405B; open-weight and commercial) and three multimodal audio–text models were evaluated with adaptations including:  

- **In-Context Learning (ICL)** with four demonstration selection policies (most-similar, least-similar, prototype, random)  
- **Reasoning-Augmented Prompting** (self-/teacher-generated rationales, self-consistency, Tree-of-Thought with domain experts)  
- **Parameter-Efficient Fine-Tuning** (token-level vs. added classification head)  
- **Multimodal Audio–Text Integration**  

Key findings:
- Prototype demonstrations achieved the best ICL performance (F1 up to 0.81).  
- Reasoning boosted smaller models (e.g., LLaMA-8B: F1 0.72 → 0.76).  
- Token-level fine-tuning consistently achieved the highest scores (F1 up to 0.83).  
- Adding a classification head improved models weak under token-level supervision.  
- Multimodal models performed competitively but did not surpass the best text-only systems.  

---

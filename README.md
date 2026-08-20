# 🎯 DPO for Tool/Function Calling

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-red?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![Transformers 4.36+](https://img.shields.io/badge/Transformers-4.36%2B-yellow?style=flat-square&logo=huggingface)](https://huggingface.co/transformers/)
[![LoRA PEFT](https://img.shields.io/badge/LoRA-PEFT-green?style=flat-square)](https://github.com/huggingface/peft)
[![DPO TRL](https://img.shields.io/badge/DPO-TRL-orange?style=flat-square)](https://github.com/huggingface/trl)
[![License MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Colab Compatible](https://img.shields.io/badge/Google%20Colab-Compatible-purple?style=flat-square&logo=googlecolab)](https://colab.research.google.com/)

> **Fine-tune language models to intelligently decide when to call external APIs vs generate text using Direct Preference Optimization (DPO).**

Direct Preference Optimization (DPO) has emerged as a powerful alternative to RLHF for aligning language models with human preferences. This project demonstrates how to apply DPO to teach models when and how to use external tools and function calls, achieving **94.1% schema accuracy** on Llama-2-7B with minimal parameter updates using LoRA.

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Motivation](#motivation)
4. [Key Features](#key-features)
5. [Workflow & Pipeline](#workflow--pipeline)
6. [Architecture](#architecture)
7. [Tech Stack](#tech-stack)
8. [Project Structure](#project-structure)
9. [Quick Start](#quick-start)
10. [Detailed Setup](#detailed-setup)
11. [Usage Examples](#usage-examples)
12. [Results & Benchmarks](#results--benchmarks)
13. [Evaluation Metrics](#evaluation-metrics)
14. [Docker & Deployment](#docker--deployment)
15. [CI/CD Pipeline](#cicd-pipeline)
16. [Testing](#testing)
17. [Limitations](#limitations)
18. [Future Improvements](#future-improvements)
19. [Contributing](#contributing)
20. [Citation](#citation)
21. [License](#license)

---

## Overview

This project implements a **hybrid SFT + DPO training pipeline** to fine-tune language models (specifically Llama-2-7B) for intelligent tool calling. The model learns to:

✅ **Recognize** when external tools/APIs are needed  
✅ **Generate** valid JSON schemas for function arguments  
✅ **Handle** errors gracefully without hallucinating parameters  
✅ **Decide** when to respond directly without tool invocation  

### Key Results

| Metric | Baseline | SFT Only | SFT + DPO | Improvement |
|--------|----------|----------|-----------|------------|
| **Schema Accuracy** | 81.2% | 88.5% | 94.1% | +12.9% |
| **Tool Precision** | 78.4% | 84.1% | 91.8% | +13.4% |
| **Tool Recall** | 85.0% | 89.2% | 93.5% | +8.5% |
| **Latency** | 0 ms | +12 ms | +14 ms | +14 ms |

---

## Problem Statement

### The Challenge

Most fine-tuned LLMs struggle with tool calling:

❌ **Hallucinating API parameters** that don't exist  
❌ **Invoking tools unnecessarily** when text suffices  
❌ **Invalid JSON schemas** that break downstream systems  
❌ **No graceful error handling** for edge cases  
❌ **Poor decision-making** about when to use tools vs. generate text  

**Real-world impact:** AI agents fail silently, systems crash on malformed JSON, user experience degrades.

### Why This Matters

Tool calling is fundamental for:
- **AI Agents**: Autonomous systems that interact with real-world APIs
- **Chatbots**: Delegating tasks to external services (weather, search, calculations)
- **Workflow Automation**: LLMs orchestrating multi-step processes
- **Integration Layers**: Connecting LLMs to enterprise systems

---

## Motivation

### Why DPO?

Traditional approaches rely on RLHF (Reinforcement Learning from Human Feedback), which is:
- ⚠️ **Computationally expensive** (requires reward model training)
- ⚠️ **Complex to implement** (multiple model forward passes)
- ⚠️ **Unstable** (reward hacking, training oscillations)

**DPO solves this** by:
✅ Training directly on preference pairs without a reward model  
✅ Simple binary cross-entropy loss over log probabilities  
✅ 2-3x faster training than RLHF  
✅ More stable convergence  

### Why LoRA?

Full fine-tuning requires:
- 💾 **70GB+ VRAM** to train 7B models
- ⏱️ **Days of training** even on H100 GPUs
- 📦 **Multiple model copies** for deployment

LoRA achieves **99.22% parameter reduction**:
- ✅ Train on **single T4 GPU** (16GB)
- ✅ **2-3 hours** for full training
- ✅ **~20MB adapter files** instead of 13GB models

---
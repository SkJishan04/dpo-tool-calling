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
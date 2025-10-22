# Theories of Aging Challenge

## Description

A system for collecting full texts of scientific articles and answers to 9 questions on them using artificial intelligence has been built. The goal is to classify sathy on these issues and create a dataset that will help develop a new or unified theory of aging or understand existing ones. There is a benchmark system for different models. As well as a blank for annotating texts.

Stack:
- Web-Client: Vite.js, React, TypeScript, WebSocket
- Backend: Python, FastAPI, PyTorch, Transformers, LLM, PubMed Central E-utilities API
- DB: Qdrant

## Requirements

- OS: Windows.
- Install Git LFS before clone repo: `git lfs install`
- Terminal: run PowerShell as an Administrator
- Use Python version: ^3.10
- Install Poetry: `pipx install poetry`
- Copy and rename `.env.exapmle` to `.env`.
- In the `.env` file, replace the
  - Hugging Face token example with **[your own (follow the link)](https://huggingface.co/settings/tokens)** .
  - Open Router API Key example with **[your own (follow the link)](https://openrouter.ai/settings/keys)**

## Quick start

Complete the Requirements 👆

```powershell
# PowerShell terminal
.\run.ps1
```

## Start benchmark of models (This is not necessary for the system to work)

There is a benchmark of models. It runs separately. Before launching, set the `enabled` property to `true` in the `models_config_v2.yaml` file for each model that you want to include in the benchmark. To run it, after running `.\run.ps1`, run the command `poetry run python model_benchmark_v2.py` from the folder `data_d_questions_and_criterias_classifier`.

More info in "[Full_Project_Description.md](Full_Project_Description.md)" file.
# Theories of Aging Challenge

## Description



Построена система сбора полных текстов научных статей и ответов на 9 вопросов по ним с помощью искусственного интеллекта. Цель — классифицировать сатьи по этим вопросам и создать датасет, что поможет выработать новую или единую теорию старения или разобраться в существующих. Есть система бенчмарка разных моделей. А также заготовка для аннотирования текстов.

Stack:
- Web-Client: Vite.js, React, TypeScript, WebSocket
- Backend: Python, FastAPI, PyTorch, Transformers, LLM, PubMed Central E-utilities API
- DB: Qdrant

## Requirements

- OS: Windows.
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
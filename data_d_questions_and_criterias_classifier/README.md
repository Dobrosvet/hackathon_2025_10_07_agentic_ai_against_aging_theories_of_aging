# Questions and Criterias Classifier Microservice

## Purpose
- Automates annotation of scientific papers against nine aging-related questions (Q1–Q9) and four evaluation criteria (C1–C4).
- Supports multiple inference approaches: NLI, sentence embeddings, cross-encoders, and autoregressive LLMs.
- Integrates with Qdrant for validation data and logging infrastructure under `data/logs`.

## Validation Scope
### Questions (Q1–Q9)
- Q1: Detects whether a paper proposes an aging biomarker.
- Q2: Captures statements about molecular mechanisms of aging.
- Q3: Identifies suggested longevity interventions.
- Q4: Flags claims that aging cannot be reversed.
- Q5: Looks for biomarkers explaining maximal lifespan gaps between species.
- Q6: Explains naked mole rat longevity.
- Q7: Explains avian longevity versus mammals.
- Q8: Explains why larger animals live longer.
- Q9: Explains calorie restriction longevity effects.

### Criteria (C1–C4)
- C1: Biomarkers explaining lifespan across species.
- C2: Biomarkers for mortality inside species.
- C3: Predicts testable longevity interventions.
- C4: Focuses on mechanistic (molecular) explanations.

## Configuration
- Main settings live in `config.yaml`, rewritten in UTF-8 without BOM.
- `classifier` section controls base model, GPU usage, quantization, and tqdm progress visibility.
- `huggingface` section specifies the environment variable used for authentication (`HF_TOKEN`) and models that demand it.
- Model definitions and benchmark parameters are stored in `models_config_v2.yaml`.

## Environment Management
- All runtime variables are loaded from `.env` in the repository root using `python-dotenv` and the PowerShell launcher.
- Example `.env`:
  ```
  HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
  QDRANT_URL=http://localhost:6333
  ```
- `run.ps1` automatically reads `.env` and exposes values to all microservices; direct Poetry runs inherit the same settings via the Python autoload hook.

## Hugging Face Token (mandatory for gated LLMs)
1. Create a personal access token with at least `read` scope: https://huggingface.co/settings/tokens.
2. Add `HF_TOKEN` to `.env` and (optionally) export it in the current PowerShell session:
   - Session only: `$Env:HF_TOKEN = 'hf_xxxxxxxxxxxxxxxxxxxxx'`
   - Persistent fallback: `setx HF_TOKEN "hf_xxxxxxxxxxxxxxxxxxxxx"`
3. Validate the token via Poetry: `poetry run huggingface-cli whoami`.
4. `run.ps1` aborts if `HF_TOKEN` is missing to protect gated models such as `google/medgemma-4b-it` and `meta-llama/Llama-3.2-3B-Instruct`.

## Dependencies
- Python dependencies are managed by Poetry (`pyproject.toml` / `poetry.lock`).
- New runtime packages: `transformers >= 4.45`, `accelerate`, `huggingface-hub`, `safetensors`, `bitsandbytes`.
- GPU: NVIDIA GTX 1070 (8 GB) requires CUDA drivers compatible with PyTorch 2.1+ and bitsandbytes 0.45.
- Client dashboard now uses Bun instead of npm.

## Launching Services
1. Run PowerShell as Administrator.
2. Ensure `HF_TOKEN` is set and `bun` is available (`Get-Command bun`).
3. Execute `.\run.ps1`. The script:
   - Stops previous processes and frees required ports.
   - Verifies `HF_TOKEN`.
   - Boots all microservices via Poetry (`poetry run python main.py`).
   - Starts the client with `bun run dev`.
4. Access points (default ports):
   - Questions API: `http://127.0.0.1:8005`
   - Client UI: `http://localhost:5173`
   - Qdrant REST: `http://localhost:6333`

## Testing
- Unit tests reside under `tests/` (added for LLM authentication and generation).
- Run the full suite inside the Poetry environment:
  ```
  poetry run pytest
  ```
- Benchmark smoke test: `poetry run python model_benchmark_v2.py` (requires populated Qdrant and valid HF token).

## Useful Commands
- Inspect validation papers in Qdrant: `curl http://localhost:6333/collections`.
- Manual classifier run on a sample: `poetry run python questions_classifier_v2.py`.
- Clear Poetry virtual environment cache if needed: `poetry env remove --all`.

Keep all scripts and configuration files in UTF-8 (no BOM) and avoid executing Python outside the Poetry context to maintain dependency reproducibility.

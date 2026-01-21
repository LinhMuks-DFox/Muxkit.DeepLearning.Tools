# Repository Guidelines

## Project Structure & Module Organization
- Source modules: `audio_tools/`, `dataset_tools/`, `datasets/`, `lossfunction/`, `model_tools/`, `plot_tools/`, `score_tools/`, `train_tools/`, `vision_tools/`, `utl/`.
- Tests: per-module folders like `*/unit_test/` (and `score_tools/unittest/`).
- Docs: `docs/` (multi-language READMEs).
- Packaging: `pyproject.toml` (Poetry), `poetry.lock`.

## Build, Test, and Development Commands
- Install (via Poetry):
  
  ```bash
  poetry install
  ```
- Spawn shell / run tools:
  
  ```bash
  poetry shell          # optional
  poetry run python -V  # run any command
  ```
- Run tests (pytest if available, else unittest):
  
  ```bash
  poetry run pytest -q              # preferred
  poetry run python -m unittest discover -s . -p "test_*.py"
  ```

## Coding Style & Naming Conventions
- Python 3.10+. Follow PEP 8; 4-space indentation.
- Naming: packages/modules `snake_case`; classes `CamelCase`; functions/vars `snake_case`; constants `UPPER_SNAKE_CASE`.
- Keep modules focused; avoid cross-package cycles. Public APIs live in each package’s `__init__.py` when appropriate.
- Type hints encouraged for new/edited code.

## Testing Guidelines
- Prefer `unittest`-style tests under each package’s `unit_test/` directory; name files `test_*.py`.
- Keep tests deterministic (set seeds, control RNG) and fast; use synthetic tensors where possible.
- Aim for coverage of critical paths: data transforms, metrics, loss functions, and utility helpers.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise summary, optional scope, e.g., `fix(audio_tools): handle mono tensors`.
- Reference issues in body (`Fixes #123`) and describe rationale + impact.
- PRs: include clear description, reproduction/verification steps, and before/after results (numbers or screenshots for plots).
- Ensure tests pass locally (`poetry run pytest`) before opening/merging.

## Security & Configuration Tips
- Pin dependencies via Poetry; avoid ad-hoc `pip install`.
- Be mindful of large audio assets—do not commit datasets; use paths/config instead.
- Check Torch/Torchaudio/Torchvision version compatibility when upgrading.

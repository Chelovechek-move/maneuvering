# maneuvering

Library for spacecraft maneuvering algorithms (Python).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
pytest
```

## Structure
- `src/maneuvering/` — package code
- `tests/` — unit tests (pytest)
- `.github/workflows/ci.yml` — GitHub Actions CI
- `.pre-commit-config.yaml` — pre-commit hooks (optional)

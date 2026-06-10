# Contributing

Thank you for helping improve this research project.

## Before Opening a Change

- Check existing issues and pull requests.
- Keep changes focused on one clear problem.
- Do not commit checkpoints, generated figures, private data, or unrelated files.
- Preserve the distinction between functional communication evidence and
  descriptive language analysis.

## Development Setup

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
```

Run the quality checks before submitting:

```bash
python -m ruff check .
python -m unittest discover -s tests -v
```

On Windows, the bundled quality script runs compilation, tests, and status:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/check_project.ps1
```

## Pull Requests

Explain:

1. the problem being solved;
2. the behavior changed;
3. how the change was tested;
4. whether it affects experiment comparability or checkpoint compatibility.

New training or analysis behavior should include focused tests and documentation.

## Research Integrity

Report all communication controls, not only the best result. Keep preliminary
results clearly labeled, use isolated run directories, and compare multiple
seeds before making strong claims.

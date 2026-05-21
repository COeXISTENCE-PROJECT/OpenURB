# Contributing to OpenURB

Thanks for contributing to OpenURB.

OpenURB is a benchmark repository for route-choice learning in mixed-autonomy traffic. Please keep changes focused on maintaining the benchmark and its reproducibility.

## Before you start

- Prefer OpenURB AEC-facing interfaces only.
- Keep scripts, configs, tests, metrics, and documentation consistent with each other.
- Keep documentation factual and concise.

## Making changes

- Keep changes small and easy to review.
- If you change a script or config contract, update the related docs and tests in the same pass.
- If you edit the README source in `docs/README.template.md`, regenerate `README.md` with:

```bash
venv/bin/python tools/build_readme.py
```

## Testing

Run the test suite with:

```bash
venv/bin/python -m pytest tests/
```

Some tests require SUMO and may skip automatically if SUMO is not installed or unavailable.

## Pull requests

When opening a pull request, please:

- explain the problem and the intended change,
- list any scripts, configs, metrics, or docs affected,
- mention the tests you ran,
- keep the PR focused on one topic when possible.

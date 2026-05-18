# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`seametrics` is SEA.AI's library for evaluating AI pipelines. It is a **metrics library**: results from this code feed model selection and production decisions, so correctness matters more than throughput of changes. Treat every metric as a load-bearing API.

## Commands

Dependencies and environment are managed via `uv` (see `pyproject.toml` + optional extras: `fiftyone`, `panoptic`, `test`).

```bash
uv sync --all-extras              # install dev + all optional deps
uv run pytest                     # full suite (coverage configured in pyproject.toml)
uv run pytest tests/detection -k pr_rec_f1   # single test / pattern
uv run pytest --cov-report=xml    # what CI runs (Sonar uploads this)
uv run ruff check . && uv run ruff format --check .
uv run pylint seametrics          # CI runs pylint on changed files only
pre-commit run --all-files        # ruff, gitleaks, toml/json formatting, etc.
```

CI (`.github/workflows/ci.yaml`) runs ruff (lint + format), pylint, pytest+coverage, and SonarQube on PRs to `develop`. Lint/pylint run only on changed files; tests + Sonar run on the full repo.

## Architecture

Top-level package `seametrics/` is split by metric family. Each subpackage is independently importable and exposes its metric classes at the package root:

- `detection/` — object detection metrics. `PrecisionRecallF1Support` is the public entry; the package selects a `tm/` (torchmetrics-backed) or `np/` (pure-numpy) implementation at import time via `detection/imports.py::_TORCHMETRICS_AVAILABLE`. `cocoeval.py` is a modified `pycocotools` core; touch it with extreme care — numerical changes ripple through every downstream report.
- `tracking/` — MOT metrics. `TrackingMetrics` wraps `motmetrics` (MOTA/MOTP/IDF1…); `HOTAMetrics` implements HOTA. Both share an interface so `compute_all_metrics_by_sequence` can evaluate them in one pass. Failed sequences are logged into `metric.failed_sequences` rather than raised — preserve that contract.
- `panoptic/` — panoptic-quality metrics, also has a `tm/` variant guarded by an `imports.py` flag.
- `payload/` — `processor.py` defines the `Sequence` / `Resolution` data carriers used across metrics. `Sequence` allows arbitrary dynamic attributes; do not lock it down without checking all call sites.
- `horizon/`, `customMAP/`, `annotations/`, `fo_utils/`, `user_friendly/` — domain-specific helpers and FiftyOne integrations.

Optional heavy dependencies (`torch`, `torchmetrics`, `fiftyone`, `transformers`, `cleanlab`) are gated behind extras and import-availability flags. **Never import them unconditionally** from a module that lives outside its dedicated `tm/` / fiftyone subtree — it breaks the base install.

## Working in this repo

- **Tests must be real.** No mocks that stub the metric under test; no `assert result is not None` placeholders. A new metric or branch needs at least one test with hand-computed expected values (small, inspectable inputs) so a reviewer can verify the math by eye. Before changing a metric's math, find the existing test that pins its numeric output and update it intentionally — never "just to make CI green".
- **Respect the np/tm split.** When adding behavior to a metric, mirror it in both backends or document why only one is supported. Asymmetric backends are a long-term maintenance trap.
- **Optional-deps discipline.** Guard new imports of `torch`/`fiftyone`/etc. behind the existing `_*_AVAILABLE` flags or a `try/except ImportError` at the package boundary, and add the dep to the right extra in `pyproject.toml`.
- **Failed-sequence semantics.** In `tracking/`, per-sequence errors are routed to `metric.failed_sequences` via `log_failed_sequence` rather than raised — this is a tested contract (`tests/tracking/test_tracking_metrics.py`). Preserve it when extending.

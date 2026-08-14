"""
Benchmark multiple jsonpickle versions and plot per-file performance deltas.

Usage: python3 benchmarking/benchmark_versions.py 3.0.0,3.1.0,3.2.0

Requires matplotlib in the current environment and internet access that can reach PyPi
to install jsonpickle into the per-version venvs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from subprocess import run
from venv import EnvBuilder

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = ROOT / "benchmarking" / "jsonpickle_pytest_benchmarks.py"
IMAGES_DIR = ROOT / "images"
DATA_DIR = IMAGES_DIR / "benchmark-data"
WORKTREE_DIR = ROOT / ".bench-worktrees"
MIN_PYTEST_CONFIG = ROOT / "benchmarking" / ".bench-pytest.ini"


def _sanitize_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _venv_dir_for(version: str) -> Path:
    safe = _sanitize_tag(version)
    return ROOT / ".bench-venvs" / f"jsonpickle-{safe}"


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _ensure_venv(version: str) -> Path:
    venv_dir = _venv_dir_for(version)
    if not venv_dir.exists():
        EnvBuilder(with_pip=True).create(venv_dir)
    python = _venv_python(venv_dir)
    run([str(python), "-m", "pip", "install", "-U", "pip"], check=True)
    run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "pytest",
            "pytest-benchmark[histogram]",
            "pygal",
            "pygaljs",
            "matplotlib",
            "numpy",
            "scikit-learn",
            "pandas",
            "pymongo",
            "ecdsa",
        ],
        check=True,
    )
    return python


def _git_tags() -> set[str]:
    result = run(
        ["git", "-C", str(ROOT), "tag", "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _resolve_tag(version: str, tags: set[str]) -> str:
    if version in tags:
        return version
    candidate = f"v{version}"
    if candidate in tags:
        return candidate
    raise RuntimeError(f"No git tag found for version {version}")


def _ensure_worktree(tag: str) -> Path:
    WORKTREE_DIR.mkdir(exist_ok=True)
    worktree = WORKTREE_DIR / f"jsonpickle-{_sanitize_tag(tag)}"
    if worktree.exists():
        current = run(
            ["git", "-C", str(worktree), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        target = run(
            ["git", "-C", str(ROOT), "rev-parse", tag],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if current == target:
            return worktree
        run(
            ["git", "-C", str(ROOT), "worktree", "remove", "--force", str(worktree)],
            check=True,
        )
    run(
        ["git", "-C", str(ROOT), "worktree", "add", "--force", str(worktree), tag],
        check=True,
    )
    return worktree


def _ensure_min_pytest_config() -> Path:
    if MIN_PYTEST_CONFIG.exists():
        return MIN_PYTEST_CONFIG
    MIN_PYTEST_CONFIG.write_text(
        "[pytest]\naddopts=\npython_functions = test_* simple_* complex_* state_*\n",
        encoding="utf-8",
    )
    return MIN_PYTEST_CONFIG


def _run_benchmarks(version: str, python: Path, worktree: Path) -> None:
    env = os.environ.copy()
    env["JSONPICKLE_BENCH_TAG"] = version
    env["JSONPICKLE_EXPECT_VERSION"] = version
    env["JSONPICKLE_EXPECT_PATH"] = str((worktree / "jsonpickle").resolve())
    env["JSONPICKLE_BENCH_OUTPUT_ROOT"] = str(ROOT)
    env["JSONPICKLE_TESTS_DIR"] = str(worktree / "tests")
    # avoid old pytest.ini options that require missing plugins
    env["JSONPICKLE_PYTEST_CONFIG"] = str(_ensure_min_pytest_config())
    env["JSONPICKLE_ROOTDIR"] = str(worktree / "tests")
    env["PYTHONPATH"] = os.pathsep.join([str(worktree), str(worktree / "tests")])
    env["PYTHONNOUSERSITE"] = "1"
    work_dir = str(ROOT / "benchmarking")
    run([str(python), str(BENCH_SCRIPT)], check=False, cwd=work_dir, env=env)


def _run_benchmarks_local() -> None:
    env = os.environ.copy()
    env["JSONPICKLE_BENCH_TAG"] = "local"
    env["JSONPICKLE_EXPECT_VERSION"] = "local"
    env["JSONPICKLE_EXPECT_PATH"] = str((ROOT / "jsonpickle"))
    env["JSONPICKLE_BENCH_OUTPUT_ROOT"] = str(ROOT)
    env["JSONPICKLE_TESTS_DIR"] = str(ROOT / "tests")
    env["JSONPICKLE_PYTEST_CONFIG"] = str(ROOT / "pytest.ini")
    env["JSONPICKLE_ROOTDIR"] = tempfile.mkdtemp(prefix="jsonpickle-bench-root-")
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    run(
        [sys.executable, str(BENCH_SCRIPT)],
        check=False,
        cwd=str(ROOT / "benchmarking"),
        env=env,
    )


def _load_benchmarks_for(version: str) -> dict[str, dict[str, float]]:
    tag = _sanitize_tag(version)
    per_file: dict[str, dict[str, float]] = {}
    if not DATA_DIR.exists():
        return {}
    for path in DATA_DIR.rglob("*.json"):
        if tag not in path.name:
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        for bench in payload.get("benchmarks", []):
            group = bench.get("group") or "unknown"
            name = bench.get("name") or bench.get("fullname") or "unknown"
            mean = bench.get("stats", {}).get("mean")
            if mean is None:
                continue
            per_file.setdefault(group, {})[name] = mean
    return per_file


def _compute_file_means(
    baseline: str,
    versions: list[str],
    series: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    baseline_files = series.get(baseline, {})
    for file_name, baseline_tests in baseline_files.items():
        baseline_set = set(baseline_tests.keys())
        if not baseline_set:
            continue
        for version in versions:
            tests = series.get(version, {}).get(file_name, {})
            if not tests:
                continue
            common = baseline_set.intersection(tests.keys())
            if not common:
                continue
            values = [tests[name] for name in common]
            result.setdefault(version, {})[file_name] = sum(values) / len(values)
    return result


def _plot_deltas(
    baseline: str, versions: list[str], series: dict[str, dict[str, float]]
) -> Path:
    files = sorted(series.get(baseline, {}).keys())
    if not files:
        raise RuntimeError("No benchmark data found to plot")

    baseline_stats = series.get(baseline, {})
    labels = files
    x = range(len(labels))
    width = 0.8 / max(len(versions), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.6), 5))
    offset = 0
    for version in versions:
        values = []
        for file_name in labels:
            base = baseline_stats.get(file_name)
            cur = series.get(version, {}).get(file_name)
            if base is None or cur is None:
                values.append(float("nan"))
            else:
                values.append(cur / base)
        positions = [i + offset for i in x]
        ax.bar(positions, values, width=width, label=version)
        offset += width

    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks([i + width * (len(versions) / 2) for i in x])
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Time relative to baseline (baseline = 1.0)")
    ax.set_title(f"jsonpickle benchmark time ratios vs {baseline}")
    ax.legend()
    fig.tight_layout()

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H%M%S%z")
    output = IMAGES_DIR / f"benchmark-compare-{timestamp}.png"
    fig.savefig(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "versions",
        help="Comma-delimited list of jsonpickle versions (e.g. 3.0.0,3.1.0)",
    )
    parser.add_argument(
        "--output",
        help="Output path for the comparison plot (PNG)",
    )
    args = parser.parse_args()

    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    if not versions:
        print("Please provide at least one version!", file=sys.stderr)
        return

    (ROOT / ".bench-venvs").mkdir(exist_ok=True)
    IMAGES_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    tags = _git_tags()
    _run_benchmarks_local()
    for version in versions:
        tag = _resolve_tag(version, tags)
        worktree = _ensure_worktree(tag)
        python = _ensure_venv(version)
        _run_benchmarks(version, python, worktree)

    versions_all = ["local", *versions]
    raw = {version: _load_benchmarks_for(version) for version in versions_all}
    series = _compute_file_means("local", versions_all, raw)
    output = _plot_deltas("local", versions_all, series)

    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(output.read_bytes())
        print(f"Saved plot to {target}")
    else:
        print(f"Saved plot to {output}")
    return


if __name__ == "__main__":
    raise SystemExit(main())

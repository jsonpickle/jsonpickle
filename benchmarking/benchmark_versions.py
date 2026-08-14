"""
Benchmark multiple jsonpickle versions and plot per-file performance deltas.

Usage: python3 benchmarking/benchmark_versions.py 3.0.0,3.1.0,3.2.0

Requires matplotlib in the current environment and internet access that can reach PyPi
to install jsonpickle into the per-version venvs.
"""

from __future__ import annotations

import argparse
import json
import math
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


def _env_for(version: str, worktree: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env["JSONPICKLE_BENCH_TAG"] = version
    env["JSONPICKLE_EXPECT_VERSION"] = version
    env["JSONPICKLE_BENCH_OUTPUT_ROOT"] = str(ROOT)
    env["PYTHONNOUSERSITE"] = "1"
    if worktree is None:
        env["JSONPICKLE_EXPECT_PATH"] = str((ROOT / "jsonpickle"))
        env["JSONPICKLE_TESTS_DIR"] = str(ROOT / "tests")
        env["JSONPICKLE_PYTEST_CONFIG"] = str(ROOT / "pytest.ini")
        env["JSONPICKLE_ROOTDIR"] = tempfile.mkdtemp(prefix="jsonpickle-bench-root-")
        env["PYTHONPATH"] = str(ROOT)
    else:
        env["JSONPICKLE_EXPECT_PATH"] = str((worktree / "jsonpickle").resolve())
        env["JSONPICKLE_TESTS_DIR"] = str(worktree / "tests")
        # avoid old pytest.ini options that require missing plugins
        env["JSONPICKLE_PYTEST_CONFIG"] = str(_ensure_min_pytest_config())
        env["JSONPICKLE_ROOTDIR"] = str(worktree / "tests")
        env["PYTHONPATH"] = os.pathsep.join([str(worktree), str(worktree / "tests")])
    return env


def _collect_tests(version: str, python: Path, worktree: Path | None) -> set[str]:
    """
    Return the test keys this version's suite contains, without timing them.
    This is useful for ensuring that all tested versions test the same tests.
    """
    interpreter = str(python) if worktree is not None else sys.executable
    env = _env_for(version, worktree)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        target = Path(handle.name)
    env["JSONPICKLE_BENCH_COLLECT"] = str(target)
    result = run(
        [interpreter, str(BENCH_SCRIPT)],
        check=False,
        cwd=str(ROOT / "benchmarking"),
        env=env,
    )
    if result.returncode != 0:
        print(f"Warning: collection for {version} exited with {result.returncode}")
    try:
        keys = set(json.loads(target.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        print(f"Warning: no collected tests for {version}")
        keys = set()
    target.unlink(missing_ok=True)
    return keys


def _run_benchmarks(
    version: str, python: Path | None, worktree: Path | None, only: Path
) -> None:
    interpreter = str(python) if worktree is not None else sys.executable
    env = _env_for(version, worktree)
    env["JSONPICKLE_BENCH_ONLY"] = str(only)
    run(
        [interpreter, str(BENCH_SCRIPT)],
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


def _geometric_mean(values: list[float]) -> float:
    """
    Benchmark times are on the order of 1e-5 seconds, so multiplying a few
    hundred of them together would flush to zero long before we took the root.
    Therefore, we compute it in log-space.
    """
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _compute_file_means(
    versions: list[str],
    series: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    file_names: set[str] = set()
    for version in versions:
        file_names.update(series.get(version, {}))

    result: dict[str, dict[str, float]] = {}
    for file_name in sorted(file_names):
        per_version = [series.get(v, {}).get(file_name, {}) for v in versions]
        missing = [v for v, tests in zip(versions, per_version) if not tests]
        if missing:
            print(f"Skipping {file_name}: no data for {', '.join(missing)}")
            continue
        common = set(per_version[0])
        for tests in per_version[1:]:
            common &= set(tests)
        # log() needs strictly positive input, and a zero-length timing is junk
        # data anyway, so drop those tests rather than crashing on them
        common = {
            name for name in common if all(tests[name] > 0 for tests in per_version)
        }
        if not common:
            print(f"Skipping {file_name}: no tests shared by every version")
            continue
        dropped = max(len(tests) for tests in per_version) - len(common)
        if dropped:
            print(f"{file_name}: comparing {len(common)} tests, ignoring {dropped}")
        for version, tests in zip(versions, per_version):
            result.setdefault(version, {})[file_name] = _geometric_mean(
                [tests[name] for name in common]
            )
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
    ax.set_ylabel("Geomean time relative to baseline (baseline = 1.0)")
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
    targets: list[tuple[str, Path | None, Path | None]] = [("local", None, None)]
    for version in versions:
        tag = _resolve_tag(version, tags)
        worktree = _ensure_worktree(tag)
        python = _ensure_venv(version)
        targets.append((version, python, worktree))

    # benchmarking a test that only some versions have is wasted time,
    # and averaging it in would skew the result
    collected = {
        version: _collect_tests(version, python, worktree)
        for version, python, worktree in targets
    }
    common = set.intersection(*collected.values()) if collected else set()
    if not common:
        print("No tests are shared by every version under test!", file=sys.stderr)
        return
    for version, keys in collected.items():
        if len(keys) > len(common):
            print(f"{version}: benchmarking {len(common)} of {len(keys)} tests")

    only = ROOT / "benchmarking" / ".bench-common-tests.json"
    only.write_text(json.dumps(sorted(common)), encoding="utf-8")

    for version, python, worktree in targets:
        _run_benchmarks(version, python, worktree, only)

    versions_all = ["local", *versions]
    raw = {version: _load_benchmarks_for(version) for version in versions_all}
    series = _compute_file_means(versions_all, raw)
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

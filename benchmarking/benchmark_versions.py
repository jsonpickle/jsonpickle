"""
Benchmark multiple jsonpickle versions and plot how they compare.

Every version gets one box and whisker plot over its per-file timings, and the
run ends with a single comparison plot of the whole suite's geometric mean for
each version.

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

# the benchmark runner lives next to this file and owns the shared plotting code
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jsonpickle_pytest_benchmarks import (
    geometric_mean,
    plot_file_summary,
    print_file_summary,
)

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
        env["JSONPICKLE_EXPECT_PATH"] = str(ROOT / "jsonpickle")
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


def _shared_tests(
    versions: list[str],
    series: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, list[float]]]:
    file_names: set[str] = set()
    for version in versions:
        file_names.update(series.get(version, {}))

    result: dict[str, dict[str, list[float]]] = {version: {} for version in versions}
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
        ordered = sorted(common)
        for version, tests in zip(versions, per_version):
            result[version][file_name] = [tests[name] for name in ordered]
    return result


def _plot_version_summaries(
    shared: dict[str, dict[str, list[float]]], timestamp: str
) -> list[Path]:
    outputs = []
    for version, per_file in shared.items():
        if not per_file:
            print(f"Skipping the summary plot for {version}: no shared test data")
            continue
        print_file_summary(per_file, header=f"{version} benchmark summary")
        output = plot_file_summary(
            per_file,
            f"jsonpickle {version} test suite benchmark summary",
            IMAGES_DIR / f"benchmark-summary-{timestamp}-{_sanitize_tag(version)}.png",
        )
        if output is not None:
            outputs.append(output)
            print(f"Saved the {version} summary plot to {output}")
    return outputs


def _suite_geomeans(shared: dict[str, dict[str, list[float]]]) -> dict[str, float]:
    """
    Collapse every shared test in the suite down to one number per version.
    """
    result = {}
    for version, per_file in shared.items():
        values = [mean for means in per_file.values() for mean in means]
        if values:
            result[version] = geometric_mean(values)
    return result


def _plot_comparison(
    baseline: str,
    versions: list[str],
    suite: dict[str, float],
    test_count: int,
    timestamp: str,
) -> Path:
    labels = [version for version in versions if version in suite]
    if not labels:
        raise RuntimeError("No benchmark data found to plot")

    values = [suite[version] * 1e6 for version in labels]
    base = suite.get(baseline)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.4), 5))
    bars = ax.bar(range(len(labels)), values, width=0.6, color="#3b6fb0")
    for version, value, bar in zip(labels, values, bars):
        text = f"{value:.3f}us"
        if base:
            text += f"\n{suite[version] / base:.3f}x"
        ax.annotate(
            text,
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
            color="#3c4450",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Geometric mean time per test (microseconds)")
    # two lines because one long title runs off the end of a narrow figure
    subtitle = f"over {test_count} tests shared by every version"
    if base:
        subtitle += f" (x = ratio to {baseline})"
    ax.set_title(f"jsonpickle suite-wide benchmark geomean\n{subtitle}")
    ax.grid(axis="y", color="#d7dbe0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.margins(y=0.15)
    fig.tight_layout()

    output = IMAGES_DIR / f"benchmark-compare-{timestamp}.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
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
    shared = _shared_tests(versions_all, raw)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H%M%S%z")
    _plot_version_summaries(shared, timestamp)

    suite = _suite_geomeans(shared)
    test_count = sum(len(means) for means in shared[versions_all[0]].values())
    print("\n===== suite-wide geomean =====")
    for version in versions_all:
        if version in suite:
            print(f"{version}: {suite[version] * 1e6:.3f}us over {test_count} tests")
    output = _plot_comparison("local", versions_all, suite, test_count, timestamp)

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

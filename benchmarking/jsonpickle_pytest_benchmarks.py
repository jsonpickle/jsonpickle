"""
Run the Pytest test suite through pytest-benchmark.

Files are benchmarked one at a time by default, but you can set
JSONPICKLE_BENCH_JOBS to run several files concurrently.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from subprocess import run
from threading import Lock

import pytest

# internal flags we hand to the per-file workers
_FILE_FLAG = "--jsonpickle-bench-file"
_TIMESTAMP_FLAG = "--jsonpickle-bench-timestamp"


def _item_key(item):
    file_path = str(getattr(item, "path", "")) or item.nodeid.split("::", 1)[0]
    return f"{Path(file_path).name}::{item.name}"


class _CollectKeysPlugin:
    def __init__(self):
        self.keys = []

    def pytest_collection_modifyitems(self, session, config, items):
        self.keys.extend(_item_key(item) for item in items)


class _BenchmarkAllTestsPlugin:
    def __init__(self, allowed=None):
        # for benchmark_versions we sometimes need to restrict the tests that run
        self.allowed = allowed

    def pytest_collection_modifyitems(self, session, config, items):
        if self.allowed is None:
            return
        items[:] = [item for item in items if _item_key(item) in self.allowed]

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        if not isinstance(item, pytest.Function):
            yield
            return
        if "benchmark" in item.fixturenames:
            # this is for the case of the legacy jsonpickle_benchmarks.py file
            yield
            return

        file_path = str(getattr(item, "path", "")) or item.nodeid.split("::", 1)[0]
        file_label = Path(file_path).name

        original_runtest = item.runtest

        def benchmarked_runtest():
            benchmark = item._request.getfixturevalue("benchmark")
            if hasattr(benchmark, "group"):
                benchmark.group = file_label
            benchmark(original_runtest)

        item.runtest = benchmarked_runtest
        try:
            yield
        finally:
            item.runtest = original_runtest


def _sanitize_tag(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _tag_suffix():
    tag = os.environ.get("JSONPICKLE_BENCH_TAG")
    return f"-{_sanitize_tag(tag)}" if tag else ""


def _images_dir():
    output_root = os.environ.get("JSONPICKLE_BENCH_OUTPUT_ROOT", ".")
    return Path(output_root) / "images"


def _data_dir():
    return _images_dir() / "benchmark-data"


def _default_benchmark_args(timestamp, file_label, extra_args):
    existing = {arg.split("=", 1)[0] for arg in extra_args}
    defaults = []
    tag_suffix = _tag_suffix()
    data_dir = _data_dir()
    if file_label != "jsonpickle_test.py" and "--benchmark-disable-gc" not in existing:
        defaults.append("--benchmark-disable-gc")
    if "--benchmark-warmup" not in existing:
        defaults.append("--benchmark-warmup=on")
    if "--benchmark-warmup-iterations" not in existing:
        defaults.append("--benchmark-warmup-iterations=1")
    if "--benchmark-precision" not in existing:
        # ensure that it benchmarks until the estimated mean is within 2% of the true mean
        defaults.append("--benchmark-precision=0.02")
    if "--benchmark-confidence" not in existing:
        # make the 2% thing be with at least 98% confidence
        defaults.append("--benchmark-confidence=0.98")
    if "--benchmark-min-rounds" not in existing:
        # no more than 5 seconds regardless of the precision/confidence we want
        defaults.append("--benchmark-max-time=5")
    # no per-file histogram, the run ends with one summary plot over every file
    if "--benchmark-storage" not in existing:
        data_dir.mkdir(parents=True, exist_ok=True)
        defaults.append(f"--benchmark-storage={data_dir}")
    if "--benchmark-save" not in existing:
        defaults.append(f"--benchmark-save={timestamp}-{file_label}{tag_suffix}")
    return defaults


def _taskset_pin_if_available():
    # match logic from the make benchmark command
    if os.environ.get("JSONPICKLE_TASKSET", "") == "1":
        return
    if sys.platform != "linux":
        return
    taskset = "taskset"
    if not _which(taskset):
        return
    cpu_count = os.cpu_count() or 1
    if cpu_count > 16:
        target_core = 7
    else:
        target_core = max(cpu_count // 2 - 1, 0)
    os.environ["JSONPICKLE_TASKSET"] = "1"
    script = str(Path(__file__).resolve())
    os.execvp(
        taskset,
        [taskset, "-c", str(target_core), sys.executable, script, *sys.argv[1:]],
    )


def _parse_cpu_list(raw):
    # sysfs writes these as "0,8" on some kernels and "0-1" on others
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values


def _physical_cores():
    """
    In order to avoid pinning CPUs to SMT siblings, we only return
    one logical CPU per physical core
    """
    seen = set()
    cores = []
    cpu_root = Path("/sys/devices/system/cpu")
    try:
        cpu_dirs = sorted(cpu_root.glob("cpu[0-9]*"), key=lambda p: int(p.name[3:]))
    except OSError:
        cpu_dirs = []
    for cpu_dir in cpu_dirs:
        try:
            raw = (cpu_dir / "topology" / "thread_siblings_list").read_text(
                encoding="utf-8"
            )
        except OSError:
            continue
        siblings = frozenset(_parse_cpu_list(raw.strip()))
        if siblings in seen:
            continue
        seen.add(siblings)
        cores.append(int(cpu_dir.name[3:]))
    if not cores:
        # we can't find any sysfs topology so we're probably not on linux
        # therefore we can just assume SMT and take every other logical CPU
        cores = list(range(0, os.cpu_count() or 1, 2))
    return cores


def _resolve_jobs(requested):
    # never use more than half of the machine's cores to avoid SMT issues
    limit = max((os.cpu_count() or 2) // 2, 1)
    try:
        jobs = int(requested)
    except (TypeError, ValueError):
        print(f"Ignoring non-numeric job count {requested!r}", file=sys.stderr)
        return 1
    if jobs < 1:
        return 1
    if jobs > limit:
        print(
            f"You requested {jobs} jobs but more than {limit} will produce "
            f"unreliable measurements so this will use {limit}!",
            file=sys.stderr,
        )
        return limit
    return jobs


def _benchmark_cores(jobs):
    """
    Cores 0 and 1 handle the kernel's interrupt load and the last core
    handles a bunch of other kernel tasks, so we start from the middle of the
    list and walk outward.
    """
    cores = _physical_cores()
    if not cores:
        return []
    start = min(max(len(cores) // 2 - 1, 0), len(cores) - 1)
    ordered = cores[start:] + cores[:start]
    return ordered[:jobs]


def _which(executable):
    for path in os.environ.get("PATH", "").split(os.pathsep):
        candidate = Path(path) / executable
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _verify_expected_version():
    expected = os.environ.get("JSONPICKLE_EXPECT_VERSION")
    if not expected:
        return True
    try:
        import jsonpickle  # type: ignore
    except Exception as exc:  # ruff: ignore[BLE001]
        print(f"Failed to import jsonpickle: {exc}", file=sys.stderr)
        return False

    actual = getattr(jsonpickle, "__version__", "unknown")
    module_path = Path(getattr(jsonpickle, "__file__", "")).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    repo_pkg = (repo_root / "jsonpickle").resolve()
    expect_path = os.environ.get("JSONPICKLE_EXPECT_PATH")
    if expect_path:
        expected_pkg = Path(expect_path).resolve()
        if expected_pkg not in module_path.parents:
            print(
                f"Expected jsonpickle under {expected_pkg}, got {module_path}",
                file=sys.stderr,
            )
            return False
        # if the expected path matches, allow version mismatches for older tags
        return True

    if expected == "local":
        if repo_pkg not in module_path.parents:
            print(
                f"Expected local jsonpickle under {repo_pkg}, got {module_path}",
                file=sys.stderr,
            )
            return False
        return True

    if actual != expected:
        print(
            f"Expected jsonpickle {expected}, got {actual} ({module_path})",
            file=sys.stderr,
        )
        return False
    if repo_pkg in module_path.parents:
        print(
            f"Expected site-packages jsonpickle for {expected}, got local path {module_path}",
            file=sys.stderr,
        )
        return False
    return True


def _split_args(argv):
    single_file = None
    timestamp = None
    extra_args = []
    for arg in argv:
        name, _, value = arg.partition("=")
        if name == _FILE_FLAG:
            single_file = value
        elif name == _TIMESTAMP_FLAG:
            timestamp = value
        else:
            extra_args.append(arg)
    return single_file, timestamp, extra_args


def _pytest_config_args():
    args = []
    pytest_config = os.environ.get("JSONPICKLE_PYTEST_CONFIG")
    pytest_rootdir = os.environ.get("JSONPICKLE_ROOTDIR")
    if pytest_config:
        args.extend(["-c", pytest_config])
    if pytest_rootdir:
        args.extend(["--rootdir", pytest_rootdir])
    return args


def _load_allowed():
    only = os.environ.get("JSONPICKLE_BENCH_ONLY")
    if not only:
        return None
    with open(only, encoding="utf-8") as handle:
        return set(json.load(handle))


def _collect_keys(test_files):
    plugin = _CollectKeysPlugin()
    for test_file in test_files:
        args = _pytest_config_args()
        args.append(str(test_file))
        args.extend(["--collect-only", "-q", "-p", "no:cacheprovider"])
        pytest.main(args, plugins=[plugin])
    return sorted(set(plugin.keys))


_warmed_up = False


def _warm_up_cpu():
    """
    Spin before the first measurement so that the core is already at its working clock.

    On my (Theelx's) laptop, a core that has been idle starts around 40% slower and
    takes a few hundred milliseconds to ramp up, which is long enough to inflate the
    first test files of a run.
    """
    global _warmed_up
    if _warmed_up:
        return
    _warmed_up = True
    try:
        seconds = float(os.environ.get("JSONPICKLE_BENCH_WARMUP_SECONDS", "1.0"))
    except ValueError:
        seconds = 1.0
    if seconds <= 0:
        return
    deadline = time.perf_counter() + seconds
    total = 0
    while time.perf_counter() < deadline:
        for i in range(20000):
            total += i * i
    return total


def _run_one_file(test_file, timestamp, extra_args):
    _warm_up_cpu()
    file_label = test_file.name
    args = _pytest_config_args()
    args.append(str(test_file))
    args.extend(_default_benchmark_args(timestamp, file_label, extra_args))
    if file_label == "jsonpickle_test.py":
        extra_args = [arg for arg in extra_args if arg != "--benchmark-disable-gc"]
    args.extend(extra_args)
    return pytest.main(args, plugins=[_BenchmarkAllTestsPlugin(_load_allowed())])


def _run_serial(test_files, timestamp, extra_args):
    # report the last failure we saw, so a bad file can fail the whole run
    exit_code = 0
    for test_file in test_files:
        code = _run_one_file(test_file, timestamp, extra_args)
        if code != 0:
            exit_code = code
    return exit_code


def _run_parallel(test_files, timestamp, extra_args, cores):
    script = str(Path(__file__).resolve())
    available = queue.Queue()
    for core in cores:
        available.put(core)
    print_lock = Lock()

    # we run the longest file first so that a slow one isn't the last one left
    ordered = sorted(test_files, key=lambda p: p.stat().st_size, reverse=True)

    def run_one(test_file):
        core = available.get()
        try:
            command = []
            if _which("taskset"):
                command.extend(["taskset", "-c", str(core)])
            command.extend(
                [
                    sys.executable,
                    script,
                    f"{_FILE_FLAG}={test_file}",
                    f"{_TIMESTAMP_FLAG}={timestamp}",
                    *extra_args,
                ]
            )
            env = os.environ.copy()
            env["JSONPICKLE_TASKSET"] = "1"
            result = run(command, env=env, capture_output=True, text=True, check=False)
        finally:
            available.put(core)
        with print_lock:
            print(f"===== {test_file.name} (core {core}) =====")
            sys.stdout.write(result.stdout)
            sys.stderr.write(result.stderr)
            sys.stdout.flush()
        return result.returncode

    exit_code = 0
    with ThreadPoolExecutor(max_workers=len(cores)) as pool:
        for code in pool.map(run_one, ordered):
            if code != 0:
                exit_code = code
    return exit_code


def geometric_mean(values):
    """
    Benchmark times are on the order of 1e-5 seconds, so multiplying a few
    hundred of them together would flush to zero long before we took the root.
    Therefore, we compute it in log-space.
    """
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _load_run_means(timestamp):
    """
    Return ({file_label: [mean seconds per test]}, unconverged) for the saved runs of
    this invocation. pytest-benchmark writes one JSON file per --benchmark-save, under a
    machine-id subdirectory of the storage dir.

    ``unconverged`` lists the tests whose mean never reached --benchmark-precision, since
    those are the points on the plot that the next run is least likely to reproduce.
    """
    data_dir = _data_dir()
    if not data_dir.exists():
        return {}, []
    unconverged = []
    per_file = {}
    for path in sorted(data_dir.rglob("*.json")):
        # every save name for this run is "<timestamp>-<file label><tag suffix>"
        if timestamp not in path.name:
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            print(f"Ignoring unreadable benchmark data at {path}", file=sys.stderr)
            continue
        # strip the "0001_" counter pytest-benchmark prepends, then the run tags
        fallback = path.stem.split("_", 1)[-1].split(f"{timestamp}-", 1)[-1]
        for bench in payload.get("benchmarks", []):
            label = bench.get("group") or fallback
            mean = bench.get("stats", {}).get("mean")
            # log() needs strictly positive input, and a zero-length timing is
            # junk data anyway, so drop those tests rather than crashing on them
            if mean is None or mean <= 0:
                continue
            per_file.setdefault(label, []).append(mean)
            precision = bench.get("precision")
            if precision and not precision.get("converged", True):
                achieved = precision.get("achieved")
                achieved = "unknown" if achieved is None else f"{achieved:.2%}"
                unconverged.append(
                    f"{label}::{bench.get('name')} (+-{achieved}, "
                    f"wanted +-{precision.get('target', 0):.2%})"
                )
    return per_file, unconverged


def plot_file_summary(per_file, title, output):
    """
    Draw a single box and whisker plot for a whole run: one box per test file
    over that file's per-test mean times (in seconds), marked with the
    geometric mean of those means. Returns the written path, or None if
    matplotlib is unavailable.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed, skipping the summary plot",
            file=sys.stderr,
        )
        return None

    # alphabetical, so the same file sits in the same spot across runs
    labels = sorted(per_file)
    # microseconds keep the tick labels readable for the fastest tests
    data = [[mean * 1e6 for mean in per_file[name]] for name in labels]
    geomeans = [geometric_mean(values) for values in data]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.7), 6))
    ink = "#3b6fb0"
    ax.boxplot(
        data,
        showfliers=True,
        widths=0.55,
        medianprops={"color": ink, "linewidth": 2},
        boxprops={"color": "#5f6b7a", "linewidth": 1},
        whiskerprops={"color": "#5f6b7a", "linewidth": 1},
        capprops={"color": "#5f6b7a", "linewidth": 1},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "none",
            "markeredgecolor": "#9aa4b1",
        },
    )
    ax.plot(
        range(1, len(labels) + 1),
        geomeans,
        linestyle="none",
        marker="D",
        markersize=8,
        markerfacecolor=ink,
        markeredgecolor="white",
        label="geometric mean of per-test means",
    )
    # set the ticks by hand because boxplot renamed its label argument in 3.9
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(
        [f"{name}\n({len(values)} tests)" for name, values in zip(labels, data)]
    )
    ax.set_yscale("log")
    ax.set_ylabel("Mean time per test (microseconds, log scale)")
    # the legend sits above the axes so that it can't cover a tall box
    ax.set_title(title, pad=26)
    ax.grid(axis="y", color="#d7dbe0", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output


def print_file_summary(per_file, header="benchmark summary"):
    print(f"\n===== {header} (geometric mean of per-test means) =====")
    for name in sorted(per_file):
        values = per_file[name]
        print(f"{name}: {geometric_mean(values) * 1e6:.3f}us over {len(values)} tests")


def _report_summary(timestamp):
    per_file, unconverged = _load_run_means(timestamp)
    if not per_file:
        print("No benchmark data was saved, skipping the summary plot")
        return
    print_file_summary(per_file)
    if unconverged:
        print(
            f"\n{len(unconverged)} benchmark(s) never reached --benchmark-precision, so "
            f"the means will move between runs."
        )
        for entry in unconverged[:10]:
            print(f"   {entry}")
        if len(unconverged) > 10:
            print(f"   ... and {len(unconverged) - 10} more")
    output = plot_file_summary(
        per_file,
        f"jsonpickle test suite benchmark summary ({timestamp})",
        _images_dir() / f"benchmark-summary-{timestamp}{_tag_suffix()}.png",
    )
    if output is not None:
        print(f"Saved summary plot to {output}")


def main():
    single_file, timestamp, extra_args = _split_args(sys.argv[1:])

    if single_file:
        if not _verify_expected_version():
            return 2
        return _run_one_file(Path(single_file), timestamp, extra_args)

    # collect-only mode, so benchmark_versions.py can work out which tests every
    # version has in common before any of them are actually benchmarked
    collect_target = os.environ.get("JSONPICKLE_BENCH_COLLECT")
    if collect_target:
        if not _verify_expected_version():
            return 2
        tests_dir = Path(os.environ.get("JSONPICKLE_TESTS_DIR", "tests")).resolve()
        keys = _collect_keys(sorted(tests_dir.rglob("*.py")))
        Path(collect_target).write_text(json.dumps(keys), encoding="utf-8")
        return 0

    jobs = _resolve_jobs(int(os.environ.get("JSONPICKLE_BENCH_JOBS", "1")))
    cores = _benchmark_cores(jobs) if jobs > 1 else []
    if jobs > 1 and len(cores) < jobs:
        print(
            f"Only found {len(cores)} usable physical core(s); using {len(cores)} job(s)",
            file=sys.stderr,
        )
        jobs = len(cores)
    if jobs <= 1:
        _taskset_pin_if_available()
    if not _verify_expected_version():
        return 2

    tests_dir = Path(os.environ.get("JSONPICKLE_TESTS_DIR", "tests")).resolve()
    test_files = sorted(tests_dir.rglob("*.py"))
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H%M%S%z")
    if jobs <= 1:
        exit_code = _run_serial(test_files, timestamp, extra_args)
    else:
        print(f"Benchmarking {len(test_files)} files across cores {cores}")
        exit_code = _run_parallel(test_files, timestamp, extra_args, cores)

    # benchmark_versions.py plots its own cross-version comparison, so a
    # per-version summary plot there would just be noise
    if not os.environ.get("JSONPICKLE_BENCH_ONLY"):
        _report_summary(timestamp)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

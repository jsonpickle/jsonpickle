## Purpose
This directory exists to allow benchmarking and detection of performance regressions with jsonpickle.
This is important because there are a number of different uses of jsonpickle with huge datasets, which can take minutes to load, so we don't want to bump that up to hours accidentally.

## Running the code
### Pre-reqs
**You MUST be on Linux for any of these suggestions to work. If you are running these on Windows or Mac, the results will not be valid.**

The automatic portion of the "reproducible setup" that jsonpickle strives for currently consists of pinning the task to a single CPU to ensure that the kernel doesn't switch the task between CPUs. 
The core switching can cause huge outliers between runs of a single benchmark, on the order of tens of thousands of times slower.
If you want to run the benchmarks yourself, some things that you'll want to do to ensure even more reproducible results include:
- Turning Turbo Boost off for Intel CPUs
- Using something like ``sudo cpupower -c<core #> frequency-set -g ondemand -f <base freq>GHz`` to set a specific frequency for the pinned CPU on AMD CPUs (similar effect to turning off Turbo Boost).
- Turning off a core's SMT sibling with ``echo 0 > /sys/devices/system/cpu/cpu<core #>/online``, to make it less likely for the kernel to schedule tasks on the logical core.
- Setting NOHZ_FULL kernel parameter for the core that you run benchmarks on, to prevent kernel timer interrupts on that core (for AMD CPUs: ``GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_pstate=passive nohz_full=<core #>"``).
- All the other amazing suggestions in https://pyperf.readthedocs.io/en/latest/system.html (note: isolcpus won't work on modern Linux systems with cgroups v2).

jsonpickle's script will assume that the core you choose to run this on is the total cores of your machine (including SMT) divided by 2, minus 1. In code, this would be ``c_run = (len(cores) / 2) - 1``.
This core tends to be one of the lesser-used ones on your system, assuming you have 8 or more total cores (including SMT, so a CPU advertised as quad-core would count).

### Actually running the code
You can just run ``make benchmark`` in the root of the project, and it will automatically do as much as it can to ensure a reproducible setup, and easy-to-visualize results! It should output a histogram of the results in the images/ directory.
Another thing you can do is ``cd benchmarking && ./analyze_benchmarks.sh``. This will compare the mean speed for each benchmark between the current branch that you're working on, and the main branch, so you can check improvements or regressions.

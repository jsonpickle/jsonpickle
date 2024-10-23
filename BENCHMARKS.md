**TL;DR:**

- The jsonpickle 3.x series has some minor performance boosts, but everything is more niche
- The jsonpickle 2.x series shows significant improvement in encoding/decoding speeds for virtually every test, the main benefits come in class encoding/decoding, but there are more minor speedups for other common use cases also.

Onto the raw data!

Environment:
- Tested on CPython 3.9.13, 3.10.9, and 3.11. Pictures for 3.10 are displayed here, since that's the most common version as of October 2024, but other versions can be found in the ``images/`` directory.
- 16-core AMD Ryzen 7950X3D CPU, 128GB DDR5-4800 RAM
- Linux Mint 22 (Based off Ubuntu 24.04), Linux kernel 6.8.0
- All tests were run on physical core 7/16 (NUMA Node 0), with SMT disabled using the ondemand governor, NOHZ_FULL enabled for that core in grub settings, and the core's clock speed fixed to the turbo speed of 5.7GHz.
- Used this command to run:
```bash
taskset -c 7 pytest --benchmark-disable-gc --benchmark-warmup=on --benchmark-warmup-iterations=1 --benchmark-min-rounds=10000 --benchmark-histogram=./images/benchmark-$(date +%Y-%m-%dT%T%z) ./jsonpickle_benchmarks.py
```

jsonpickle 3.3.0-3.10 (drops support for python <3.7, minor unpickling perf boosts)

<figure><img src="images/jsonpickle-3.3.0-3.10.svg"></figure>

jsonpickle 2.2.0-3.10 (last release of 2.0 series)

<figure><img src="images/jsonpickle-2.2.0-3.10.svg"></figure>

jsonpickle 1.5.2-3.10 (major performance increases reverted due to accidental breaking change)

<figure><img src="images/jsonpickle-1.5.2-3.10.svg"></figure>

jsonpickle 1.5.1-3.10 (major performance increases in this version)

<figure><img src="images/jsonpickle-1.5.1-3.10.svg"></figure>

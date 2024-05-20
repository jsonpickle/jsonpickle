***WARNING: CERTAIN BENCHMARKS WILL SHOW VERY HIGH OUTLIER VALUES, TAKE ALL BENCHMARKS WITH A GRAIN OF SALT.***

**TL;DR**

The jsonpickle 2.x series shows significant improvement in encoding/decoding speeds for virtually every test, the main benefits come in class encoding/decoding, but there are more minor speedups for other common use cases also. Onto the raw data!

Environment:
- Tested on CPython 3.10.9. 3.11 and above did not work with my setup.
- 8-core Intel i7-11800H CPU (laptop)
- Ubuntu 22.04, Linux kernel 6.5.0
- Turbo boost disabled, plugged in + on battery

jsonpickle 3.1.0-3.10 (drops support for python <3.7)

<figure><img src="images/jsonpickle-3.1.0-3.10.svg"></figure>

jsonpickle 2.2.0-3.10 (last release of 2.0 series)

<figure><img src="images/jsonpickle-2.2.0-3.10.svg"></figure>

jsonpickle 1.5.2-3.10 (major performance increases reverted due to accidental breaking change)

<figure><img src="images/jsonpickle-1.5.2-3.10.svg"></figure>

jsonpickle 1.5.1-3.10 (major performance increases in this version)

<figure><img src="images/jsonpickle-1.5.1-3.10.svg"></figure>

jsonpickle 0.9.6-3.9 (please upgrade from this!)

<figure><img src="images/jsonpickle-0.9.6-3.9.svg"></figure>

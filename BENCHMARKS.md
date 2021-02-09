***WARNING: CERTAIN BENCHMARKS WILL SHOW VERY HIGH OUTLIER VALUES, TAKE ALL BENCHMARKS WITH A GRAIN OF SALT.***

**TL;DR**

jsonpickle 2.0.0 shows significant improvement in encoding/decoding speeds for virtually every test, the main benefits come in class encoding/decoding, but there are more minor speedups for other common use cases also. Onto the raw data!

jsonpickle 2.0.0 and Python 3.9.1, on a 3 vCPU (3.8GHz Ryzen 3900X) VPS running Ubuntu 20.04

<figure><img src="images/jsonpickle-2.0.0.svg"><figcaption>Jsonpickle 2.0.0 Graph</figcaption></figure>

jsonpickle 1.5.2 and Python 3.9.1, on a 3 vCPU (3.8GHz Ryzen 3900X) VPS running Ubuntu 20.04

<figure><img src="images/jsonpickle-1.5.2.svg"><figcaption>Jsonpickle 1.5.2 Graph</figcaption></figure>

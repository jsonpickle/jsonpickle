import csv

# makes math far easier
import numpy as np


def filter_results_by_name(results, substr):
    perfs = []
    for name, perf in results.items():
        if substr in name:
            perfs.append(perf)
    return np.array(perfs)


def main():
    # using dicts seems easier at first but we'd have to eventually convert
    # to a list for easy elementwise division or use an ugly loop
    funcs = []
    main = []
    new = []

    with open("perf.csv", "r") as f:
        csv_reader = csv.reader(f)
        # skip header row
        next(csv_reader)

        # csv is sorted by name so we can assume even rows are main and odd are new
        for indx, (func, median) in enumerate(csv_reader):
            func = func.split("::")[-1]
            # indx is even
            if not (indx % 2):
                funcs.append(func)
                main.append(float(median))
            else:
                new.append(float(median))

    main = np.array(main)
    new = np.array(new)

    relative_perfs = new / main
    # more stats will be included when needed
    print(
        f"The new branch is {round(relative_perfs.mean(), 4)}x faster than the main branch aross all tests!"
    )

    results = {}

    for func_name, relative_perf in zip(funcs, relative_perfs):
        # this allows us to categorize classes of results, such as encode vs decode
        results[func_name] = relative_perf

    encode_perfs = filter_results_by_name(results, "encode")
    decode_perfs = filter_results_by_name(results, "decode")
    print(
        f"The new branch is {round(encode_perfs.mean(), 4)}x faster than the main branch aross encode tests!"
    )
    print(
        f"The new branch is {round(decode_perfs.mean(), 4)}x faster than the main branch aross decode tests!"
    )


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import jsonpickle.ext.pandas as jsonpickle_pandas

jsonpickle_numpy.register_handlers()
jsonpickle_pandas.register_handlers()


class Report:
    def __init__(self, report_name: str, features: np.ndarray, summary: pd.DataFrame):
        self.report_name = report_name
        self.features = features
        self.summary = summary


def main():
    features = np.array(
        [
            ["Apple", 2.0],
            ["Orange", 2.0],
            ["Banana", 2.0],
        ]
    )
    summary = pd.DataFrame(features, columns=["Product", "Sale Price"])

    report = Report(report_name="Business Report", features=features, summary=summary)

    # indent smaller than 4 for small screens
    encoded = jsonpickle.encode(report, indent=2)
    print(f"Encoded:\n{encoded}\n")

    decoded = jsonpickle.decode(encoded)
    print(f"Round-trip:\n{decoded.model_name}")
    print(decoded.features.shape)
    print(decoded.summary.head(2))


if __name__ == "__main__":
    main()

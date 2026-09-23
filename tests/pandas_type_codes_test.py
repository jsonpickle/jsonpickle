"""
Regression tests for the dtype codes the pandas DataFrame handler writes
"""

import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

import jsonpickle
import jsonpickle.ext.pandas
from jsonpickle import tags_pd

# this is used to that we can spawn multiple processes
# in order to more reliably reproduce the issues
REPO = Path(jsonpickle.__file__).resolve().parents[1]

NUMERIC_COLUMNS = {
    "i8": np.array([-100, 100], dtype="int8"),
    "i16": np.array([-30000, 30000], dtype="int16"),
    "i32": np.array([-2_000_000_000, 2_000_000_000], dtype="int32"),
    "i64": np.array([-(2**62), 2**62], dtype="int64"),
    "u8": np.array([0, 250], dtype="uint8"),
    "u16": np.array([0, 65000], dtype="uint16"),
    "u32": np.array([0, 4_000_000_000], dtype="uint32"),
    "u64": np.array([0, 2**63 + 5], dtype="uint64"),
    "f32": np.array([0.5, 1.25], dtype="float32"),
    "f64": np.array([0.1, 1e300], dtype="float64"),
}

LEGACY_BARE_CODES = {"pd/f", "pd/i", "pd/u"}


@pytest.fixture(scope="module", autouse=True)
def pandas_extension():
    jsonpickle.ext.pandas.register_handlers()
    yield
    jsonpickle.ext.pandas.unregister_handlers()


def _run_in_fresh_process(code):
    """
    Run code in a new interpreter and return its stdout
    """
    env = dict(os.environ, PYTHONPATH=str(REPO))
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return result.stdout


def _frame_codes(encoded):
    meta = json.loads(encoded)["meta"]
    return [code for code, _ in meta["dtypes_rle"]]


def test_subclasses_order_is_deterministic():
    """
    The code assignment is order dependent, so the input order must be fixed
    """
    found = tags_pd.all_subclasses(pd.api.extensions.ExtensionDtype)
    expected = sorted(
        found,
        key=lambda c: (
            c.__module__.partition(".")[0] != "pandas",
            c.__module__,
            c.__qualname__,
        ),
    )
    assert found == expected


def test_type_codes_are_identical_everywhere():
    dump = (
        "import json; from jsonpickle import tags_pd; "
        "print(json.dumps({str(k): v for k, v in tags_pd.TYPE_MAP.items()}, "
        "sort_keys=True))"
    )
    here = json.dumps({str(k): v for k, v in tags_pd.TYPE_MAP.items()}, sort_keys=True)
    assert _run_in_fresh_process(dump).strip() == here


def test_frame_encoded_in_other_process_preserves_dtypes():
    columns = {name: arr.tolist() for name, arr in NUMERIC_COLUMNS.items()}
    dtypes = {name: str(arr.dtype) for name, arr in NUMERIC_COLUMNS.items()}
    encode = (
        "import json, numpy as np, pandas as pd, jsonpickle, jsonpickle.ext.pandas\n"
        "jsonpickle.ext.pandas.register_handlers()\n"
        f"columns, dtypes = {columns!r}, {dtypes!r}\n"
        "frame = pd.DataFrame({k: np.array(v, dtype=dtypes[k]) "
        "for k, v in columns.items()})\n"
        "print(jsonpickle.encode(frame))\n"
    )
    restored = jsonpickle.decode(_run_in_fresh_process(encode))
    for name, expected in NUMERIC_COLUMNS.items():
        assert restored[name].dtype == expected.dtype, name
        assert restored[name].tolist() == expected.tolist(), name


def test_numeric_columns_roundtrip():
    frame = pd.DataFrame(NUMERIC_COLUMNS)
    restored = jsonpickle.decode(jsonpickle.encode(frame))
    pd.testing.assert_frame_equal(restored, frame, check_exact=True)


def test_encoder_never_writes_ambiguous_codes():
    """
    Only width-suffixed codes are unambiguous across jsonpickle versions
    """
    codes = _frame_codes(jsonpickle.encode(pd.DataFrame(NUMERIC_COLUMNS)))
    assert not LEGACY_BARE_CODES & set(codes)
    assert codes == [
        "pd/i8",
        "pd/i16",
        "pd/i32",
        "pd/i64",
        "pd/u8",
        "pd/u16",
        "pd/u32",
        "pd/u64",
        "pd/f32",
        "pd/f64",
    ]


def _legacy_frame(code, values):
    """
    Produce a buggy frame document just as old version of jsonpickle used to.
    """
    return json.dumps(
        {
            "py/object": "pandas.DataFrame",
            "values": json.dumps({"c": values}),
            "txt": True,
            "meta": {
                "dtypes_rle": [[code, 1]],
                "index": json.dumps(list(range(len(values)))),
                "index_names": None,
                "columns": json.dumps(["c"]),
                "column_names": None,
                "is_multiindex": False,
                "is_multicolumns": False,
            },
        }
    )


@pytest.mark.parametrize(
    "code, values, expected_dtype",
    [
        # each of these used to be read as the narrow type in some processes,
        # which rounds 0.1 and wraps the integers
        ("pd/f", [0.1, 1e300], "float64"),
        ("pd/i", [1000, -2_000_000_000, 2**40], "int64"),
        ("pd/u", [300, 4_000_000_000, 2**63 + 5], "uint64"),
    ],
)
def test_ambiguous_codes_decode_to_the_widest_dtype(code, values, expected_dtype):
    restored = jsonpickle.decode(_legacy_frame(code, values))
    assert str(restored["c"].dtype) == expected_dtype
    assert restored["c"].tolist() == values


@pytest.mark.parametrize(
    "code, dtype",
    [
        ("pd/f32", "float32"),
        ("pd/f64", "float64"),
        ("pd/i8", "int8"),
        ("pd/i16", "int16"),
        ("pd/i32", "int32"),
        ("pd/i64", "int64"),
        ("pd/u8", "uint8"),
        ("pd/u16", "uint16"),
        ("pd/u32", "uint32"),
        ("pd/u64", "uint64"),
    ],
)
def test_width_suffixed_codes_keep_meaning(code, dtype):
    """
    These have always meant the same thing and weren't
    affected by the bug
    """
    restored = jsonpickle.decode(_legacy_frame(code, [1, 2]))
    assert str(restored["c"].dtype) == dtype
    assert restored["c"].tolist() == [1, 2]


def test_pdinterval_on_integer_preserves_values():
    """
    Pandas 2 sometimes wrote "pd/in" for int8/int16 columns
    """
    with warnings.catch_warnings():
        # the column can't be cast to interval, which is reported
        warnings.simplefilter("ignore")
        restored = jsonpickle.decode(_legacy_frame("pd/in", [-100, 30000]))
    assert restored["c"].tolist() == [-100, 30000]
    assert np.issubdtype(restored["c"].dtype, np.integer)


def test_third_party_dtypes_cant_steal_pandas_codes():
    """
    Ensure that an extension dtype defined before import
    can't steal a pandas prefix
    """
    dump = (
        "import json\n"
        "from pandas.api.extensions import ExtensionDtype\n"
        "class Rival(ExtensionDtype):\n"
        "    name = 'boolean_rival'\n"
        "    type = bool\n"
        "Rival.__module__ = 'aaa_third_party'\n"
        "from jsonpickle import tags_pd\n"
        "print(json.dumps({str(k): v for k, v in tags_pd.TYPE_MAP.items()}))\n"
    )
    with_rival = json.loads(_run_in_fresh_process(dump))
    for dtype, code in tags_pd.TYPE_MAP.items():
        assert with_rival[str(dtype)] == code, dtype

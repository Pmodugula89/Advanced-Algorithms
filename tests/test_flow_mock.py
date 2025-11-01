import importlib
import types
from pathlib import Path


def make_fake_df(write_ok=True):
    class FakeSeries(list):
        def isnull(self):
            return [v is None for v in self]

        def sum(self):
            return sum(1 for v in self if v is None)

        def median(self):
            vals = [v for v in self if v is not None]
            return sum(vals) / len(vals) if vals else 0

        def mode(self):
            vals = [v for v in self if v is not None]
            return [vals[0]] if vals else [None]

        def fillna(self, val, inplace=False):
            for i, v in enumerate(self):
                if v is None:
                    self[i] = val

    class FakeDF(dict):
        def __init__(self, data):
            super().__init__(data)

        def info(self):
            return "FakeDF info"

        def describe(self):
            return {k: 'desc' for k in self.keys()}

        @property
        def columns(self):
            return list(self.keys())

        def __getitem__(self, key):
            val = super().__getitem__(key)
            return val

        def select_dtypes(self, include=None):
            # naive: return self where values are numeric lists
            class Cols:
                def __init__(self, cols):
                    self._cols = cols

                @property
                def columns(self):
                    return self._cols

            numeric = []
            for k, v in self.items():
                # treat list of numbers as numeric
                if all(isinstance(x, (int, float)) or x is None for x in v):
                    numeric.append(k)
            return Cols(numeric)

        def to_csv(self, path, index=False):
            if not write_ok:
                raise IOError("disk full")
            Path(path).write_text("col1,col2\n1,2\n")

        def head(self):
            return {k: (v[:1] if isinstance(v, list) else v) for k, v in self.items()}

    return FakeDF({
        "total_purchase": FakeSeries([100, 200, None]),
        "num_transactions": FakeSeries([1, 2, 3]),
        "last_transaction_date": ["2025-10-01", "2025-10-15", "2025-09-01"],
    })


def test_main_flow_writes_output(tmp_path, monkeypatch):
    Main = importlib.import_module("src.Main")

    fake = make_fake_df(write_ok=True)

    # Monkeypatch pipeline steps to return our fake df and avoid heavy libs
    monkeypatch.setattr(Main, "load_data", lambda fp: fake)
    monkeypatch.setattr(Main, "handle_missing_values", lambda d: d)
    monkeypatch.setattr(Main, "detect_outliers", lambda d: d)
    monkeypatch.setattr(Main, "feature_engineering", lambda d: d)
    monkeypatch.setattr(Main, "scale_features", lambda d: d)

    out = tmp_path / "out.csv"
    Main.main(["--data-path", "ignored.csv", "--no-plot", "--output", str(out)])

    assert out.exists()


def test_main_flow_handles_write_error(tmp_path, monkeypatch, capsys):
    Main = importlib.import_module("src.Main")
    fake = make_fake_df(write_ok=False)

    monkeypatch.setattr(Main, "load_data", lambda fp: fake)
    monkeypatch.setattr(Main, "handle_missing_values", lambda d: d)
    monkeypatch.setattr(Main, "detect_outliers", lambda d: d)
    monkeypatch.setattr(Main, "feature_engineering", lambda d: d)
    monkeypatch.setattr(Main, "scale_features", lambda d: d)

    out = tmp_path / "out.csv"
    Main.main(["--data-path", "ignored.csv", "--no-plot", "--output", str(out)])

    captured = capsys.readouterr()
    assert "Failed to write output" in captured.out

"""AlgorithmPredictor.load() caching behavior"""

import os
import time

import joblib

from src.model.predict import AlgorithmPredictor


class _FakeModel:
    """Stand-in for a trained classifier: always predicts `label`"""

    def __init__(self, label: str) -> None:
        self.label = label

    def predict(self, rows):
        return [self.label] * len(rows)


def test_load_returns_cached_model_for_unchanged_file(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(_FakeModel("zstd"), model_path)

    first = AlgorithmPredictor.load(model_path)
    second = AlgorithmPredictor.load(model_path)

    assert first._model is second._model


def test_load_picks_up_a_retrained_model_written_to_the_same_path(tmp_path):
    model_path = tmp_path / "model.joblib"
    joblib.dump(_FakeModel("zstd"), model_path)

    predictor = AlgorithmPredictor.load(model_path)
    assert predictor._model.label == "zstd"

    # bump mtime explicitly in case filesystem's resolution is too coarse to notice fast rewrite
    joblib.dump(_FakeModel("brotli"), model_path)
    future = time.time() + 5
    os.utime(model_path, (future, future))

    reloaded = AlgorithmPredictor.load(model_path)
    assert reloaded._model.label == "brotli"

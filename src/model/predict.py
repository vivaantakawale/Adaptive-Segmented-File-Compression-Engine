"""Load a trained model and predict best compression algorithm for a chunk"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.features.extract import FEATURE_NAMES, extract_vector

DEFAULT_MODEL_PATH = Path("models/algo_selector.joblib")


# mtime_ns/size are part of cache key so a retrained model overwritten at same path invalidates cache instead of serving stale model
@lru_cache(maxsize=None)
def _load_model(resolved_path: str, mtime_ns: int, size: int) -> Any:
    return joblib.load(resolved_path)


class AlgorithmPredictor:
    """Wraps a trained sklearn compatible classifier for per-chunk prediction"""

    def __init__(self, model: Any) -> None:
        self._model = model

    @classmethod
    def load(cls, model_path: Path = DEFAULT_MODEL_PATH) -> "AlgorithmPredictor":
        """Load a trained model from disk

        Args:
            model_path: Path to a joblib serialized sklearn compatible model

        Returns:
            An AlgorithmPredictor wrapping loaded model
        """
        resolved = Path(model_path).resolve()
        stat = resolved.stat()
        return cls(_load_model(str(resolved), stat.st_mtime_ns, stat.st_size))

    def predict_vector(self, vector: np.ndarray) -> str:
        """Predict best algorithm from precomputed feature vector

        Args:
            vector: Feature vector, ordered by `FEATURE_NAMES` (see `src.features.extract.extract_vector`)

        Returns:
            Predicted algorithm name
        """
        row = pd.DataFrame([vector], columns=FEATURE_NAMES)
        return self._model.predict(row)[0]

    def predict_chunk(self, data: bytes) -> str:
        """Predict best algorithm for a single chunk

        Args:
            data: Chunk bytes

        Returns:
            Predicted algorithm name
        """
        return self.predict_vector(extract_vector(data))

    def predict_chunks(self, chunks: list[bytes]) -> list[str]:
        """Predict best algorithm for each chunk in a batch

        Args:
            chunks: Chunk byte strings

        Returns:
            Predicted algorithm names
        """
        if not chunks:
            return []
        matrix = pd.DataFrame([extract_vector(c) for c in chunks], columns=FEATURE_NAMES)
        return list(self._model.predict(matrix))


__all__ = ["AlgorithmPredictor", "FEATURE_NAMES", "DEFAULT_MODEL_PATH"]

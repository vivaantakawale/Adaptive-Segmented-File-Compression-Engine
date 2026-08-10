"""Brute force labeling: run every registered compression algorithm on a chunk, 
record each one's compressed size and wall-clock time, and report the winner
"""

import time
from dataclasses import dataclass, field

from src.compressors import registry


@dataclass
class AlgorithmResult:
    algorithm: str
    compressed_size: int
    seconds: float


@dataclass
class LabelResult:
    best_algorithm: str
    best_size: int
    results: dict[str, AlgorithmResult] = field(default_factory=dict)

    @property
    def sizes(self) -> dict[str, int]:
        return {name: r.compressed_size for name, r in self.results.items()}

    @property
    def times(self) -> dict[str, float]:
        return {name: r.seconds for name, r in self.results.items()}


def label_chunk(data: bytes, algorithms: list[str] | None = None) -> LabelResult:
    """Compress `data` with every candidate algorithm and find the smallest.

    Args:
        data: Chunk bytes to label
        algorithms: Algorithm names to try, Defaults to `registry.list_algorithms()`

    Returns:
        LabelResult with smallest size winner
        full per-algorithm size/time table
    """
    names = algorithms if algorithms is not None else registry.list_algorithms()

    results: dict[str, AlgorithmResult] = {}
    for name in names:
        start = time.perf_counter()
        compressed = registry.compress(data, name)
        elapsed = time.perf_counter() - start
        results[name] = AlgorithmResult(
            algorithm=name, compressed_size=len(compressed), seconds=elapsed
        )

    best_name = min(results, key=lambda n: results[n].compressed_size)
    return LabelResult(
        best_algorithm=best_name,
        best_size=results[best_name].compressed_size,
        results=results,
    )


def label_chunks(chunks: list[bytes], algorithms: list[str] | None = None) -> list[LabelResult]:
    """Label a sequence of chunks

    Args:
        chunks: Chunk byte strings to label
        algorithms: Algorithm names to try (See `label_chunk`)

    Returns:
        One LabelResult per chunk in same order as `chunks`
    """
    return [label_chunk(c, algorithms) for c in chunks]

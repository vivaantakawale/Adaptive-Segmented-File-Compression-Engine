from src.compressors import registry
from src.labeling.brute_force_label import label_chunk, label_chunks

TEXT_SAMPLE = b"the quick brown fox jumps over the lazy dog. " * 200


def test_label_chunk_covers_every_algorithm():
    result = label_chunk(TEXT_SAMPLE)
    assert set(result.results.keys()) == set(registry.list_algorithms())


def test_label_chunk_picks_true_smallest():
    result = label_chunk(TEXT_SAMPLE)
    assert result.best_size == min(result.sizes.values())
    assert result.sizes[result.best_algorithm] == result.best_size


def test_label_chunk_records_nonnegative_times():
    result = label_chunk(TEXT_SAMPLE)
    assert all(t >= 0 for t in result.times.values())


def test_label_chunk_store_never_beaten_by_worse_than_identity():
    result = label_chunk(TEXT_SAMPLE)
    assert result.sizes["store"] == len(TEXT_SAMPLE)
    assert result.best_size <= len(TEXT_SAMPLE)


def test_label_chunk_respects_algorithm_subset():
    result = label_chunk(TEXT_SAMPLE, algorithms=["store", "gzip"])
    assert set(result.results.keys()) == {"store", "gzip"}


def test_label_chunks_labels_each_chunk_independently():
    chunks = [TEXT_SAMPLE, b"\x00" * 1000, b"random-ish-binary-\x01\x02\x03" * 50]
    results = label_chunks(chunks)
    assert len(results) == len(chunks)
    for chunk, result in zip(chunks, results):
        assert result.sizes["store"] == len(chunk)

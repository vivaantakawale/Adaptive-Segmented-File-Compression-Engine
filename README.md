# Automated Adaptive Segmented Compression Engine

Adaptive, segmented file compression. Splits the input into chunks and
uses a trained classifier to predict which compression algorithm wins on
each one.

Real files are rarely homogeneous. A log file might have a text header, a
binary payload, and a base64 blob, and each of those compresses best under
a different algorithm. Picking one algorithm for the whole file is fast
but leaves ratio on the table wherever the file isn't uniform. Brute-force
(try every algorithm on every chunk, keep the smallest) always finds the
best answer but costs `O(chunks × algorithms)` compression passes, most of
which get thrown away.

smart-zip trains a classifier on cheap chunk features (entropy, byte
histogram, printable ratio, bigram frequencies - none of which require
actually compressing anything) to predict the winner directly, then
compresses once per chunk with that prediction. The archive always records
which algorithm actually got used per chunk, so if a prediction is wrong
you just lose some ratio - the file still decompresses fine.

Benchmark, 6-file / 769-chunk held-out set (Gutenberg text, generated
CSV/JSON, fresh random bytes, a system binary):

| strategy            | ratio  | total time |
|---------------------|--------|------------|
| always_store        | 1.000x | 0.000s     |
| always_gzip         | 2.049x | 0.052s     |
| always_zstd         | 2.094x | 0.051s     |
| brute_force_ceiling | 2.229x | 1.578s     | 
| model_predicted     | 2.186x | 0.795s     |

98.1% of brute force's ratio in about half the time. Beats always-zstd
outright.

## Requirements

Python 3.13. Everything else is in `requirements.txt`: `zstandard` and
`brotli` for the compression backends, `numpy`/`scipy`/`scikit-learn`/
`pandas`/`pyarrow`/`joblib` for the ML side, `streamlit` for the GUI,
`pytest` for tests.

## Install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`models/algo_selector.joblib` is already trained and checked in. No
training step needed to compress or decompress - only to rebuild the
model from a different corpus.

## Usage

### CLI

```bash
PYTHONPATH=. python scripts/compress.py input.file output.file.vtzip
PYTHONPATH=. python scripts/decompress.py output.file.vtzip restored.file
```

`compress.py` chunks the input (4KB fixed-size blocks by default, matching
what the shipped model trained on), predicts an algorithm per chunk,
compresses, and writes the archive. Prints a summary when it's done -
original/compressed size, ratio, per-algorithm chunk breakdown. It also
writes a manifest next to the archive (`output.file.vtzip.manifest.json`);
pass `--no-keep-manifest` if you don't want that kept around.

```
--chunker {fixed_size,content_aware}   default: fixed_size
--chunk-size N                          default: 4096
--model-path PATH                       default: models/algo_selector.joblib
--batch-size N                          default: 256 (chunks predicted per batch)
```

`decompress.py` checks per-chunk and whole-file SHA-256 before it writes
anything out, and raises on any mismatch.

### GUI

```bash
streamlit run app.py
```

`http://localhost:8501`, three tabs:

- **Compress** - upload a file, get the archive back plus the same
  summary the CLI prints.
- **Decompress** - upload an archive this tool made, get the original
  file back, integrity checked.
- **Analysis** - runs `model_predicted` against the single-algorithm
  baselines and the brute-force ceiling on a file you pick, so you can see
  how close the model actually gets. It doesn't produce an archive you'd
  keep - use the Compress tab for that.

Local only. Nothing here is deployed, there's no auth, don't put it on a
shared host.

### Retraining

```bash
# 1. build a labeled dataset from data/raw/ - this brute-forces every
#    algorithm on every chunk to get ground-truth labels, so it's slow
PYTHONPATH=. python scripts/build_dataset.py --corpus data/raw --out data/labeled/dataset.parquet --chunk-size 4096

# 2. train
PYTHONPATH=. python src/model/train.py --dataset data/labeled/dataset.parquet --out models/algo_selector.joblib --class-weight balanced

# 3. check it against the baselines and the brute-force ceiling
PYTHONPATH=. python src/benchmark/compare.py --files-dir data/raw_holdout --model models/algo_selector.joblib
```

`build_dataset.py` shards its output per source file under
`data/labeled/.shards/`, so if it dies partway through a big corpus, you
only lose the one file that was in flight - rerunning skips everything
that already has a shard.

## Architecture

```
raw file
   │
   ▼
chunking              src/chunking/{fixed_size,content_aware}.py
   │  bytes -> list[bytes]
   ▼
feature extraction    src/features/extract.py
   │  chunk -> 271-dim vector (entropy, 256-bin byte histogram,
   │  printable ratio, byte mean/variance, top-10 bigram
   │  frequencies, chunk length)
   ▼
algorithm selection   src/model/predict.py  (or brute-force, for training labels)
   │  features -> predicted algorithm name
   ▼
compression            src/compressors/registry.py
   │  chunk + algorithm -> compressed bytes
   ▼
archive format
   │  header + per-chunk metadata table + payload blob
   ▼
archive file
```

The model itself is a scikit-learn `HistGradientBoostingClassifier`,
trained on 5,045 labeled chunks (`src/model/train.py`). Labels come from
brute force (`src/labeling/brute_force_label.py`) - every chunk in the
training set actually gets compressed with every candidate algorithm, and
whichever comes out smallest is the label.

There are two chunkers. `fixed_size` cuts equal-sized blocks - 4KB by
default when it's driven through the archive/CLI path (that's what the
shipped model trained on), though the module's own default if you call it
directly is 16KB. `content_aware` segments text/binary runs with a
sliding-window scan; see the Findings section for what it's actually good
at. It's opt-in, and under `mode="model"` its chunk boundaries look
nothing like what the model trained on, so nobody's benchmarked
predictions there.

### Two archive pipelines

This repo has two separate implementations of "chunk metadata + payload
blob," both built on `src/compressors/registry.py` for algorithm ids:

`src/archive/{format,pack,unpack}.py` is the original one - loads the
whole file into memory, chunks and compresses it, writes one `.szip`
archive in a single pass. `src/benchmark/compare.py` and the GUI's
Analysis tab still use it internally, and you can drive it directly with
`python -m src.archive.pack input output.szip --mode {brute_force,model}`.

`encoder/`, `decoder/`, and `scripts/ml_to_manifest.py` are what actually
ships to users now - `scripts/compress.py`/`decompress.py` and the GUI's
Compress/Decompress tabs run through this. It streams: at most one
chunk's bytes are in memory at a time, no matter how big the file is.
`ml_to_manifest.py` does the chunking and prediction and writes those
decisions to a manifest; `encoder/encode.py` reads the manifest back,
re-checksums every chunk against the actual source file to catch a stale
or hand-edited manifest, and streams out the compressed archive in its
own format, `SZE1`, which carries a per-chunk SHA-256 the older `.szip`
format lacks. `decoder/decode.py` reverses all of that.

The `src/archive` path only sticks around because benchmarking and the
analysis view still call it. New work belongs in `encoder`/`decoder`.

## Project layout

```
data/                  raw sample files, downloaded corpora, labeled datasets
src/chunking/          fixed-size and content-aware (printable-ratio + entropy) chunking
src/compressors/       thin wrappers exposing a uniform compress/decompress API
src/features/          per-chunk feature extraction for the ML model
src/labeling/          brute-force labeling to build training data
src/model/             training, inference, and naive baselines
src/archive/           original in-memory .szip format, pack/unpack
src/benchmark/         comparisons against single-algorithm baselines
encoder/               streaming SZE1 encoder + manifest schema
decoder/               streaming SZE1 decoder
scripts/               CLI entry points (compress, decompress, build_dataset, run_training, run_benchmark, ml_to_manifest)
app.py                 Streamlit GUI (Compress / Decompress / Analysis)
tests/                 unit tests
notebooks/             exploratory analysis
```

## Testing

```bash
pytest
```

132 tests. Chunking, feature extraction, compressors, labeling, both
archive pipelines end to end, the ML-to-manifest connector.

## Findings

### Class weighting

Started with a 30-file corpus (~6MB), then added a 14MB Silesia subset on
the theory that those files would have more internal heterogeneity. That
addition skewed the label distribution hard toward brotli (40% to 74% of
all labels - Silesia's text-heavy files compress best under brotli almost
across the board), and just retraining on the combined corpus made the
held-out ratio worse: 2.179x down to 1.935x, even though the brute-force
ceiling hadn't moved at all. lzma recall on the test split dropped to 2%.
The model had basically stopped predicting lzma.

| | unweighted | `class_weight="balanced"` |
|---|---|---|
| overall accuracy | 79.9% | 92.3% |
| lzma precision / recall | 0.11 / 0.02 | 0.90 / 0.93 |
| zstd precision / recall | 0.00 / 0.00 | 0.89 / 1.00 |

Setting `class_weight="balanced"` on the `HistGradientBoostingClassifier`
fixed it and then some - 2.186x on the held-out benchmark, actually a
touch above the original 30-file model's 2.179x, and it kept the bigger
training set. What's left is mostly brotli getting confused with bzip2 -
both do well on similar text-like content, a narrow, specific confusion.
`models/` still has all three variants - 30-file, 34-file unweighted,
34-file weighted - for comparison; the weighted 34-file one ships as the
default.

Growing a corpus without watching what it does to class balance can make
a model measurably worse even when every new example is individually
fine and the brute-force ceiling doesn't budge.

### Content-aware chunking

The idea was that a printable-ratio sliding window could find text/binary
boundaries inside a file. On the first 30-file corpus it basically never
fired - collapsed to ~1 chunk per file, because it turned out every file
in that corpus was internally homogeneous.

Tested four Silesia files directly as plausible mixed-content candidates
- `dickens`, `webster`, `samba`, `xml` - sweeping window size, printable
threshold, and minimum chunk size from default all the way to
aggressively loose:

| file | default | loosest tested | responds to tuning? |
|---|---|---|---|
| dickens | 1 chunk | 1 chunk | no |
| webster | 1 chunk | 1 chunk | no |
| samba | 270 chunks | 600 chunks | yes |
| xml | 40 chunks | 43 chunks | yes |

`dickens` and `webster` never budged at any setting. `webster` is exactly
100.0000% printable ASCII the whole way through - there's nothing there
for the heuristic to find, no matter how loose the thresholds get. `samba`
and `xml` responded right away once tested directly. So the heuristic
works fine; the first corpus just happened to be made of files with no
internal mixing to detect.

So the next step was building files that actually have the structure
this thing is supposed to find (`tests/test_content_aware_mixed_files.py`),
and comparing against `fixed_size(4KB)` under `mode="brute_force"` for
both, to isolate the chunker from the model:

| file | content_aware chunks | fixed_size chunks | ratio: content_aware | ratio: fixed_size | winner |
|---|---|---|---|---|---|
| multi-part (clean alternating text/binary) | 12 | 9 | 1.563x | 1.518x | content_aware |
| structured log + binary payload dumps | 20 | 13 | 2.340x | 2.384x | fixed_size |
| HTML with embedded base64 image blob | 16 | 9 | 2.105x | 2.117x | fixed_size |

Clean alternation is where it's supposed to shine, and the ratio backs
that up. The other two are more interesting. In both, content_aware's
boundaries tracked the real text/binary transitions closely - you can
check this, printable ratios per chunk cluster near 1.0 or near 0.4,
right where the underlying material actually is - and it still lost on
ratio. Its chunks come out smaller on average than fixed_size's flat
4KB, and apparently that costs more than getting the content type right
saves. Splitting a file correctly by content type turned out to be a
separate problem from getting a better compression ratio.

Also found an actual bug while building these test files: base64-encoded
binary - a data-URI image embedded in HTML, say - is printable ASCII by
definition, so the original printable-ratio-only check couldn't
distinguish it from surrounding prose at all. The HTML file above used to
collapse into a single chunk. Fixed it by also checking Shannon entropy
(`ENTROPY_THRESHOLD = 5.0` bits/byte - prose and source code sit around
4.2-4.3, base64 of random data sits around 5.9, plenty of room between
them). A window only counts as "text" now if it's printable and under the
entropy threshold, and the HTML case above now splits into 16 chunks like
it should. That didn't fix everything, though - hex-encoded data only
uses 16 symbols, so its entropy tops out at 4 bits/byte, at or below
normal prose, and entropy alone can't see it. Closed that with a third
check: a window is also classified "binary" if `HEX_RATIO_THRESHOLD`
(95%) of its bytes are `[0-9a-fA-F]`, since prose and source code don't
run long stretches of nothing but hex digits by accident
(`test_hex_encoded_binary_now_detected_via_character_class`, guarded
against false positives on hex-digit-heavy prose by
`test_prose_with_incidental_hex_like_words_is_not_misclassified`).

## Limitations

- **Small training corpus.** 34 files, ~20MB, 5,045 labeled chunks. Big
  enough to surface the class-imbalance problem above, nowhere near big
  enough to cover real-world file types. This is also currently blocked -
  `scripts/download_corpora.sh` is an empty stub, and the corpus that
  actually produced the shipped model isn't checked into this repo.
- **gzip isn't in the default candidate set.** It won brute-force
  selection on 5 out of 5,045 chunks, never by much - strictly dominated
  by zstd/brotli at this chunk size. Still works via `--algorithm gzip`
  or `registry.list_algorithms(include_excluded=True)`, it's just not
  tried automatically anymore. That's an artifact of this corpus and
  these chunk sizes, and might not hold at a different chunk size or on
  different data.
- **brotli/bzip2 still get confused for each other.** This is the biggest
  remaining error mode. Both algorithms are strong on similar text-like
  content, so the model has to draw a boundary that's genuinely fuzzy in
  the underlying data - more training examples right at that boundary
  would probably help, but it needs the missing corpus to test.
- **content_aware's character-class check is untuned beyond the synthetic
  tests that motivated it.** `HEX_RATIO_THRESHOLD` closes the hex-encoded
  blind spot on the specific hex/prose mix built to expose it, but hasn't
  been swept the way `ENTROPY_THRESHOLD` was, and doesn't generalize to
  other restricted alphabets (base32, uuencoding) - those remain
  invisible to all three signals.
- **content_aware under `mode="model"` isn't benchmarked.** Its chunk
  sizes and boundaries don't look like the ~4KB fixed_size chunks the
  model trained on. Whatever the classifier does with a chunk that shape
  is extrapolation, and nobody's checked how good or bad that
  extrapolation is. Only `mode="brute_force"` has a verified accuracy
  guarantee.
- **Every algorithm is a fixed, off-the-shelf codec.** gzip, bzip2, lzma,
  zstd, brotli - nothing learned. A neural entropy coder as another
  candidate, or as a replacement for brute-force labeling's ground truth,
  is plausible but a much bigger project than anything else here.

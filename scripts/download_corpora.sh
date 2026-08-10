#!/usr/bin/env bash
# Download diverse corpus of files for training/benchmarking 
#
# Aims for variety across content types (text, source code, binaries, already-compressed archives) 
# so labeled dataset covers a broad range of per-chunk compression behavior
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/data/corpora"
mkdir -p "${OUT_DIR}"

echo "Corpora will be downloaded into: ${OUT_DIR}"

download() {
    local name="$1"
    local url="$2"
    local dest="${OUT_DIR}/${name}"
    if [[ -e "${dest}" ]]; then
        echo "skip ${name} (already exists)"
        return
    fi
    echo "fetching ${name} ..."
    curl -fSL --retry 3 -o "${dest}" "${url}"
}

# TODO: 
# populate with actual corpus sources, compression corpora, snapshot of enwik8, sample source-code archives, handful of already-compressed files (png/zip) as negative examples

# download "silesia.zip" "https://example.invalid/silesia.zip"
# download "enwik8.zip"  "https://example.invalid/enwik8.zip"

echo "Not configured yet — add download entries above"

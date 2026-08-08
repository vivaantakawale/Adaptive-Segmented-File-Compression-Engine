"""
local demo - Streamlit

Run with: streamlit run app.py

Tabs: Compress, Decompress, Analysis
"""

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.archive import format as fmt
from src.archive.pack import pack_bytes
from src.benchmark.compare import DEFAULT_CHUNK_SIZE, load_chunks, run_all
from src.model.predict import DEFAULT_MODEL_PATH
from scripts.compress import compress
from scripts.decompress import decompress

LARGE_FILE_WARNING_BYTES = 20 * 1024 * 1024


def human_size(n: int) -> str:
    """
    Render byte count as readable string

    Args:
        n: Byte count

    Returns:
        Formatted string 
    """
    if n < 1024:
        return f"{n:,} B"
    value = float(n)
    for unit in ("KB", "MB", "GB"):
        value /= 1024
        if value < 1024:
            return f"{value:.1f} {unit}"
    return f"{value:.1f} TB"


st.set_page_config(page_title="adaptive-zip")

st.title("Adaptive Segmented File Compression Engine")
st.caption("Local demo only, not deployed")

if not DEFAULT_MODEL_PATH.exists():
    st.error(
        f"No trained model found at `{DEFAULT_MODEL_PATH}`. Train one first "
        "see README's Usage section, then reload this page"
    )
    st.stop()

tab_compress, tab_decompress, tab_analysis = st.tabs(["Compress", "Decompress", "Analysis"])


with tab_compress:
    st.write("Upload file, get back encoded `.vtzip` archive and report")
    up = st.file_uploader("File to compress", type=None, key="compress_upload")

    if up is not None:
        data = up.getvalue()
        size = len(data)
        st.write(f"**{up.name}** - {human_size(size)}")

        if size == 0:
            st.info("Empty file - nothing to compress")
        else:
            if size > LARGE_FILE_WARNING_BYTES:
                st.warning(
                    f"File is over {LARGE_FILE_WARNING_BYTES // (1024 * 1024)}MB. "
                    "Compression linear in file size, but browser upload may take a while"
                )

            if st.button("Compress", type="primary"):
                try:
                    with st.spinner("Chunking, predicting, and compressing..."):
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            tmp_dir = Path(tmp_dir)
                            in_path = tmp_dir / up.name
                            in_path.write_bytes(data)
                            out_path = tmp_dir / (up.name + ".vtzip")

                            summary = compress(in_path, out_path)
                            archive_bytes = out_path.read_bytes()
                except Exception as exc:
                    st.error(f"Compression failed: {exc}")
                else:
                    st.success(f"Compressed to {summary['ratio']:.3f}x ({human_size(summary['compressed_size'])})")
                    if summary["ratio"] < 1:
                        st.info(
                            "Ratio below 1x - archive is larger than original. "
                            "Can happen on small or already incompressible files, where per-chunk header/metadata overhead outweighs savings"
                        )

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Original size", human_size(summary["original_size"]))
                    col2.metric("Compressed size", human_size(summary["compressed_size"]))
                    col3.metric("Ratio", f"{summary['ratio']:.3f}x")

                    st.subheader("Algorithm breakdown")
                    algo_df = pd.DataFrame(
                        {"chunk_count": summary["algorithm_counts"]}
                    ).sort_values("chunk_count", ascending=False)
                    st.bar_chart(algo_df)
                    algo_df["pct"] = (100 * algo_df["chunk_count"] / summary["chunk_count"]).round(1)
                    st.dataframe(algo_df, use_container_width=True)
                    st.caption(f"{summary['chunk_count']} chunks total")

                    st.download_button(
                        "Download encoded file",
                        data=archive_bytes,
                        file_name=up.name + ".vtzip",
                        mime="application/octet-stream",
                    )


with tab_decompress:
    st.write("Upload a `.vtzip` archive produced by this tool, get back original file")
    up = st.file_uploader("Archive to decompress", type=None, key="decompress_upload")

    if up is not None:
        data = up.getvalue()
        st.write(f"**{up.name}** - {human_size(len(data))}")

        if len(data) == 0:
            st.info("Empty file - nothing to decompress")
        elif not up.name.endswith(".vtzip"):
            st.warning(
                "This file doesn't have the `.vtzip` extension. "
                "You can still try decompressing it, but likely not a valid archive"
            )

        if len(data) > 0 and st.button("Decompress", type="primary"):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = Path(tmp_dir)
                archive_path = tmp_dir / up.name
                archive_path.write_bytes(data)
                out_name = up.name[:-6] if up.name.endswith(".vtzip") else up.name + ".restored"
                out_path = tmp_dir / out_name

                try:
                    with st.spinner("Decoding and verifying integrity..."):
                        summary = decompress(archive_path, out_path)
                        restored_bytes = out_path.read_bytes()
                except Exception as exc:
                    st.error(f"Decompression failed: {exc}")
                    st.caption(
                        "Not a '.vtzip' file"
                    )
                else:
                    st.success(
                        f"Restored {human_size(summary['restored_size'])} - "
                        "per-chunk and whole-file checksums verified"
                    )
                    st.download_button(
                        "Download restored file",
                        data=restored_bytes,
                        file_name=out_name,
                        mime="application/octet-stream",
                    )


with tab_analysis:
    st.write("Compares `model_predicted` against single algorithm baselines and the brute force ceiling for exploring how close the model gets")
    up = st.file_uploader("File to analyze", type=None, key="analysis_upload")

    if up is not None:
        data = up.getvalue()
        size = len(data)
        st.write(f"**{up.name}** - {human_size(size)}")

        if size == 0:
            st.info("Empty file - nothing to compress")
        else:
            if size > LARGE_FILE_WARNING_BYTES:
                st.warning(
                    f"This file is over {LARGE_FILE_WARNING_BYTES // (1024 * 1024)}MB. "
                    "`brute_force_ceiling` tries every algorithm on every chunk, so this may take a while"
                )

            with st.spinner("Chunking, compressing, and predicting..."):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir) / up.name
                    tmp_path.write_bytes(data)

                    chunks = load_chunks([tmp_path], chunk_size=DEFAULT_CHUNK_SIZE)
                    results = run_all(chunks, model_path=DEFAULT_MODEL_PATH)

                archive = pack_bytes(data, mode="model", chunker="fixed_size", chunk_size=DEFAULT_CHUNK_SIZE)

            result_by_name = {r.name: r for r in results}
            model_result = result_by_name["model_predicted"]
            brute_result = result_by_name["brute_force_ceiling"]

            st.subheader("Compressed size by strategy")
            size_df = pd.DataFrame(
                {"compressed_size": {r.name: r.total_compressed for r in results}}
            )
            st.bar_chart(size_df)

            st.subheader("Overall ratio")
            col1, col2 = st.columns(2)
            col1.metric("model_predicted", f"{model_result.ratio:.3f}x")
            col2.metric(
                "brute_force_ceiling",
                f"{brute_result.ratio:.3f}x",
                help="Best possible per-chunk ratio -- tries every algorithm on every chunk.",
            )
            st.caption(
                f"model_predicted reaches {model_result.ratio / brute_result.ratio * 100:.1f}% "
                "of the brute-force ceiling's ratio"
            )

            st.subheader("Which algorithm won each chunk (model mode)")
            st.caption(
                "Read directly off the packed archive's own chunk metadata table "
                "(src.archive.format), not re-derived -- this is what actually got used"
            )
            header = fmt.read_archive_header(archive)
            offset = fmt.metadata_table_offset()
            algo_counts: dict[str, int] = {}
            for _ in range(header.chunk_count):
                meta = fmt.read_chunk_meta(archive, offset)
                offset += fmt.chunk_meta_size()
                name = fmt.registry_name(meta.algorithm_id)
                algo_counts[name] = algo_counts.get(name, 0) + 1

            algo_df = pd.DataFrame(
                {"chunk_count": algo_counts}
            ).sort_values("chunk_count", ascending=False)
            st.bar_chart(algo_df)
            st.dataframe(algo_df, use_container_width=True)
            st.caption(f"{header.chunk_count} chunks total, {DEFAULT_CHUNK_SIZE // 1024}KB each (last chunk may be smaller)")

            st.subheader("Speed: model vs. brute-force ceiling")
            speed_df = pd.DataFrame(
                {"seconds": {"model_predicted": model_result.seconds, "brute_force_ceiling": brute_result.seconds}}
            )
            st.bar_chart(speed_df)
            if model_result.seconds > 0:
                st.caption(
                    f"model_predicted: {model_result.seconds:.3f}s vs. "
                    f"brute_force_ceiling: {brute_result.seconds:.3f}s "
                    f"({brute_result.seconds / model_result.seconds:.1f}x faster)"
                )

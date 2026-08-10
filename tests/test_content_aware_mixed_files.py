"""Content-aware chunking on genuinely mixed-content files: 
clean text/binary alternation, base64-encoded blobs, and hex-encoded blobs
"""

import base64
import os

from src.chunking import content_aware


def test_alternating_text_and_binary_sections_produce_multiple_chunks():
    text_section = (b"the quick brown fox jumps over the lazy dog. " * 80)  # ~3.7KB
    binary_section = os.urandom(3000)
    data = (text_section + binary_section) * 3

    chunks = list(content_aware.chunk(data))

    assert len(chunks) > 1
    assert b"".join(chunks) == data
    for c in chunks:
        ratio = content_aware.printable_ratio(c)
        assert ratio >= 0.9 or ratio <= 0.6


def test_base64_encoded_binary_now_detected_via_entropy():
    prose = b"<p>" + b"some perfectly ordinary paragraph text here. " * 60 + b"</p>\n"
    blob = base64.b64encode(os.urandom(3000))  # pure binary, but 100% printable once encoded
    img_tag = b'<img src="data:image/png;base64,' + blob + b'">\n'
    data = prose + img_tag + prose

    chunks = list(content_aware.chunk(data))

    assert len(chunks) > 1  # blob gets split out from surrounding prose
    assert b"".join(chunks) == data
    assert content_aware.printable_ratio(data) >= content_aware.PRINTABLE_THRESHOLD


def test_hex_encoded_binary_now_detected_via_character_class():
    prose = b"<p>" + b"some perfectly ordinary paragraph text here. " * 60 + b"</p>\n"
    hex_blob = os.urandom(3000).hex().encode("ascii")  # binary data, hex-encoded (printable)
    data = prose + hex_blob + prose

    chunks = list(content_aware.chunk(data))

    assert len(chunks) > 1  # hex blob gets split out from surrounding prose
    assert b"".join(chunks) == data
    assert content_aware.printable_ratio(data) >= content_aware.PRINTABLE_THRESHOLD


def test_prose_with_incidental_hex_like_words_is_not_misclassified():
    # "cab", "beef", "decaf", "facade" etc. shouldn't trip hex-ratio threshold once diluted by rest of alphabet and punctuation
    prose = (
        b"<p>a cab full of beef and a decaf, "
        b"the facade of a deed abed, ace cafe deadbeef cabbage. </p>\n"
    ) * 20

    chunks = list(content_aware.chunk(prose))

    assert len(chunks) == 1  # still reads as text, no false split
    assert b"".join(chunks) == prose

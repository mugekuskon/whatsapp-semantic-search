import re
import unicodedata

_WHITESPACE_RE = re.compile(r"[\s ]+")
_CONTROL_RE = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")
_PUNCT_MAP = str.maketrans({
    "‘": "'", "’": "'",   # left/right single quotation marks
    "“": '"', "”": '"',   # left/right double quotation marks
    "–": "-", "—": "-",   # en-dash, em-dash
    "…": "...",                # ellipsis
})


def normalize_text(text: str) -> str:
    """Canonical text normalization applied before embedding.

    Must be used identically at ingest time and query time —
    any divergence causes query and document vectors to land in
    different parts of the embedding space.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_PUNCT_MAP)
    text = _CONTROL_RE.sub("", text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text

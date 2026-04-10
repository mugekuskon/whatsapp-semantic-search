import logging
import re
import unicodedata
from typing import Any
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"[\s\u00a0]+")

# Matches C0/C1 control characters except newline (kept for multi-line messages)
_CONTROL_RE = re.compile(r"[\x00-\x09\x0b-\x1f\x7f-\x9f]")

_PUNCT_MAP = str.maketrans({
    "\u2018": "'", "\u2019": "'",   # left/right single quotation marks
    "\u201c": '"', "\u201d": '"',   # left/right double quotation marks
    "\u2013": "-", "\u2014": "-",   # en-dash, em-dash
    "\u2026": "...",                # ellipsis
})


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_PUNCT_MAP)
    text = _CONTROL_RE.sub("", text)
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text

MODEL_NAME = EMBEDDING_MODEL
EXPECTED_DIM = 384


class EmbeddingManager:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        log.info("Loaded model '%s' (dim=%d)", self.model_name, self.dim)

    def embed_single(self, text: str) -> list[float]:
        """Normalize then embed one string; return a flat list of floats. Returns [] for empty input."""
        if not text:
            return []
        vector = self.model.encode(_normalize(text))
        return vector.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Normalize then embed a list of strings in batches; returns a list of embedding vectors. Returns [] for empty input."""
        if not texts:
            return []
        normalized = [_normalize(t) for t in texts]
        vectors = self.model.encode(normalized, batch_size=batch_size, show_progress_bar=True)
        return [v.tolist() for v in vectors]

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        """
        Accept our chunk format from data_processor.py, embed each chunk's
        `chunk_text` field, and return the same dicts with an added `embedding` key.
        """
        if not chunks:
            return []
        texts = [c.get("chunk_text", "") for c in chunks]
        vectors = self.embed_batch(texts)
        return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom else 0.0


if __name__ == "__main__":
    manager = EmbeddingManager()

    test_chunks = json.loads((Path(__file__).parent / "test_chunks.json").read_text(encoding="utf-8"))

    result = manager.embed_chunks(test_chunks)

    for i, chunk in enumerate(result):
        emb = chunk["embedding"]
        assert isinstance(emb, list), f"Chunk {i}: embedding must be a list"
        assert all(isinstance(v, float) for v in emb), f"Chunk {i}: all values must be floats"
        assert len(emb) == EXPECTED_DIM, f"Chunk {i}: expected dim {EXPECTED_DIM}, got {len(emb)}"

    # Similar texts (coffee meeting) should score higher than unrelated pair
    sim_related = _cosine_similarity(result[0]["embedding"], result[1]["embedding"])
    sim_unrelated = _cosine_similarity(result[0]["embedding"], result[2]["embedding"])
    assert sim_related > sim_unrelated, (
        f"Similarity check failed: related={sim_related:.4f}, unrelated={sim_unrelated:.4f}"
    )

    print(f"\nModel      : {manager.model_name}")
    print(f"Dimension  : {manager.dim}")
    print(f"Chunks     : {len(result)}")
    print(f"Sim (related)  : {sim_related:.4f}")
    print(f"Sim (unrelated): {sim_unrelated:.4f}")
    print("All assertions passed.")

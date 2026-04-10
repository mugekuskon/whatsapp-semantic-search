"""
RAG (Retrieval-Augmented Generation) pipeline for WhatsApp semantic search.

Flow:
    1. hybrid_search()  — retrieve top N relevant chunks from ChromaDB
    2. build_prompt()   — format chunks into a structured English prompt
    3. generate()       — send prompt to local Ollama model and stream answer
"""

import logging
import time
import ollama
from search_engine import hybrid_search

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OLLAMA_MODEL = "aya-expanse:8b"
N_RESULTS = 3


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a structured English prompt.

    The prompt instructs the model to:
    - Answer only from the provided context
    - Admit when the answer is not found in the messages
    - Keep the answer concise and grounded in the chat history
    """
    context_blocks = []
    for chunk in chunks:
        meta         = chunk["meta"]
        source       = meta.get("source") or "unknown"
        start        = meta.get("start_datetime") or "unknown"
        end          = meta.get("end_datetime") or "unknown"
        participants = meta.get("participants") or "unknown"
        text         = chunk["text"]
        context_blocks.append(
            f"[Kaynak: {source} | {start} — {end}]\n"
            f"[Kişiler: {participants}]\n"
            f"{text}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    return f"""You are a helpful assistant answering questions about a person's WhatsApp chat history.

STRICT RULES:
1. Answer ONLY using information explicitly stated in the chat snippets below.
2. Do NOT infer, guess, or add any information not present in the messages.
3. If the answer is not clearly in the snippets, say exactly: "Bu bilgi sohbet geçmişinde bulunamadı."
4. When referencing a message, cite it using the source name and time (e.g. "friend_group sohbetinde, Aralık 2025'te...").
5. Answer in Turkish regardless of the question language.
6. Keep the answer short and factual.

--- CHAT SNIPPETS ---

{context}

--- QUESTION ---
{question}

--- ANSWER ---"""


def _parse_results(raw_results: dict) -> list[dict]:
    """Convert hybrid_search() output into a flat list of chunk dicts."""
    docs  = raw_results.get("documents", [[]])[0]
    metas = raw_results.get("metadatas", [[]])[0]
    return [{"text": doc, "meta": meta} for doc, meta in zip(docs, metas)]


def ask(question: str, n_results: int = N_RESULTS, months_ago: int | None = None) -> str:
    """
    Full RAG pipeline: retrieve → augment → generate.

    Args:
        question:   The user's natural language question.
        n_results:  Number of chunks to retrieve and pass as context.
        months_ago: Optional — restrict search to last N months.

    Returns:
        The model's answer as a string.
    """
    t0 = time.perf_counter()

    # 1. Retrieve
    raw_results = hybrid_search(question, n_results=n_results, months_ago=months_ago)
    chunks = _parse_results(raw_results)
    t_retrieve = time.perf_counter()

    if not chunks:
        return "No relevant conversations found in the chat history."

    # 2. Augment
    prompt = build_prompt(question, chunks)
    t_augment = time.perf_counter()

    # 3. Generate
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    t_generate = time.perf_counter()

    log.info(
        "Timing — retrieve: %.2fs | augment: %.2fs | generate: %.2fs | total: %.2fs",
        t_retrieve - t0,
        t_augment - t_retrieve,
        t_generate - t_augment,
        t_generate - t0,
    )

    return response["message"]["content"].strip()


def ask_stream(question: str, n_results: int = N_RESULTS, months_ago: int | None = None):
    """
    Same as ask() but streams the response token by token.
    Yields string chunks as they arrive from the model.
    After the stream ends, logs a timing breakdown.
    """
    t0 = time.perf_counter()

    raw_results = hybrid_search(question, n_results=n_results, months_ago=months_ago)
    chunks = _parse_results(raw_results)
    t_retrieve = time.perf_counter()

    if not chunks:
        yield "No relevant conversations found in the chat history."
        return

    prompt = build_prompt(question, chunks)
    t_augment = time.perf_counter()

    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]

    t_generate = time.perf_counter()
    log.info(
        "Timing — retrieve: %.2fs | augment: %.2fs | generate: %.2fs | total: %.2fs",
        t_retrieve - t0,
        t_augment - t_retrieve,
        t_generate - t_augment,
        t_generate - t0,
    )


if __name__ == "__main__":
    print(f"WhatsApp RAG — powered by Ollama / {OLLAMA_MODEL}")
    print("Type your question and press Enter. Ctrl+C to quit.\n")

    while True:
        try:
            question = input("Question: ").strip()
            if not question:
                continue
            print("\nAnswer: ", end="", flush=True)
            for token in ask_stream(question):
                print(token, end="", flush=True)
            print("\n")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

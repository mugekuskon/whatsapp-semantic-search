"""
RAG (Retrieval-Augmented Generation) pipeline for WhatsApp semantic search.

Flow:
    1. hybrid_search()  — retrieve top N relevant chunks from ChromaDB
    2. build_prompt()   — format chunks into a structured English prompt
    3. generate()       — send prompt to local Ollama model and stream answer
"""

import logging
import ollama
from search_engine import hybrid_search

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OLLAMA_MODEL = "qwen2.5:7b-instruct-q4_k_m"  # Ollama model name to use for generation
N_RESULTS = 5


def build_prompt(question: str, chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a structured English prompt.

    The prompt instructs the model to:
    - Answer only from the provided context
    - Admit when the answer is not found in the messages
    - Keep the answer concise and grounded in the chat history
    """
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["meta"]
        time_range = f"{meta.get('start_datetime') or 'unknown'} → {meta.get('end_datetime') or 'unknown'}"
        participants = meta.get("participants") or "unknown"
        text = chunk["text"]
        context_blocks.append(
            f"[Conversation {i}]\n"
            f"Time     : {time_range}\n"
            f"People   : {participants}\n"
            f"Messages :\n{text}"
        )

    context = "\n\n".join(context_blocks)

    return f"""You are a helpful assistant that answers questions about WhatsApp chat history.
You are given several conversation snippets retrieved from the chat logs.
Answer the user's question using ONLY the information found in these conversations.
If the answer cannot be found in the provided conversations, say so clearly.
Keep your answer concise and directly grounded in the chat messages.
The conversations may be in Turkish — understand them but answer in the same language as the question.

---
RETRIEVED CONVERSATIONS:

{context}

---
QUESTION: {question}

ANSWER:"""


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
    # 1. Retrieve
    log.info("Retrieving top %d chunks for: %s", n_results, question)
    raw_results = hybrid_search(question, n_results=n_results, months_ago=months_ago)
    chunks = _parse_results(raw_results)

    if not chunks:
        return "No relevant conversations found in the chat history."

    # 2. Augment
    prompt = build_prompt(question, chunks)
    log.info("Prompt built with %d conversation chunks.", len(chunks))

    # 3. Generate
    log.info("Sending to Ollama model '%s'...", OLLAMA_MODEL)
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"].strip()


def ask_stream(question: str, n_results: int = N_RESULTS, months_ago: int | None = None):
    """
    Same as ask() but streams the response token by token.
    Yields string chunks as they arrive from the model.
    """
    raw_results = hybrid_search(question, n_results=n_results, months_ago=months_ago)
    chunks = _parse_results(raw_results)

    if not chunks:
        yield "No relevant conversations found in the chat history."
        return

    prompt = build_prompt(question, chunks)

    stream = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    for part in stream:
        yield part["message"]["content"]


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

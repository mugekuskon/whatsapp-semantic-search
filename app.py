"""
Gradio interface for WhatsApp Semantic Search.

Run:
    python app.py

Then open http://localhost:7860 in your browser.

Two tabs:
  Ask    — full RAG pipeline, streams the answer token by token
  Search — raw hybrid search results, useful for inspecting retrieval quality
"""

import logging
import ollama
import gradio as gr
from rag import ask_stream, OLLAMA_MODEL
from search_engine import hybrid_search, get_db_collection

log = logging.getLogger(__name__)


def _warmup():
    """Load the embedding model and Ollama into memory before the first query."""
    log.info("Warming up — loading ChromaDB collection and embedding model...")
    get_db_collection()
    log.info("Warming up — loading Ollama model '%s'...", OLLAMA_MODEL)
    ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "merhaba"}])
    log.info("Warmup complete — ready.")


_warmup()

def stream_answer(question: str, n_results: int, months_ago: int, use_months: bool):
    """Stream the RAG answer token by token into the Gradio chatbot."""
    if not question.strip():
        yield "Lütfen bir soru girin."
        return

    months = months_ago if use_months else None
    answer = ""
    for token in ask_stream(question, n_results=n_results, months_ago=months):
        answer += token
        yield answer

def run_search(query: str, n_results: int, months_ago: int, use_months: bool, use_reranker: bool):
    """Run hybrid search and format results as readable markdown."""
    if not query.strip():
        return "Lütfen bir arama sorgusu girin."

    months = months_ago if use_months else None
    results = hybrid_search(query, n_results=n_results, months_ago=months, use_reranker=use_reranker)

    docs          = results.get("documents", [[]])[0]
    metas         = results.get("metadatas", [[]])[0]
    distances     = results.get("distances", [[]])[0]
    rerank_scores = results.get("rerank_scores", [[]])[0] or [None] * len(docs)

    if not docs:
        return "Sonuç bulunamadı."

    lines = []
    for i, (doc, meta, dist, rscore) in enumerate(
        zip(docs, metas, distances, rerank_scores), start=1
    ):
        if rscore is not None:
            match_label = f"rerank score: {rscore:.3f}"
        elif dist >= 1.0:
            match_label = "keyword match"
        else:
            match_label = f"similarity: {(1 - dist) * 100:.1f}%"

        start = meta.get("start_datetime", "")[:16] or "?"
        end   = meta.get("end_datetime",   "")[:16] or "?"
        people = meta.get("participants", "?")
        source = meta.get("source", "?")

        lines.append(f"### Result #{i} — {match_label}")
        lines.append(f"**Time:** {start} → {end}  |  **From:** {people}  |  **Source:** {source}")
        lines.append("")
        lines.append(f"```\n{doc}\n```")
        lines.append("")

    return "\n".join(lines)

def _shared_controls():
    """Returns (n_results, use_months, months_ago) widgets."""
    n_results = gr.Slider(
        minimum=1, maximum=10, step=1, value=3,
        label="Number of results",
        info="How many chunks to retrieve and show",
    )
    use_months = gr.Checkbox(
        value=False,
        label="Filter by time",
        info="Restrict search to the last N months",
    )
    months_ago = gr.Number(
        value=6, minimum=1, maximum=120, precision=0,
        label="Months ago",
        visible=False,
    )
    use_months.change(fn=lambda x: gr.update(visible=x), inputs=use_months, outputs=months_ago)
    return n_results, use_months, months_ago


with gr.Blocks(title="WhatsApp Semantic Search") as demo:

    gr.Markdown("# WhatsApp Semantic Search")
    gr.Markdown("Ask questions about your chat history or search for specific messages.")

    with gr.Tab("Ask"):
        gr.Markdown("Full RAG pipeline — retrieves relevant chunks and generates an answer in Turkish.")

        ask_input = gr.Textbox(
            placeholder="Ne zaman buluştuk?",
            label="Question",
            lines=2,
        )
        with gr.Row():
            ask_n, ask_use_months, ask_months = _shared_controls()

        ask_btn    = gr.Button("Ask", variant="primary")
        ask_output = gr.Textbox(label="Answer", lines=8, interactive=False)

        ask_btn.click(
            fn=stream_answer,
            inputs=[ask_input, ask_n, ask_months, ask_use_months],
            outputs=ask_output,
        )
        ask_input.submit(
            fn=stream_answer,
            inputs=[ask_input, ask_n, ask_months, ask_use_months],
            outputs=ask_output,
        )

    with gr.Tab("Search"):
        gr.Markdown("Raw hybrid search results — useful for inspecting retrieval quality.")

        search_input = gr.Textbox(
            placeholder="uçak bileti",
            label="Query",
            lines=2,
        )
        with gr.Row():
            search_n, search_use_months, search_months = _shared_controls()
            use_reranker = gr.Checkbox(
                value=True,
                label="Use reranker",
                info="Cross-encoder reranking (slower but more accurate)",
            )

        search_btn    = gr.Button("Search", variant="primary")
        search_output = gr.Markdown(label="Results")

        search_btn.click(
            fn=run_search,
            inputs=[search_input, search_n, search_months, search_use_months, use_reranker],
            outputs=search_output,
        )
        search_input.submit(
            fn=run_search,
            inputs=[search_input, search_n, search_months, search_use_months, use_reranker],
            outputs=search_output,
        )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

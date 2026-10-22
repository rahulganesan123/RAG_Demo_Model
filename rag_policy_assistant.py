"""
================================================================================
                      RAG POLICY ASSISTANT
        Retrieval-Augmented Generation over Company PDF Documents
================================================================================

Ask natural-language questions about your company policy PDFs and get
grounded answers that cite the source document and page.

WORKFLOW
--------

  +-----------+     +-------------+     +-------------------+
  | PDF files | --> | Extract text| --> | Clean + sentence  |
  |  (data/)  |     | (per page)  |     | chunking (overlap)|
  +-----------+     +-------------+     +---------+---------+
                                                  |
                                                  v
  +-----------+     +-------------+     +-------------------+
  | User      | --> | Embed query | --> | FAISS similarity  |
  | question  |     | (MiniLM)    |     | search (top-k)    |
  +-----------+     +-------------+     +---------+---------+
                                                  |
                                                  v
                    +-------------+     +-------------------+
                    | Final answer| <-- | Gemini LLM with   |
                    | + citations |     | context-only prompt|
                    +-------------+     +-------------------+

  (Chunks are also embedded with MiniLM and stored in FAISS at start-up.)

USAGE
-----
    python rag_policy_assistant.py
    python rag_policy_assistant.py --data-dir data --top-k 3 --chunk-size 3

REQUIREMENTS
------------
    See requirements.txt and set GEMINI_API_KEY in a .env file.
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import faiss
import nltk
import numpy as np
from dotenv import load_dotenv
from google import genai
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gemini-3.5-flash-lite"  # override with GEMINI_MODEL in .env
NOT_FOUND_MESSAGE = "The information is not available in the provided documents."

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("rag_policy_assistant")


@dataclass
class Page:
    """Text extracted from a single PDF page."""
    source: str
    page: int
    text: str


@dataclass
class Chunk:
    """A small piece of text that is embedded and searched."""
    source: str
    page: int
    text: str


# ------------------------------------------------------------------------------
# 1. Load PDF documents
# ------------------------------------------------------------------------------

def load_pdfs(data_dir: Path) -> list[Page]:
    """Read every PDF in `data_dir` and return the text of each page."""
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {data_dir.resolve()}")

    pages: list[Page] = []
    for pdf_path in pdf_files:
        logger.info("Loading: %s", pdf_path.name)
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                pages.append(Page(pdf_path.name, page_number, text))

    logger.info("Loaded %d PDF(s), %d page(s) with text", len(pdf_files), len(pages))
    return pages


# ------------------------------------------------------------------------------
# 2. Chunking
# ------------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove blank lines and join the remaining lines into one string."""
    return " ".join(line.strip() for line in text.splitlines() if line.strip())


def create_chunks(pages: list[Page], chunk_size: int = 3, overlap: int = 1) -> list[Chunk]:
    """
    Split each page into overlapping sentence-based chunks.

    Example (chunk_size=3, overlap=1) on sentences S0..S5:
        [S0 S1 S2], [S2 S3 S4], [S4 S5]
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    step = chunk_size - overlap
    chunks: list[Chunk] = []

    for page in pages:
        sentences = sent_tokenize(clean_text(page.text))
        for start in range(0, len(sentences), step):
            window = sentences[start:start + chunk_size]
            chunks.append(Chunk(page.source, page.page, " ".join(window)))
            if start + chunk_size >= len(sentences):
                break  # last window reached; avoid a duplicate tail chunk

    logger.info("Created %d chunk(s)", len(chunks))
    return chunks


# ------------------------------------------------------------------------------
# 3 & 4. Embeddings and FAISS index
# ------------------------------------------------------------------------------

def build_index(chunks: list[Chunk], model: SentenceTransformer) -> faiss.IndexFlatL2:
    """Embed all chunks and store the vectors in a FAISS index."""
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    logger.info("FAISS index ready: %d vectors, dimension %d", index.ntotal, embeddings.shape[1])
    return index


# ------------------------------------------------------------------------------
# 5. Retrieval
# ------------------------------------------------------------------------------

def retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    chunks: list[Chunk],
    top_k: int = 3,
) -> list[tuple[Chunk, float]]:
    """Return the `top_k` chunks closest to the query (smaller distance = closer)."""
    query_vec = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vec, min(top_k, len(chunks)))
    return [(chunks[i], float(d)) for i, d in zip(indices[0], distances[0]) if i != -1]


# ------------------------------------------------------------------------------
# 6. Augmentation (prompt building)
# ------------------------------------------------------------------------------

def build_prompt(query: str, results: list[tuple[Chunk, float]]) -> str:
    """Combine retrieved chunks and the question into a strict, context-only prompt."""
    context = "\n".join(
        f"Source: {chunk.source}\nPage: {chunk.page}\n\nContent:\n{chunk.text}\n"
        for chunk, _ in results
    )

    return f"""You are a company policy assistant.

Answer the user's question using ONLY the context provided below.

==============================
CONTEXT
==============================
{context}
==============================
QUESTION
==============================
{query}

==============================
RULES
==============================
1. Use only the provided context.
2. Do not invent information.
3. Do not use outside knowledge.
4. If the answer is not available in the context, say:
   "{NOT_FOUND_MESSAGE}"
5. Keep the answer clear and concise.
6. Mention the source document and page when possible.
"""


# ------------------------------------------------------------------------------
# 7. Generation
# ------------------------------------------------------------------------------

def generate_answer(client: genai.Client, model_name: str, prompt: str) -> str:
    """Send the prompt to Gemini and return the answer text."""
    response = client.models.generate_content(model=model_name, contents=prompt)
    return (response.text or "").strip() or NOT_FOUND_MESSAGE


# ------------------------------------------------------------------------------
# Main application
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RAG assistant for company policy PDFs.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Folder containing PDFs")
    parser.add_argument("--chunk-size", type=int, default=3, help="Sentences per chunk")
    parser.add_argument("--overlap", type=int, default=1, help="Sentences shared between chunks")
    parser.add_argument("--top-k", type=int, default=3, help="Chunks retrieved per question")
    parser.add_argument("--show-context", action="store_true", help="Print retrieved chunks")
    return parser.parse_args()


def ensure_nltk_data() -> None:
    """Download the NLTK sentence tokenizer data if it is missing."""
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def main() -> None:
    args = parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        sys.exit("GEMINI_API_KEY is not configured. Add it to your .env file.")
    llm_model = os.getenv("GEMINI_MODEL", DEFAULT_LLM_MODEL)

    ensure_nltk_data()
    client = genai.Client(api_key=api_key)

    # Build the knowledge base once at start-up
    pages = load_pdfs(args.data_dir)
    chunks = create_chunks(pages, args.chunk_size, args.overlap)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    index = build_index(chunks, embedding_model)

    print("\nPolicy assistant ready. Type 'exit' to quit.")
    print("Examples: 'How many paid leave days do employees get?' | 'What is the remote work policy?'")

    # Ask questions in a loop
    while True:
        query = input("\nAsk your question: ").strip()
        if query.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break
        if not query:
            continue

        results = retrieve(query, embedding_model, index, chunks, args.top_k)

        if args.show_context:
            for rank, (chunk, distance) in enumerate(results, start=1):
                print(f"\nResult {rank} | {chunk.source} | page {chunk.page} | distance {distance:.4f}")
                print(chunk.text)

        try:
            answer = generate_answer(client, llm_model, build_prompt(query, results))
        except Exception as error:  # network / API errors
            logger.error("Generation failed: %s", error)
            continue

        print("\n==============================")
        print("FINAL ANSWER")
        print("==============================")
        print(answer)


if __name__ == "__main__":
    main()

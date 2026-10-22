# RAG Policy Assistant — Retrieval-Augmented Generation over PDF Documents

## 1. Project Overview

The **RAG Policy Assistant** is a Python-based application that allows users to ask natural-language questions about company policies and retrieve answers from PDF documents.

Instead of relying entirely on the language model's existing knowledge, the application searches the provided PDF documents, retrieves relevant text, and passes that information to Google's Gemini model to generate an answer.

The application also includes source document names and page numbers in the retrieved context, helping users identify where the relevant information came from.

### Example

**Input:**
```text
How many paid leave days do employees get?
```

**Output:**
```text
Employees are eligible for the paid leave specified
in the company policy.

Source: Employee_Policy.pdf
Page: 5
```

*Note: The actual answer depends on the content of the PDF documents.*

## 2. Project Objectives

- Extract text from company policy PDF documents.
- Clean and divide extracted text into smaller, manageable chunks.
- Convert text into numerical vector representations using Sentence Transformers.
- Store the vectors in a FAISS similarity-search index.
- Retrieve the most relevant chunks for a user's question.
- Use Google Gemini to generate answers based only on the retrieved context.
- Preserve source document names and page numbers for traceability.
- Provide an interactive question-and-answer interface through the terminal.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyPDF2 | Extract text from PDF documents |
| NLTK | Split extracted text into sentences |
| Sentence Transformers | Convert text into numerical embeddings |
| all-MiniLM-L6-v2 | Pre-trained sentence embedding model |
| FAISS | Perform vector similarity searches |
| NumPy | Handle numerical embedding arrays |
| Google Gemini API | Generate natural-language answers |
| python-dotenv | Load environment variables from a `.env` file |
| argparse | Accept command-line configuration options |
| logging | Display application progress and errors |
| dataclasses | Organize page and chunk information |

## 4. Project Architecture

The application follows the Retrieval-Augmented Generation (RAG) architecture.

```text
                 PDF DOCUMENTS
                       |
                       v
                Extract PDF Text
                  (PyPDF2)
                       |
                       v
                  Clean Text
                       |
                       v
              Sentence Tokenization
                    (NLTK)
                       |
                       v
               Create Text Chunks
                With Overlap
                       |
                       v
              Generate Embeddings
              (Sentence Transformers)
                       |
                       v
                Build FAISS Index
                       |
                       |
USER QUESTION --------+
       |
       v
Generate Query Embedding
       |
       v
FAISS Similarity Search
       |
       v
Retrieve Relevant Chunks
       |
       v
Build Context-Aware Prompt
       |
       v
Google Gemini API
       |
       v
Generate Grounded Answer
       |
       v
Display Answer in Terminal
With Retrieved Source Context
```

The PDF processing and FAISS indexing stages run when the application starts. For each question, the application generates a query embedding, retrieves relevant chunks, and sends the resulting context to Gemini.

## 5. How Retrieval-Augmented Generation Works

Retrieval-Augmented Generation combines two main processes:

### A. Retrieval

The application searches the provided documents to find text that is semantically similar to the user's question.

For example, if the user asks about annual leave, the retrieval process attempts to find chunks discussing annual leave, leave eligibility, or related policy information.

### B. Generation

The retrieved text is passed to the Gemini language model along with the user's question.

Gemini generates a readable answer using the supplied context rather than being instructed to answer from general knowledge.

### Why use RAG?

A language model may not know the contents of a company's private policy documents. RAG makes those documents available to the model at question-answering time without requiring the model to be trained specifically on them.

It can also reduce unsupported answers when the prompt explicitly restricts the model to the retrieved context. However, retrieval quality and prompt instructions do not guarantee that every answer will be correct.

---

## 6. Detailed Code Explanation

### 6.1. Imports and Dependencies

The application imports the libraries required for PDF processing, text chunking, embeddings, similarity search, API communication, and command-line execution.

```python
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
```

**Explanation:**

- `argparse`: Reads optional settings such as the PDF directory and number of retrieved chunks.
- `logging`: Reports the application's progress and errors.
- `os`: Reads environment variables, including the Gemini API key.
- `sys`: Allows the application to exit with a useful error message.
- `dataclass`: Defines structured objects for PDF pages and text chunks.
- `Path`: Provides convenient and reliable filesystem path handling.
- `faiss`: Stores vectors and searches for similar embeddings.
- `nltk`: Supports natural-language text processing.
- `numpy`: Converts embedding arrays to the numerical format expected by FAISS.
- `load_dotenv`: Loads configuration values from a `.env` file.
- `genai`: Communicates with the Google Gemini API.
- `sent_tokenize`: Splits text into individual sentences.
- `PdfReader`: Reads PDF documents and extracts their text.
- `SentenceTransformer`: Loads the pre-trained embedding model.

### 6.2. Configuration

```python
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "gemini-3.5-flash-lite"
NOT_FOUND_MESSAGE = (
    "The information is not available in the provided documents."
)
```

These constants define the models and fallback response used by the application.

**Embedding model — `all-MiniLM-L6-v2`**

This pre-trained Sentence Transformer converts sentences and paragraphs into numerical vectors called embeddings.

These vectors represent semantic information, allowing the application to compare the meaning of a question with the meaning of document chunks.

**Language model — Gemini**

The Gemini model receives the question and retrieved document context, then generates the final response.

The application uses the configured default model unless a different model is provided through the `GEMINI_MODEL` environment variable. The configured model name must be supported by the API and your account.

**Fallback response**

`NOT_FOUND_MESSAGE` provides a standard response when the model returns no usable answer text.

The current implementation does not independently verify that the retrieved documents contain the answer before calling Gemini.

### 6.3. Logging

```python
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger("rag_policy_assistant")
```

Logging provides visibility into the application's execution.

For example, it reports when PDF files are loaded, how many chunks are created, and when the FAISS index is ready.

This makes the application easier to debug and monitor.

### 6.4. Data Structures Using Dataclasses

The application defines two dataclasses to organize the extracted information.

```python
@dataclass
class Page:
    source: str
    page: int
    text: str


@dataclass
class Chunk:
    source: str
    page: int
    text: str
```

**Page**

Represents the text extracted from one PDF page.

- `source`: The PDF filename.
- `page`: The page number, starting from 1.
- `text`: The extracted page content.

**Chunk**

Represents a smaller portion of a page that will be embedded and searched.

- `source`: The original PDF filename.
- `page`: The original page number.
- `text`: The text contained in the chunk.

Keeping this metadata allows the application to associate retrieved text with its original document and page.

---

## 7. PDF Loading and Text Extraction

### Function: `load_pdfs()`

```python
def load_pdfs(data_dir: Path) -> list[Page]:
    pdf_files = sorted(data_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {data_dir.resolve()}"
        )

    pages: list[Page] = []

    for pdf_path in pdf_files:
        logger.info("Loading: %s", pdf_path.name)
        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(
            reader.pages, start=1
        ):
            text = page.extract_text()

            if text and text.strip():
                pages.append(
                    Page(pdf_path.name, page_number, text)
                )

    return pages
```

**Purpose:** Load PDF documents and extract their text page by page.

**Step-by-step workflow:**

1. Search the specified directory for PDF files.
2. Sort the filenames to provide a predictable processing order.
3. Raise an error if no PDF files are found.
4. Open each PDF using `PdfReader`.
5. Iterate through its pages.
6. Extract text from each page.
7. Ignore pages where no meaningful text is extracted.
8. Store the filename, page number, and extracted text in a `Page` object.

**Why process pages separately?**

Page-level processing preserves the location of the extracted information. This is important when a user needs to identify the page containing a policy statement.

**Important limitation:** `PyPDF2` extracts text from PDFs that contain accessible text. Scanned documents may require OCR (Optical Character Recognition) before their contents can be searched.

---

## 8. Text Cleaning and Chunking

### 8.1. Text Cleaning

#### Function: `clean_text()`

```python
def clean_text(text: str) -> str:
    return " ".join(
        line.strip()
        for line in text.splitlines()
        if line.strip()
    )
```

PDF extraction can produce unnecessary line breaks and blank lines.

This function removes blank lines, strips surrounding whitespace from individual lines, and joins the remaining lines into a single string.

**Example:**

Input:

```text
Employees are eligible for annual leave.

The leave policy applies to all eligible employees.
```

Output:

```text
Employees are eligible for annual leave. The leave policy applies to all eligible employees.
```

This creates cleaner text for sentence tokenization and embedding generation.

### 8.2. Sentence-Based Chunking

#### Function: `create_chunks()`

```python
def create_chunks(
    pages: list[Page],
    chunk_size: int = 3,
    overlap: int = 1
) -> list[Chunk]:
    ...
```

**Purpose:** Divide each PDF page into smaller text segments.

Large documents are not always suitable for direct retrieval as a single block. Chunking creates smaller units that can be independently embedded and retrieved.

The application uses NLTK's `sent_tokenize()` to split each cleaned page into sentences.

The `chunk_size` parameter specifies the maximum number of sentences in a chunk, while `overlap` specifies how many sentences adjacent chunks share.

**Example:**

Suppose a page contains six sentences:

```text
S0, S1, S2, S3, S4, S5
```

With `chunk_size=3` and `overlap=1`, the chunks are:

```text
Chunk 1: S0, S1, S2
Chunk 2: S2, S3, S4
Chunk 3: S4, S5
```

The application calculates the step size as:

```python
step = chunk_size - overlap
```

For these settings:

```text
step = 3 - 1 = 2
```

Each new chunk starts two sentences after the previous chunk's starting position.

**Why use overlap?**

A policy statement may span multiple sentences. Overlap helps preserve nearby information when a chunk boundary separates related sentences.

The function also validates the configuration:

- `chunk_size` must be at least 1.
- `overlap` must be non-negative.
- `overlap` must be smaller than `chunk_size`.

Each chunk retains the source filename and page number.

**Important detail:** Chunking is performed separately for each page, so a chunk does not combine text from two different pages.

---

## 9. Embedding Generation and FAISS Indexing

### Function: `build_index()`

```python
def build_index(
    chunks: list[Chunk],
    model: SentenceTransformer
) -> faiss.IndexFlatL2:
    embeddings = model.encode(
        [chunk.text for chunk in chunks],
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index
```

**Purpose:** Convert all text chunks into embeddings and store them in a FAISS index.

### Step 1: Generate embeddings

```python
embeddings = model.encode(
    [chunk.text for chunk in chunks],
    convert_to_numpy=True,
    show_progress_bar=True,
).astype("float32")
```

The model converts each chunk into a fixed-length numerical vector.

The `all-MiniLM-L6-v2` model normally produces a 384-dimensional embedding for each input.

For example:

```text
Chunk 1 → [0.12, -0.04, 0.31, ...]
Chunk 2 → [0.08,  0.17, 0.25, ...]
Chunk 3 → [-0.11, 0.22, 0.09, ...]
```

These numbers encode semantic features learned by the model.

`convert_to_numpy=True` returns NumPy arrays, while `.astype("float32")` converts the embeddings into 32-bit floating-point numbers compatible with FAISS.

### Step 2: Create the FAISS index

```python
index = faiss.IndexFlatL2(embeddings.shape[1])
```

`IndexFlatL2` performs exact nearest-neighbor searches using squared Euclidean distance.

The index dimension is taken from the embedding vectors, ensuring the index matches the selected embedding model's output size.

### Step 3: Add embeddings

```python
index.add(embeddings)
```

All document embeddings are added to the index.

Each vector corresponds to a chunk in the `chunks` list. FAISS returns the positions of matching vectors, which the application uses to retrieve the corresponding text and metadata.

**Why FAISS?**

FAISS provides an efficient and convenient way to search large collections of numerical vectors. This implementation uses an exact flat index, which is simple to understand and suitable for small to medium-sized document collections.

**Important:** The FAISS index is built in memory when the application starts. This implementation does not save the index or embeddings to disk, so they must be regenerated on the next run.

---

## 10. Query Embedding and Similarity Search

### Function: `retrieve()`

```python
def retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatL2,
    chunks: list[Chunk],
    top_k: int = 3
) -> list[tuple[Chunk, float]]:
    query_vec = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    distances, indices = index.search(
        query_vec,
        min(top_k, len(chunks))
    )

    return [
        (chunks[i], float(d))
        for i, d in zip(indices[0], distances[0])
        if i != -1
    ]
```

**Purpose:** Retrieve the document chunks most relevant to the user's question.

### Step 1: Embed the question

The user's question is converted into an embedding using the same Sentence Transformer used for the document chunks.

Using the same model ensures the query and document vectors share the same embedding space.

### Step 2: Search FAISS

```python
distances, indices = index.search(
    query_vec,
    min(top_k, len(chunks))
)
```

The index searches for the nearest vectors to the question embedding.

`top_k` controls how many chunks are retrieved.

For example, with `top_k=3`, the application retrieves up to three matching chunks.

The `min()` function prevents the search from requesting more results than the total number of chunks.

### Step 3: Match vectors to document chunks

FAISS returns:

- `indices`: Positions of the nearest vectors in the index.
- `distances`: Squared Euclidean distances between the query and matching vectors.

The application uses the returned indices to locate the corresponding `Chunk` objects.

Each result contains the chunk and its distance.

**How are distances interpreted?**

For this index, smaller distances indicate vectors that are closer under the squared Euclidean distance metric.

The distances are not probabilities or confidence percentages.

The implementation also filters out `-1`, which FAISS uses to indicate a missing result.

---

## 11. Context Construction and Prompt Engineering

### Function: `build_prompt()`

```python
def build_prompt(
    query: str,
    results: list[tuple[Chunk, float]]
) -> str:
    context = "\n".join(
        f"Source: {chunk.source}\n"
        f"Page: {chunk.page}\n\n"
        f"Content:\n{chunk.text}\n"
        for chunk, _ in results
    )

    return f"""You are a company policy assistant.

Answer the user's question using ONLY the context provided below.

CONTEXT
{context}

QUESTION
{query}

RULES
1. Use only the provided context.
2. Do not invent information.
3. Do not use outside knowledge.
4. If the answer is not available in the context, say:
   "{NOT_FOUND_MESSAGE}"
5. Keep the answer clear and concise.
6. Mention the source document and page when possible.
"""
```

**Purpose:** Combine the retrieved document chunks and the user's question into a prompt for Gemini.

### Step 1: Build the context

Each retrieved chunk is formatted with its source filename, page number, and text.

For example:

```text
Source: Employee_Policy.pdf
Page: 5

Content:
Employees are eligible for annual leave according
to the company's leave policy.
```

This metadata allows Gemini to see where the information originated.

### Step 2: Add the user's question

The question is placed after the retrieved context so the model can answer it using the supplied information.

### Step 3: Apply instructions

The prompt instructs Gemini to:

- Use only the retrieved context.
- Avoid inventing unsupported information.
- Avoid relying on outside knowledge.
- Return a predefined message when the answer is unavailable.
- Keep answers concise.
- Mention the source and page when possible.

**Why is prompt engineering important?**

Retrieval provides relevant information, but the language model still needs instructions about how to use that information.

The context-only prompt encourages grounded responses and reduces the risk of unsupported answers.

**Limitation:** These instructions are not a guarantee against hallucinations. The retrieved text may be incomplete or irrelevant, and the model may still misinterpret it. The application also does not independently validate the citations generated in the final answer.

---

## 12. Answer Generation Using Google Gemini

### Function: `generate_answer()`

```python
def generate_answer(
    client: genai.Client,
    model_name: str,
    prompt: str
) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    return (
        (response.text or "").strip()
        or NOT_FOUND_MESSAGE
    )
```

**Purpose:** Send the constructed prompt to Gemini and return the generated answer.

### Step-by-step workflow

1. Receive the Gemini API client, model name, and complete prompt.
2. Send the prompt to the configured model using `generate_content()`.
3. Extract the generated response text.
4. Remove unnecessary whitespace using `.strip()`.
5. Return the fallback message if no usable text is returned.

This stage converts retrieved document information into a readable natural-language answer.

**Security note:** The API key is loaded from an environment variable instead of being hardcoded into the Python source code.

---

## 13. Command-Line Arguments

### Function: `parse_args()`

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RAG assistant for company policy PDFs."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Folder containing PDFs"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=3,
        help="Sentences per chunk"
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=1,
        help="Sentences shared between chunks"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Chunks retrieved per question"
    )

    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved chunks"
    )

    return parser.parse_args()
```

This function allows users to change application settings without editing the source code.

| Argument | Default | Purpose |
|---|---|---|
| `--data-dir` | `data` | Directory containing PDF documents |
| `--chunk-size` | `3` | Maximum sentences per chunk |
| `--overlap` | `1` | Number of shared sentences between chunks |
| `--top-k` | `3` | Number of chunks to retrieve |
| `--show-context` | Disabled | Display retrieved chunks and distances |

For example:

```bash
python rag_policy_assistant.py --data-dir data --top-k 3 --chunk-size 3
```

To inspect the retrieved text:

```bash
python rag_policy_assistant.py --show-context
```

The default values are suitable starting points, but chunk size, overlap, and `top_k` should be tuned to the documents and the questions being asked.

---

## 14. NLTK Resource Management

### Function: `ensure_nltk_data()`

```python
def ensure_nltk_data() -> None:
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)
```

NLTK requires tokenizer data to split text into sentences.

This function checks whether the required resources are available locally.

If a resource is missing, the application attempts to download it automatically.

This reduces the chance of the application failing because the sentence tokenizer has not been downloaded.

**Note:** An internet connection may be required on the first run if the resources are missing. In restricted environments, the resources must be installed beforehand.

---

## 15. Main Application Workflow

### Function: `main()`

The `main()` function coordinates all components and manages the interactive question-and-answer loop.

### Step 1: Read configuration

```python
args = parse_args()
```

Reads the command-line arguments provided by the user.

### Step 2: Load environment variables

```python
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
```

Loads the `.env` file and retrieves the Gemini API key.

The application exits if the key is missing:

```python
if not api_key:
    sys.exit(
        "GEMINI_API_KEY is not configured. "
        "Add it to your .env file."
    )
```

The Gemini model can also be configured through the `GEMINI_MODEL` environment variable.

### Step 3: Initialize the Gemini client

```python
client = genai.Client(api_key=api_key)
```

Creates the client used to communicate with Google's Gemini API.

### Step 4: Load and process PDF documents

```python
pages = load_pdfs(args.data_dir)

chunks = create_chunks(
    pages,
    args.chunk_size,
    args.overlap
)
```

The application extracts page text, cleans it, and divides it into overlapping chunks.

### Step 5: Load the embedding model

```python
embedding_model = SentenceTransformer(
    EMBEDDING_MODEL_NAME
)
```

Loads `all-MiniLM-L6-v2` for embedding generation.

The model may need to be downloaded the first time it is used if it is not already cached locally.

### Step 6: Build the FAISS index

```python
index = build_index(chunks, embedding_model)
```

Embeds the document chunks and creates the searchable vector index.

This process runs once at startup, not for every question.

### Step 7: Start the interactive question loop

```python
while True:
    query = input("\nAsk your question: ").strip()
```

The application waits for the user to enter a question.

The user can type `exit`, `quit`, or `q` to terminate the program.

Empty questions are ignored.

### Step 8: Retrieve relevant chunks

```python
results = retrieve(
    query,
    embedding_model,
    index,
    chunks,
    args.top_k
)
```

The question is embedded and FAISS retrieves the nearest document chunks.

### Step 9: Optionally display the retrieved context

When `--show-context` is enabled, the application prints the retrieved chunks along with their source filename, page number, and distance.

This is useful for understanding why a particular answer may have been generated.

### Step 10: Generate the answer

```python
answer = generate_answer(
    client,
    llm_model,
    build_prompt(query, results)
)
```

The retrieved chunks are converted into a prompt and sent to Gemini.

If the API call raises an exception, the application logs the error and continues waiting for the next question.

### Step 11: Display the final answer

The generated response is printed in the terminal under the `FINAL ANSWER` heading.

This completes one question-and-answer cycle.

---

## 16. Application Entry Point

```python
if __name__ == "__main__":
    main()
```

This statement ensures that `main()` is executed when the Python file is run directly.

It also allows the file to be imported as a module without automatically starting the interactive application.

---

## 17. Project Folder Structure

Organize the repository as follows:

```text
rag-policy-assistant/
│
├── data/
│   ├── employee_policy.pdf
│   ├── leave_policy.pdf
│   └── remote_work_policy.pdf
│
├── rag_policy_assistant.py
├── requirements.txt
├── .env
├── .gitignore
└── README.md
```

**File descriptions:**

- `data/`: Contains the PDF documents used as the knowledge base.
- `rag_policy_assistant.py`: Contains the complete RAG implementation.
- `requirements.txt`: Lists the required Python packages.
- `.env`: Stores environment variables, including the API key.
- `.gitignore`: Prevents sensitive files and unnecessary generated files from being committed.
- `README.md`: Documents the project and explains how to install and run it.

The application searches for PDF files directly inside the configured directory. The default directory is `data/`; PDFs stored only in nested subdirectories will not be discovered by the current implementation.

### Example `.gitignore`

```gitignore
.env
.venv/
__pycache__/
*.py[cod]
```

Do not commit your actual `.env` file or API key to GitHub.

---

## 18. Installation and Setup

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-policy-assistant.git
cd rag-policy-assistant
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

A virtual environment keeps this project's dependencies separate from other Python projects.

### Step 3: Install dependencies

Create a `requirements.txt` file containing:

```text
faiss-cpu
nltk
numpy
python-dotenv
google-genai
PyPDF2
sentence-transformers
```

Install the packages:

```bash
python -m pip install -r requirements.txt
```

### Step 4: Configure the Gemini API key

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your_actual_api_key
GEMINI_MODEL=your_supported_gemini_model
```

Replace the placeholder values with your actual API key and a model name supported by your Gemini API account.

If you want to use the code's default model, you can omit `GEMINI_MODEL`.

### Step 5: Add PDF documents

Place your company policy PDFs inside the `data/` directory.

For example:

```text
data/
├── employee_policy.pdf
└── leave_policy.pdf
```

Use documents that you have permission to process, and avoid publishing confidential company documents in a public repository.

### Step 6: Run the application

```bash
python rag_policy_assistant.py
```

---

## 19. Running the Application

### Default execution

```bash
python rag_policy_assistant.py
```

Uses the default `data/` directory, a chunk size of 3 sentences, an overlap of 1 sentence, and retrieves up to 3 chunks per question.

### Customize the PDF directory

```bash
python rag_policy_assistant.py --data-dir company_documents
```

### Change the chunk size

```bash
python rag_policy_assistant.py --chunk-size 5 --overlap 1
```

This creates chunks of up to five sentences, with one shared sentence between adjacent chunks.

### Retrieve more context

```bash
python rag_policy_assistant.py --top-k 5
```

This retrieves up to five chunks per question.

Increasing `top_k` can provide more context, but it can also introduce irrelevant information or increase the amount of text sent to the language model.

### Display retrieved chunks

```bash
python rag_policy_assistant.py --show-context
```

This displays the retrieved chunks and their similarity distances before printing the final answer.

---

## 20. Sample Execution

```text
INFO | Loading: employee_policy.pdf
INFO | Loading: leave_policy.pdf
INFO | Loaded 2 PDF(s), 24 page(s) with text
INFO | Created 58 chunk(s)
INFO | FAISS index ready: 58 vectors, dimension 384

Policy assistant ready. Type 'exit' to quit.

Ask your question: What is the annual leave policy?

==============================
FINAL ANSWER
==============================

The answer is generated using the relevant retrieved
text from the policy documents.

Source: leave_policy.pdf
Page: 5
```

*The numbers and answer above are illustrative examples, not actual output from a specific document collection.*

The source citation shown in the sample is an expected response format. In the current implementation, Gemini is instructed to mention the source and page, but the application does not programmatically enforce or validate the final citation.

---

## 21. Error Handling and Limitations

The application includes basic error handling for common problems.

| Situation | Current behavior |
|---|---|
| No PDFs found | Raises `FileNotFoundError` |
| Missing Gemini API key | Exits with a configuration message |
| Missing NLTK resources | Attempts to download the resources |
| Invalid chunk configuration | Raises `ValueError` |
| Gemini API error | Logs the error and continues the question loop |
| PDF page without extracted text | Skips that page |
| Empty user question | Ignores the input |
| User enters `exit`, `quit`, or `q` | Terminates the application |

### Additional limitations

1. **Scanned PDFs:** OCR is not implemented.
2. **Retrieval relevance:** The application always attempts retrieval, even when all returned chunks are unrelated to the question.
3. **Citation validation:** Source citations in the generated answer are not automatically checked.
4. **Persistent storage:** The FAISS index and embeddings are rebuilt at startup.
5. **PDF coverage:** Only PDFs directly inside the selected directory are loaded.
6. **Answer grounding:** Prompt instructions encourage grounded answers but cannot guarantee factual correctness.
7. **API dependency:** Answer generation requires access to the Gemini API and a supported model.
8. **Security:** The application does not implement user authentication, access controls, or document-level permissions.
9. **Empty chunk collections:** The application does not explicitly handle a case where PDFs exist but no usable text chunks are produced.
10. **Resource and API usage:** Large document collections can increase startup time, memory usage, and generation costs.

---

## 22. Potential Improvements

The project can be extended in several directions:

- **Persistent FAISS indexing:** Save and reload embeddings and indexes instead of rebuilding them at every startup.
- **OCR support:** Process scanned PDFs using an OCR engine.
- **Hybrid retrieval:** Combine vector similarity search with keyword-based search.
- **Metadata filtering:** Filter retrieved chunks by document type, department, or other metadata.
- **Reranking:** Reorder retrieved chunks using a dedicated relevance model.
- **Citation enforcement:** Generate citations directly from retrieved metadata and validate them before displaying the answer.
- **Relevance thresholds:** Return the fallback response when no retrieved chunk is sufficiently relevant.
- **Improved chunking:** Use token-based or structure-aware chunking for long policies, tables, and section headings.
- **Conversation interface:** Add a Streamlit or FastAPI frontend.
- **Automated evaluation:** Measure retrieval quality, answer correctness, and citation accuracy.
- **Access control:** Enforce document permissions before retrieving or sending any content to the language model.

These improvements would make the application more reliable, scalable, and suitable for real-world document-question-answering systems.

---

## 23. Key Concepts Learned

This project demonstrates practical implementation of several concepts relevant to Generative AI and NLP engineering.

### Python

- Functions and modular program structure
- Dataclasses and type hints
- Exception handling
- File and directory management
- Environment variables
- Command-line arguments
- Logging

### Natural Language Processing

- Text extraction and cleaning
- Sentence tokenization
- Text chunking and overlap
- Semantic embeddings
- Query-to-document similarity

### Vector Search

- Vector representations
- Euclidean distance
- Nearest-neighbor retrieval
- FAISS index construction and search
- Mapping vector results back to document metadata

### Generative AI

- Retrieval-Augmented Generation
- Prompt engineering
- Context-grounded answer generation
- Gemini API integration
- Source attribution and hallucination mitigation

---

## 24. Conclusion

The RAG Policy Assistant demonstrates how a collection of PDF documents can be transformed into a searchable knowledge base and connected to a large language model.

By combining PDF text extraction, sentence-based chunking, Sentence Transformer embeddings, FAISS similarity search, and Gemini generation, the application creates an end-to-end document-question-answering pipeline.

The project provides a practical foundation for building more advanced RAG applications, including internal knowledge assistants, employee policy assistants, technical documentation search systems, and document-based support tools.

**Core workflow:**

`PDF Extraction → Text Chunking → Embeddings → FAISS Retrieval → Context Construction → Gemini Generation → Answer`

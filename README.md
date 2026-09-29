# 📚 Intelligent PDF Knowledge Assistant using RAG

A multi-document PDF question-answering and comparison system built using **Retrieval-Augmented Generation (RAG)**.

The application allows users to upload multiple PDF documents, ask questions about their contents, compare information across documents, and view the document/page sources used to generate each response.

## 🚀 Features

### 📄 Multi-PDF Upload
- Upload multiple PDF documents at the same time.
- Extract text from each document.
- Preserve document names and page numbers.

### 🔎 Semantic Search
- Sentence-aware document chunking with overlap.
- Gemini embeddings for semantic representation.
- FAISS for vector similarity search.

### 💬 Ask Questions
Ask questions about the uploaded PDF collection. The system retrieves relevant chunks, sends grounded context to Gemini, and displays the answer with supporting sources.

### 📊 Compare Documents
Compare information across multiple PDFs using:
- Overall explanation
- Comparison table
- Similarities
- Differences
- Supporting document/page references

### 🛡️ Grounded Answers
The generation prompt instructs Gemini to use only retrieved document context. When the required information is unavailable, the assistant can state that there is not enough information rather than inventing an answer.

### 📑 Source Citations
Responses display the PDF name and page numbers associated with retrieved information.

## 🏗️ Architecture

```text
PDF Upload
    ↓
Text Extraction (PyMuPDF)
    ↓
Sentence-Aware Chunking
    ↓
Gemini Embeddings
    ↓
FAISS Vector Index
    ↓
Semantic Retrieval
    ↓
┌─────────────────────────────┐
│ Ask Questions               │
│ Compare Documents           │
└──────────────┬──────────────┘
               ↓
       Gemini Generation
               ↓
    Grounded Answer + Sources
```

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| PyMuPDF | PDF text extraction |
| Google Gemini API | Embeddings and answer generation |
| Gemini `gemini-embedding-001` | Document/query embeddings |
| FAISS | Vector similarity search |
| NumPy | Vector processing |
| python-dotenv | Environment variable management |
| Git/GitHub | Version control |

## 📁 Project Structure

```text
pdf-knowledge-assistant/
│
├── src/
│   ├── pdf_processor.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   └── rag.py
│
├── data/
│   ├── uploads/
│   └── index/
│
├── app.py
├── rag_evaluation.py
├── evaluation_questions.json
├── RAG_Evaluation_Report.md
├── requirements.txt
├── .gitignore
├── .env
└── README.md
```

> `.env`, uploaded PDFs, and generated indexes should remain local and should not be committed to GitHub.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd pdf-knowledge-assistant
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.env\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 🔐 API Key Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Never commit the `.env` file or your API key to GitHub.

Recommended `.gitignore` entries:

```text
venv/
.env
__pycache__/
*.pyc
data/uploads/
data/index/
```

For deployment, configure `GEMINI_API_KEY` through the hosting platform's secrets/environment variables.

## ▶️ Run the Application

```bash
streamlit run app.py
```

The application will normally be available at:

```text
http://localhost:8501
```

## 📖 Usage

### Step 1 — Upload PDFs

Use the sidebar to upload one or more PDF documents.

### Step 2 — Process PDFs

Click **⚙️ Process PDFs**.

The application extracts text, creates chunks, generates embeddings, and builds the FAISS vector index.

### Step 3 — Ask Questions

Select **💬 Ask Questions**.

Example:

```text
What problems can occur during backpropagation through time in a standard RNN?
```

The system retrieves relevant chunks and generates a grounded answer with source pages.

### Step 4 — Compare Documents

Select **📊 Compare Documents** and upload at least two PDFs.

Example:

```text
Compare how RNN and LSTM handle long-term dependencies according to the uploaded documents.
```

The system produces an overall explanation, comparison table, similarities, differences, and source references.

## 🧪 Evaluation

The project includes a small evaluation set covering:
- Single-document retrieval
- Cross-document retrieval
- Source grounding
- Unsupported-question handling

Run:

```bash
python rag_evaluation.py
```

Top-k retrieval setting:

```text
5
```

### Results

| Metric | Result |
|---|---:|
| Single-document retrieval | **100%** |
| Cross-document retrieval | **100%** |
| Answerable retrieval tests | **9/9** |
| Cross-document tests | **1/1** |
| Unsupported-question test | **PASS** |

Manual generation tests also demonstrated:
- Correct identification of RNN vanishing/exploding gradient problems.
- Successful RNN vs. LSTM comparison using both PDFs.
- Refusal to invent an unsupported exact learning rate.

> These results come from a small evaluation set and are not a general guarantee of 100% RAG accuracy.

See [`RAG_Evaluation_Report.md`](RAG_Evaluation_Report.md) for details.

## 🔄 RAG Pipeline

1. **Document ingestion** — PDFs are uploaded.
2. **Text extraction** — PyMuPDF extracts page-level text.
3. **Chunking** — text is divided into sentence-aware overlapping chunks.
4. **Embedding generation** — chunks are converted to vectors using Gemini embeddings.
5. **Vector indexing** — vectors are stored in FAISS.
6. **Query retrieval** — the question is embedded and the most relevant chunks are retrieved.
7. **Generation** — Gemini generates an answer using the retrieved context.
8. **Source display** — document names and page numbers are shown with the response.

## 🎯 Project Objectives

- Build a practical document question-answering system.
- Implement the core RAG pipeline without a high-level RAG framework.
- Enable semantic search over multiple PDF documents.
- Support cross-document comparison.
- Provide document/page source references.
- Reduce unsupported or hallucinated responses.
- Evaluate retrieval and grounding behavior.

## 📌 Limitations

- Evaluation is based on a small manually prepared test set.
- PDF extraction quality depends on document layout, tables, figures, and scanned content.
- The application depends on Gemini API availability and usage limits.
- Retrieval quality can vary for poorly represented questions.
- The current system primarily targets text-based PDFs.

## 🚀 Future Improvements

- OCR support for scanned PDFs.
- Improved table and figure extraction.
- Hybrid keyword + semantic retrieval.
- Reranking of retrieved chunks.
- Persistent vector databases.
- Streaming responses.
- Conversation-aware retrieval.
- Larger automated RAG evaluation datasets.
- Authentication and user-specific document collections.
- Support for additional document formats.

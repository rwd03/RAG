# PDF RAG Chatbot

Upload PDF chapters and ask questions about them. The app searches the uploaded text, asks a local Ollama model to answer from the retrieved excerpts, and shows the answer's filename and page citations.

## Windows PowerShell setup

Install 64-bit Python 3.11 or newer and [Ollama for Windows](https://ollama.com/download/windows). Open PowerShell in this project folder.

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Open the Ollama app from the Start menu. If you prefer to start Ollama from a terminal, run `ollama serve` in a separate PowerShell window and keep it open. Run these commands after Ollama starts.

```powershell
ollama pull llama3.2:3b
ollama list
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local address printed by Streamlit. The first upload with extractable text downloads the embedding model. Once both models are downloaded, indexing, search, and inference can run locally. The generated answer and excerpts are visible in the browser, and the index is kept in memory for the current Streamlit session.

## Demo

1. Upload the five Enterprise Application Integration chapter PDFs together.
2. Ask one of the sample questions below. The answer depends on what the uploaded chapters say.
3. Check the cited filename and page, then open the excerpt panels to inspect the retrieved evidence.

- How do the chapters define Enterprise Application Integration, and what problem does it address?
- What integration approaches do the chapters describe, and how do they compare them?
- What role does messaging play in Enterprise Application Integration, according to the chapters?

## How it works

1. PyMuPDF extracts text page by page. Each chunk retains its filename and one-based PDF page number.
2. The app splits text within each page into 500-character chunks with up to 100 characters of overlap.
3. `all-MiniLM-L6-v2` embeds the chunks. FAISS ranks them by cosine similarity and returns up to three excerpts for each question.
   When a question names a chapter, such as `explain chapter 1`, retrieval is restricted to uploaded filenames containing that chapter number.
4. The local Ollama server receives the question and the excerpts. The prompt asks the model to use only those excerpts and report insufficient evidence when it cannot answer. The app displays the retrieved source filenames and pages directly from the PDF metadata.

The embedding model is cached across Streamlit reruns. Each session reuses its index while the uploads are the same. Embeddings and Ollama inference use the CPU. Scanned PDFs need a searchable text layer for extraction.

Broad explanations and summaries cover the three retrieved excerpts. Ask about a specific concept to get a more focused answer.

## Tests

Run the chapter retrieval regression tests after installing the dependencies.

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_app
```

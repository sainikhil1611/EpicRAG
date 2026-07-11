# EpicRAG

A multimodal Retrieval-Augmented Generation (RAG) system for corporate documents (e.g. annual reports). It extracts text and images from a source PDF, embeds text locally with a SentenceTransformer model and classifies/OCRs images with CLIP + EasyOCR, stores everything in a local ChromaDB vector store, and answers questions with OpenAI's `gpt-4o-mini` (including a calculator tool for financial math). Two entry points: a terminal REPL (`runner.py`) and a Streamlit web UI (`runner_st.py`).

## Quick start (devcontainer — recommended)

1. Prereqs: [Docker](https://docs.docker.com/get-docker/), [VS Code](https://code.visualstudio.com/) + the "Dev Containers" extension.
2. Open this folder in VS Code → "Reopen in Container" when prompted.
3. `cp .env.example .env` and fill in `OPENAI_API_KEY` (see table below).
4. Build the vector database (see [Regenerating the vector database](#regenerating-the-vector-database) — one-time, no key needed for this step).
5. Run: `streamlit run runner_st.py --server.headless true` → open http://localhost:8501
   (or `python runner.py` for the terminal REPL — needs `OPENAI_API_KEY`).

## Manual setup (no devcontainer)

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in OPENAI_API_KEY
```

Then run either entry point:

```bash
python runner.py                                                    # CLI, needs OPENAI_API_KEY
streamlit run runner_st.py --server.headless true                   # web UI on :8501, needs OPENAI_API_KEY
```

## Regenerating the vector database

`chroma_db/` is **not committed** (it's gitignored) — it must be built locally before either
entry point works, because it holds the embeddings ChromaDB needs to answer queries. The source
material (`data/` — extracted text chunks + the original PDF, and `pic_data/` — extracted page
images) *is* committed, so regeneration is a single command against data already in the repo:

```bash
python embeddings_processor.py
```

This reads every `data/text_*_*.txt` and `pic_data/page_*_img_*.png`, embeds text locally with
`sentence-transformers/all-mpnet-base-v2`, classifies/OCRs images with CLIP + EasyOCR, and writes
a fresh `chroma_db/` directory. **This step does not require `OPENAI_API_KEY`** — all embedding
models run locally — but it does need outbound network access on first run to download the
SentenceTransformer, CLIP, and EasyOCR models from Hugging Face (roughly 1GB+, cached afterward
in the container/host's Hugging Face cache dir). Verified: this command runs end-to-end without
any API key and produces a `chroma_db/` that `multimodal_retriever.py`'s `MultimodalRetriever()`
can load cleanly with the pinned `chromadb==0.4.22` client.

To regenerate `data/`/`pic_data/` from a different source PDF instead of reusing what's already
committed, run this first:

```bash
python pdf_processor.py --input path/to/your.pdf
```

Only `OPENAI_API_KEY` gates anything — it's required the moment you *query* the system
(`runner.py` / `runner_st.py`, both of which instantiate `RAGGenerator`), not for ingestion or
embedding. Both entry points validate the key up front and exit/error cleanly (no stack trace) if
it's missing.

## Environment variables

| Variable | Required | Purpose | Where to get it |
|---|---|---|---|
| `OPENAI_API_KEY` | Yes | Chat completion (`gpt-4o-mini`) with a calculator tool for financial math, used by `rag_generator.py` to answer queries | https://platform.openai.com/api-keys |

No other environment variables are read by any code in this repo.

## Testing

No automated test suite exists for this project. Verification performed:
- `python -m compileall -q .` — all files syntax-clean.
- Clean-venv install + import check on Python 3.12.8 for every module (`pdf_processor`,
  `embeddings_processor`, `multimodal_retriever`, `rag_generator`, `runner`, `runner_st`),
  confirming `import fitz` resolves to PyMuPDF.
- `python embeddings_processor.py` run end-to-end against the committed `data/`/`pic_data/`
  (no API key) — builds a working `chroma_db/` and confirms `MultimodalRetriever()` loads it.
- Missing-`OPENAI_API_KEY` path exercised for both `runner.py` (clean exit code 1, no traceback)
  and `runner_st.py` (`st.error` + `st.stop`, no traceback).
- Docker image build + container start on port 8501; devcontainer `up` + `exec compileall`.
- **Not verified**: an actual OpenAI-backed query/answer round trip — no live API key was used
  in this environment. The `OPENAI_API_KEY` validation path (missing-key → clean error) is
  verified; the happy path (key present → real answer) is not.

## Deployment

Build the production image: `docker build -t epicrag .`
Run it: `docker run --env-file .env -p 8501:8501 -v $(pwd)/chroma_db:/app/chroma_db epicrag`
(the volume mount persists the regenerated vector DB across container restarts — without it,
run `python embeddings_processor.py` again inside the container after each fresh start).
This container runs on any host (Fly.io, Render, a VPS with Docker) — see
[DEPLOYMENT.md](../DEPLOYMENT.md) in the Projects root for the full walkthrough.

## Known limitations

- `chroma_db/` must be regenerated locally (see above) — it is not shipped in the image or repo.
- The vector DB is on-disk inside the container by default; without a mounted volume it's lost on
  container restart and must be rebuilt via `python embeddings_processor.py`.
- First run of ingestion/embedding needs outbound network access to download ~1GB+ of ML models
  (SentenceTransformer, CLIP, EasyOCR) from Hugging Face; subsequent runs use the local cache.
- CLIP/EasyOCR/SentenceTransformer run on CPU unless CUDA is available — expect ingestion to be
  slow (OCR is the bottleneck) on a machine/container without a GPU.
- The local (untracked, gitignored) `.env` on the original dev machine contains five additional
  API keys (`MISTRAL_API_KEY`, `GOOGLE_API_KEY`, `HUGGINGFACE_API_KEY`, `SERPER_API_KEY`,
  `GROQ_API_KEY`) that are leftovers from a larger, since-stripped-down project — no code in this
  repo reads them, so they are intentionally omitted from `.env.example`.
- No automated test suite exists; see Testing above for what was manually verified.

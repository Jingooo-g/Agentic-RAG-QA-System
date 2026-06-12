## Project Structure

```
Agentic-RAG-QA-System/
├── app.py                      # Streamlit Web application main file (UI entry point)
├── requirements.txt            # Python dependencies list
├── README.md
└── backend/
    ├── __init__.py
    ├── data/                   # All data files centralized here
    │   ├── collection.jsonl    # Raw document corpus (Used for BM25 index)
    │   ├── bge_vector_db.jsonl # BGE Vector Database (Dense Index)
    │   ├── validation.jsonl    # Validation set queries (Gold)
    │   └── test.jsonl          # Test set queries
    ├── retrieval.py            # BGE Index Builder Script / Includes BM25 Retriever Class
    ├── generation.py           # RAG Pipeline Manager (Agent, Hybrid Dispatcher, Generation Core)
    ├── eval_hotpotqa.py        # Q&A evaluation script (EM/F1 Score)
    └── eval_retrieval.py       # Retrieval evaluation script (nDCG@10 Score)
```

## Environment Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Main dependencies include:
- `streamlit` - Web interface framework for the interactive application.
- `torch` & `transformers` - **Dense Encoder Core**: Core libraries used for loading and running the BGE encoder model for semantic retrieval.
- `rank-bm25` & `nltk` - **Sparse Retrieval**: Libraries used for the BM25 algorithm implementation and text tokenization.
- `scikit-learn` - **Vector Search Utility**: Used for fast Cosine Similarity calculation in the dense retrieval process.
- `huggingface-hub` & `datasets` - **Data Management**: Used for managing model caching, data downloads, and processing Hugging Face datasets.
- `requests` & `tqdm` - **Infrastructure**: Used for external LLM API communication and displaying process progress bars.

### 2. Configure API Key

Before using the generation functionality, you need to configure your API Key in `backend/pipeline.py` and `scripts/run_rag.py`:

```python
API_KEY = "your-api-key-here"
```

This project uses SiliconFlow API, which supports Qwen series models.

## Getting Started

### Launch Web Interactive Interface

Launch the Streamlit Web application for a user-friendly interactive interface:

```bash
streamlit run app.py
```

After launching, the browser will automatically open (default address: `http://localhost:8501`). You can:

- Select retrieval method (BM25 or Hybrid) in the sidebar
- Choose generation model (Qwen2.5-7B-Instruct or Qwen2.5-1.5B-Instruct)
- Input questions for real-time Q&A
- View retrieved documents and reasoning process


### Batch Processing Workflow

```bash
# Step 1: Run retrieval
python backend/retrieval.py

# Step 2: Run generation
python backend/generation.py

# Step 3: Evaluate results
python backend/eval_hotpotqa.py \
    --gold data/validation.jsonl \
    --pred val_prediction.jsonl
```


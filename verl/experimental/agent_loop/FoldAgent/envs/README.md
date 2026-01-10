# Search Server

A high-throughput semantic search server that uses embedding-based retrieval over the BrowseComp-Plus corpus. This server must be running for the local search environment to function.

## Requirements

### Hardware
- NVIDIA GPU(s) with CUDA support (designed for multi-GPU setups)
- Sufficient GPU memory for the Qwen3-Embedding-8B model

### Dependencies
```bash
pip install torch transformers datasets uvicorn fastapi huggingface_hub numpy pydantic
```

Flash Attention 2 is required for optimal performance:
```bash
pip install flash-attn --no-build-isolation
```

## Quick Start

```bash
python search_server.py --host 0.0.0.0 --port 8000
```

On startup, the server will:
1. Download and load the corpus from HuggingFace (`Tevatron/browsecomp-plus-corpus`)
2. Download pre-computed corpus embeddings from HuggingFace (`miaolu3/browsecomp-plus`)
3. Load the Qwen3-Embedding-8B model on each available GPU
4. Start the FastAPI server

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `0.0.0.0` | Host address to bind |
| `--port` | `8000` | Port number |
| `--model` | `Qwen/Qwen3-Embedding-8B` | Embedding model name |
| `--corpus` | `Tevatron/browsecomp-plus-corpus` | Corpus dataset on HuggingFace |
| `--corpus-embedding-dataset` | `miaolu3/browsecomp-plus` | Pre-computed embeddings dataset |
| `--corpus-embedding-file` | `corpus_embeddings.pkl` | Embeddings file name in the dataset |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_GPUS` | Auto-detected | Number of GPUs to use |
| `MAX_BATCH_SIZE` | `2048` | Maximum batch size for query processing |
| `BATCH_TIMEOUT` | `0.005` | Batch timeout in seconds (5ms) |

Example with environment variables:
```bash
NUM_GPUS=4 MAX_BATCH_SIZE=512 python search_server.py --port 8000
```

## API Endpoints

### POST /search
Search for documents matching a query.

**Request:**
```json
{
  "query": "your search query",
  "k": 20
}
```

**Response:**
```json
{
  "results": [
    {
      "docid": "doc123",
      "url": "https://example.com/page",
      "text": "Document content (truncated to 1000 words)...",
      "score": 0.85
    }
  ],
  "took_ms": 45.2
}
```

### POST /open
Retrieve full document content by docid or URL.

**Request:**
```json
{
  "docid": "doc123"
}
```
or
```json
{
  "url": "https://example.com/page"
}
```

**Response:**
```json
{
  "results": [
    {
      "docid": "doc123",
      "url": "https://example.com/page",
      "text": "Full document content (up to 15000 words)..."
    }
  ],
  "took_ms": 1.2
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "workers": 8
}
```

### GET /stats
Server statistics.

**Response:**
```json
{
  "corpus_size": 100000,
  "num_workers": 8,
  "queue_sizes": {
    "requests": 0,
    "batches": 0,
    "results": 0
  },
  "pending": 0
}
```

## Architecture

The server uses a multi-process architecture for high throughput:

1. **Main Process**: Handles HTTP requests via FastAPI/uvicorn
2. **Batcher Process**: Collects incoming requests and batches them for efficient GPU processing
3. **Worker Processes**: One per GPU, each loads the embedding model and processes query batches
4. **Result Collector Thread**: Collects results from workers and returns them to waiting requests

This design enables processing of 10,000+ concurrent requests with low latency.

## Example Usage

```python
import requests

# Search for documents
response = requests.post(
    "http://localhost:8000/search",
    json={"query": "machine learning fundamentals", "k": 10}
)
results = response.json()

# Open a specific document
response = requests.post(
    "http://localhost:8000/open",
    json={"docid": results["results"][0]["docid"]}
)
full_doc = response.json()
```

# Local Embeddings Setup Guide

This guide explains how to use local embedding models instead of OpenAI's API to reduce costs.

## Benefits of Local Embeddings

- **Zero API costs**: No charges for embedding generation
- **Faster processing**: No network latency after initial model download
- **Privacy**: All processing happens locally
- **Offline capability**: Works without internet connection

## Installation

1. Install the required packages:
```bash
pip install sentence-transformers torch
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Configuration

### Option 1: Using .env file (Recommended)

Add these lines to your `.env` file:

```env
# Use local embeddings instead of OpenAI
USE_LOCAL_EMBEDDINGS=true

# Optional: Choose a specific local model (default: all-MiniLM-L6-v2)
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Option 2: Set Environment Variables Directly

```bash
# Windows PowerShell
$env:USE_LOCAL_EMBEDDINGS="true"
$env:LOCAL_EMBEDDING_MODEL="all-MiniLM-L6-v2"

# Linux/Mac
export USE_LOCAL_EMBEDDINGS=true
export LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

## Available Local Models

The code supports any model from Hugging Face that works with `sentence-transformers`. Here are recommended options:

### Fast Models (Recommended for speed)
- **`all-MiniLM-L6-v2`** (Default)
  - Dimensions: 384
  - Size: ~80MB
  - Speed: Very fast
  - Quality: Good for most use cases

- **`multi-qa-MiniLM-L6-cos-v1`**
  - Dimensions: 384
  - Size: ~80MB
  - Optimized for question-answering and search

### Higher Quality Models (Better accuracy, slower)
- **`all-mpnet-base-v2`**
  - Dimensions: 768
  - Size: ~420MB
  - Speed: Moderate
  - Quality: Better semantic understanding

- **`all-MiniLM-L12-v2`**
  - Dimensions: 384
  - Size: ~120MB
  - Speed: Moderate
  - Quality: Better than L6, faster than mpnet

## First Run

On the first run, the model will be automatically downloaded from Hugging Face:
- Models are cached in `~/.cache/torch/sentence_transformers/`
- Subsequent runs will use the cached model
- Download happens only once per model

## Switching Between OpenAI and Local

Simply change the `USE_LOCAL_EMBEDDINGS` setting:

```env
# Use OpenAI (default)
USE_LOCAL_EMBEDDINGS=false

# Use local model
USE_LOCAL_EMBEDDINGS=true
```

**Note**: The embedding cache includes the model identifier, so embeddings from different models won't conflict.

## Performance Comparison

| Model | Speed | Quality | Cost | Dimensions |
|-------|-------|---------|------|------------|
| OpenAI text-embedding-3-small | Medium | Excellent | $0.02/1M tokens | 1536 |
| all-MiniLM-L6-v2 (local) | Fast | Good | Free | 384 |
| all-mpnet-base-v2 (local) | Moderate | Very Good | Free | 768 |

## Troubleshooting

### Model not downloading
- Ensure you have internet connection on first run
- Check disk space (models are 80-420MB)
- Try manual download: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

### Out of memory errors
- Use a smaller model like `all-MiniLM-L6-v2`
- Reduce `batch_size` in `get_or_create_chunk_embeddings()`

### Slow performance
- Use GPU if available (PyTorch will detect automatically)
- Reduce batch size if using CPU
- Use a faster model like `all-MiniLM-L6-v2`

## Notes

- Local embeddings may have slightly different dimensions than OpenAI (384 or 768 vs 1536)
- Semantic similarity scores may differ slightly, but rankings should be similar
- Both OpenAI and local embeddings work well for semantic search
- The cache automatically separates embeddings by model to avoid conflicts

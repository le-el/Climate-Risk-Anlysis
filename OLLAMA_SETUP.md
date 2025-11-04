# Ollama Setup Guide

This guide explains how to use Ollama for local LLM inference instead of OpenAI's API, reducing costs and keeping data private.

## Benefits of Using Ollama

- **Zero API costs**: No charges for LLM inference
- **Privacy**: All data processing happens locally on your machine
- **Fast inference**: Runs on your GPU (RTX 3060) for excellent performance
- **Offline capability**: Works without internet connection (after model download)
- **Full control**: No rate limits or API quotas

## Prerequisites

1. **Ollama installed** (already done ✅)
   - Verify installation: `ollama --version`
   - Ollama should be running (starts automatically on most systems)

2. **RTX 3060 GPU** (12GB VRAM) ✅
   - Ollama will automatically use your GPU for inference

3. **Python package**:
   ```bash
   pip install ollama
   ```
   
   Or install all requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Downloading Models

Before first use, you need to download a model. Run in terminal:

```bash
# Recommended for RTX 3060 (3B parameters, fast and efficient)
ollama pull llama3.2:3b

# Or try other models:
# ollama pull llama3.2:1b      # Smaller, faster
# ollama pull qwen2.5:7b       # Excellent for structured tasks
# ollama pull mistral:7b       # Good balance
# ollama pull llama3.1:8b      # Better quality, may be slower
```

Models are cached locally after download. The first run will be slower as the model loads into GPU memory.

## Configuration

### Option 1: Using .env file (Recommended)

Add these lines to your `.env` file:

```env
# Use Ollama instead of OpenAI
USE_OLLAMA=true

# Choose your Ollama model (default: llama3.2:3b)
OLLAMA_MODEL=llama3.2:3b
```

### Option 2: Set Environment Variables Directly

```bash
# Windows PowerShell
$env:USE_OLLAMA="true"
$env:OLLAMA_MODEL="llama3.2:3b"

# Linux/Mac
export USE_OLLAMA=true
export OLLAMA_MODEL=llama3.2:3b
```

## Recommended Models for RTX 3060

Your RTX 3060 has **12GB VRAM**, which can handle various model sizes:

### Fast & Efficient (Recommended)
- **`llama3.2:3b`** (Default)
  - Size: ~2GB
  - Speed: Very fast
  - Quality: Good for structured extraction
  - Best for: Quick iterations and testing

- **`llama3.2:1b`**
  - Size: ~700MB
  - Speed: Fastest
  - Quality: Adequate for simple tasks
  - Best for: Maximum speed

### Better Quality (Slower)
- **`qwen2.5:7b`**
  - Size: ~4.5GB
  - Speed: Moderate
  - Quality: Excellent for structured tasks
  - Best for: Production use with better accuracy

- **`mistral:7b`**
  - Size: ~4.5GB
  - Speed: Moderate
  - Quality: Very good
  - Best for: Balanced quality and speed

- **`llama3.1:8b`**
  - Size: ~4.7GB
  - Speed: Moderate to slow
  - Quality: Excellent
  - Best for: When quality is most important

## Testing Ollama Setup

Test that Ollama is working:

```bash
# Test in terminal
ollama run llama3.2:3b "Extract JSON: {\"name\": \"test\"}"

# Or test in Python
python -c "import ollama; print(ollama.chat(model='llama3.2:3b', messages=[{'role': 'user', 'content': 'Say hello'}]))"
```

## Usage

Once configured, simply run your analysis:

```bash
python run_full_pipeline.py
```

The script will automatically:
- Use Ollama for LLM inference
- Display which model is being used
- Process all companies and measures
- Use GPU acceleration automatically

## Performance Tips

1. **First Run**: The first extraction may be slower as the model loads into GPU memory. Subsequent extractions will be faster.

2. **Batch Processing**: Ollama handles multiple requests efficiently, so processing all measures for a company is fast.

3. **GPU Memory**: If you encounter out-of-memory errors:
   - Use a smaller model (`llama3.2:1b` or `llama3.2:3b`)
   - Close other GPU-intensive applications
   - Restart Ollama: `ollama serve`

4. **Speed**: On RTX 3060, you can expect:
   - `llama3.2:3b`: ~50-100 tokens/second
   - `qwen2.5:7b`: ~30-60 tokens/second
   - `llama3.1:8b`: ~20-40 tokens/second

## Troubleshooting

### Ollama not found
```bash
# Check if Ollama is installed
ollama --version

# If not installed, download from: https://ollama.ai
```

### Model not found
```bash
# List available models
ollama list

# Pull the model
ollama pull llama3.2:3b
```

### GPU not being used
```bash
# Check GPU usage (Windows)
nvidia-smi

# Ollama should automatically use GPU. If not, ensure CUDA is installed.
```

### Out of Memory Errors
- Use a smaller model (`llama3.2:1b` or `llama3.2:3b`)
- Close other applications using GPU
- Restart Ollama: `ollama serve`

### Slow Performance
- Ensure GPU is being used (check `nvidia-smi`)
- Use a smaller model
- Close other GPU-intensive applications

## Combining with Local Embeddings

For **100% local processing** (no API costs at all):

```env
# .env file
USE_OLLAMA=true
OLLAMA_MODEL=llama3.2:3b
USE_LOCAL_EMBEDDINGS=true
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2
```

This setup:
- ✅ Zero API costs
- ✅ Full privacy (everything runs locally)
- ✅ Fast processing on your GPU
- ✅ Works offline

## Switching Back to OpenAI

Simply change in `.env`:

```env
USE_OLLAMA=false
```

Or remove/comment out the `USE_OLLAMA` line.

## Model Recommendations by Task

- **Structured Data Extraction**: `qwen2.5:7b` or `llama3.2:3b`
- **Fast Iterations**: `llama3.2:1b` or `llama3.2:3b`
- **Maximum Quality**: `llama3.1:8b` or `qwen2.5:7b`
- **Limited GPU Memory**: `llama3.2:1b`

## Notes

- Models are cached after first download
- Ollama automatically manages GPU memory
- The code includes JSON extraction to handle any extra text the model adds
- Temperature is set to 0.2 for consistent, structured outputs
- Response length is limited to 2000 tokens to keep outputs focused

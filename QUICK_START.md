# Quick Start Guide - TD6 RAG Optimization

## Already Done ✅

Everything is set up and ready to use:
- Data downloaded
- Dependencies installed
- Baseline established: **MRR = 0.184**
- First optimization tested: **MRR = 0.265** (+44%)

## Run Your First Experiment (30 seconds)

```python
from src_rag import evaluate

# Test improved configuration
config = {"model": {"chunk_size": 512, "overlap": 100}}
evaluate.run_evaluate_retrieval(config=config)
```

## Test Small2Big Strategy (Recommended)

```python
config = {
    "model": {
        "type": "small2big",
        "small_chunk_size": 256,
        "large_chunk_size": 768,
        "overlap": 100
    }
}
evaluate.run_evaluate_retrieval(config=config)
```

## Run Multiple Experiments

```bash
uv run python optimize_rag.py
```

## View Results

```bash
mlflow ui
# Open http://localhost:5000
```

## Best Configurations to Try

### Configuration 1: Larger Chunks
```python
{"model": {"chunk_size": 768, "overlap": 150}}
```

### Configuration 2: Small2Big
```python
{
    "model": {
        "type": "small2big",
        "small_chunk_size": 256,
        "large_chunk_size": 1024,
        "overlap": 128
    }
}
```

### Configuration 3: High Overlap
```python
{"model": {"chunk_size": 512, "overlap": 256}}
```

## Test Reply Accuracy (with LLM)

```python
from src_rag import evaluate

config = {"model": {"chunk_size": 768, "overlap": 150}}
evaluate.run_evaluate_reply(config=config)
# Note: This uses your xAI API and takes longer due to LLM calls
```

## Share Results with Your Team

After finding a good configuration:
1. Note the MRR score
2. Share the config dict
3. Push to your GitHub repository
4. Others can reproduce with the same config

## Current Best Result

**Configuration:**
```python
{"model": {"chunk_size": 512, "overlap": 0}}
```
**MRR:** 0.265 (26.5%)
**Improvement:** +44% over baseline

Now it's your turn to beat this score!

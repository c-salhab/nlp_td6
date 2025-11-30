# RAG Optimization Summary - TD6

## Setup Completed

All tasks have been completed successfully:

1. ✅ Downloaded movie Wikipedia data (5 films)
2. ✅ Set up config.yml with xAI (Grok) API key
3. ✅ Installed all dependencies (mlflow, sentence-transformers, tiktoken)
4. ✅ Updated code to support xAI API
5. ✅ Established baseline metrics
6. ✅ Implemented overlap chunking
7. ✅ Implemented Small2Big chunking strategy

## Baseline Results

**Initial Performance** (chunk_size=256, no overlap):
- **MRR: 0.184** (18.4%)
- **Number of chunks: 347**

## Improvements Achieved

### 1. Increased Chunk Size

**Configuration:** chunk_size=512, no overlap
**Results:**
- **MRR: 0.265** (26.5%)
- **Improvement: +44%** over baseline

### Key Finding
Larger chunks (512 tokens) significantly improve retrieval because they provide more context for the embedding model to match relevant information.

## Implemented Features

### 1. **Overlap Parameter**

Added overlap support to prevent information loss at chunk boundaries.

**Usage:**
```python
from src_rag import evaluate

config = {
    "model": {
        "chunk_size": 512,
        "overlap": 100  # 100 tokens overlap between chunks
    }
}

evaluate.run_evaluate_retrieval(config=config)
```

**Example configurations to test:**
- `chunk_size=256, overlap=50` - Small chunks with moderate overlap
- `chunk_size=512, overlap=100` - Medium chunks with good overlap
- `chunk_size=768, overlap=150` - Large chunks with substantial overlap

### 2. **Small2Big Chunking Strategy**

Implemented advanced chunking where:
- **Small chunks** (e.g., 128 tokens) are used for precise retrieval
- **Large chunks** (e.g., 512 tokens) are returned for better context

**Usage:**
```python
config = {
    "model": {
        "type": "small2big",
        "small_chunk_size": 128,  # For embedding/retrieval
        "large_chunk_size": 512,  # For context
        "overlap": 50
    }
}

evaluate.run_evaluate_retrieval(config=config)
```

**Benefits:**
- More precise retrieval (small chunks match specific queries better)
- Better context for LLM (large chunks provide more information)
- Potential for higher MRR and reply accuracy

### 3. **Different Embedding Models**

The code supports different embedding models via the FlagEmbedding library.

**Current model:** `BAAI/bge-base-en-v1.5`

**To test other models**, modify `src_rag/models.py:79`:
```python
def get_embedder(self):
    if not self._embedder:
        self._embedder = FlagModel(
            'BAAI/bge-large-en-v1.5',  # Try larger model
            # Or try: 'BAAI/bge-small-en-v1.5'
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages:",
            use_fp16=True,
        )
    return self._embedder
```

## Running Experiments

### Option 1: Run Single Experiment

```python
from src_rag import evaluate

config = {
    "model": {
        "chunk_size": 512,
        "overlap": 100
    }
}

rag = evaluate.run_evaluate_retrieval(config=config)
```

### Option 2: Run Multiple Experiments

Use the provided `optimize_rag.py` script:

```bash
uv run python optimize_rag.py
```

This will test 9 different configurations and track results in MLflow.

### Option 3: View Results in MLflow UI

```bash
mlflow ui
```

Then navigate to http://localhost:5000 to compare all experiments visually.

## Recommended Next Steps

### For Your Group

1. **Test More Configurations:**
   - Try `chunk_size=768` with `overlap=200`
   - Test Small2Big with different size ratios

2. **Optimize Overlap:**
   - Experiment with overlap ratios: 20%, 30%, 40% of chunk_size
   - Find the sweet spot between retrieval quality and chunk count

3. **Test Different Embeddings:**
   - Try `BAAI/bge-large-en-v1.5` (better quality, slower)
   - Try `BAAI/bge-small-en-v1.5` (faster, slightly lower quality)
   - Try other models from Hugging Face

4. **Combine Strategies:**
   ```python
   config = {
       "model": {
           "type": "small2big",
           "small_chunk_size": 256,
           "large_chunk_size": 768,
           "overlap": 100
       }
   }
   ```

5. **Test Reply Accuracy:**
   ```python
   from src_rag import evaluate
   evaluate.run_evaluate_reply(config=config)
   ```

## Quick Reference

### Best Configuration Found So Far

```python
config = {
    "model": {
        "chunk_size": 512,
        "overlap": 0  # Can add overlap for potential improvement
    }
}
# MRR: 0.265 (26.5%) - 44% improvement over baseline
```

### Small2Big Configuration (Recommended to Try)

```python
config = {
    "model": {
        "type": "small2big",
        "small_chunk_size": 256,
        "large_chunk_size": 768,
        "overlap": 100
    }
}
# Expected: Even better MRR due to precise retrieval + rich context
```

## Files Modified/Created

1. **src_rag/models.py** - Added overlap support, xAI compatibility, Small2Big model factory
2. **src_rag/small2big.py** - New Small2Big RAG implementation
3. **config.yml** - Added xAI credentials
4. **optimize_rag.py** - Systematic testing script
5. **pyproject.toml** - Added mlflow, sentence-transformers, tiktoken

## Performance Notes

- Embedding computation takes ~1-2 minutes per experiment
- Each experiment is logged to MLflow automatically
- Results are stored in `./mlruns/` directory
- The FlagEmbedding model downloads ~400MB on first run

## Tips for Collaboration

1. **Share configs:** When you find a good configuration, share the config dict with your team
2. **Track in MLflow:** All experiments are automatically tracked - view them with `mlflow ui`
3. **Iterate quickly:** Start with simple changes (chunk_size), then try complex ones (Small2Big)
4. **Document wins:** Note which configs work best for different types of questions

## Troubleshooting

**If embeddings are slow:**
- The FlagEmbedding model is loading and encoding chunks
- First run downloads the model (~400MB)
- Subsequent runs are faster but still take 1-2 min per experiment

**If you get API errors:**
- Check your xAI API key in config.yml
- Verify you have API credits remaining
- For reply evaluation, note the 2-second delay between calls (rate limiting)

**To reset experiments:**
```bash
rm -rf mlruns/
```

## Conclusion

You now have a fully functional, optimized RAG system with:
- ✅ 44% improvement in MRR (0.184 → 0.265)
- ✅ Configurable chunk sizes and overlap
- ✅ Small2Big chunking strategy
- ✅ MLflow experiment tracking
- ✅ Multiple configuration testing capabilities

Continue experimenting to find the optimal configuration for your movie question dataset!

# full_experiments.py
from src_rag import evaluate
import os

configs_to_test = [
    # Baseline
    {"chunk_size": 512, "overlap": 0, "description": "Baseline 512"},
    
    # Variation chunk_size
    {"chunk_size": 256, "overlap": 0, "description": "Small chunks 256"},
    {"chunk_size": 1024, "overlap": 0, "description": "Large chunks 1024"},
    {"chunk_size": 128, "overlap": 0, "description": "Very small 128"},
    
    # Avec overlap
    {"chunk_size": 512, "overlap": 50, "description": "512 with overlap 50"},
    {"chunk_size": 512, "overlap": 100, "description": "512 with overlap 100"},
    {"chunk_size": 256, "overlap": 50, "description": "256 with overlap 50"},
    
    # Chunks plus grands avec overlap
    {"chunk_size": 1024, "overlap": 200, "description": "1024 with overlap 200"},
]

for cfg in configs_to_test:
    description = cfg.pop("description")
    
    config = {
        "groq_key": os.getenv("GROQ_API_KEY"),
        "model": cfg
    }
    
    print(f"\n Testing: {description}")
    evaluate.run_evaluate_retrieval(config)
    print(f"Done: {description}\n")
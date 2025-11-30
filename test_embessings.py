from src_rag import evaluate
import os

embedding_models = [
    "BAAI/bge-base-en-v1.5",      
    "BAAI/bge-large-en-v1.5",     
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-m3" 
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1",
]


best_chunk_config = {
    "chunk_size": 500,  
    "overlap": 120,       
}

for model_name in embedding_models:
    config = {
        "groq_key": os.getenv("GROQ_API_KEY"),
        "model": {
            **best_chunk_config,
            "embedding_model": model_name
        }
    }
    
    print(f"\n[TEST] Testing embedding: {model_name}")  
    evaluate.run_evaluate_retrieval(config, rag=None)
    print(f"[DONE]\n")
from src_rag import evaluate

def run_experiments():
    configurations = [
        {"chunk_size": 256, "chunk_overlap": 0, "embedding_model": "all-MiniLM-L6-v2"},
        {"chunk_size": 256, "chunk_overlap": 50, "embedding_model": "all-MiniLM-L6-v2"},
        {"chunk_size": 512, "chunk_overlap": 0, "embedding_model": "all-MiniLM-L6-v2"},
        {"chunk_size": 512, "chunk_overlap": 50, "embedding_model": "all-MiniLM-L6-v2"},
        {"chunk_size": 1024, "chunk_overlap": 0, "embedding_model": "all-MiniLM-L6-v2"},
        {"chunk_size": 1024, "chunk_overlap": 100, "embedding_model": "all-MiniLM-L6-v2"},
        
        {"chunk_size": 512, "chunk_overlap": 50, "embedding_model": "all-mpnet-base-v2"},
        {"chunk_size": 512, "chunk_overlap": 50, "embedding_model": "BAAI/bge-base-en-v1.5"},
    ]
    
    print(f"Launching {len(configurations)} experiment")
    
    for i, config in enumerate(configurations):
        print(f"\n--- Experience {i+1}/{len(configurations)} ---")
        print(f"Configuration: {config}")
        
        model_config = {"model": config}
        
        try:
            rag_model = evaluate.run_evaluate_retrieval(config=model_config)
            
            if i % 2 == 0:  # une config sur 2
                evaluate.run_evaluate_reply(config=model_config, rag=rag_model)
                
        except Exception as e:
            print(f"Error in experience: {e}")
            continue

    print("Check results in : mlflow ui")

if __name__ == "__main__":
    run_experiments()
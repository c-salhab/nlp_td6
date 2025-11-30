"""
Script to systematically test different RAG configurations and track results in MLflow
"""
from src_rag import evaluate
import time

# Configurations to test
CONFIGS = [
    # Baseline
    {"model": {"chunk_size": 256, "overlap": 0}, "description": "baseline_256_no_overlap"},

    # Test different chunk sizes without overlap
    {"model": {"chunk_size": 128, "overlap": 0}, "description": "chunk_128_no_overlap"},
    {"model": {"chunk_size": 512, "overlap": 0}, "description": "chunk_512_no_overlap"},
    {"model": {"chunk_size": 1024, "overlap": 0}, "description": "chunk_1024_no_overlap"},

    # Test overlap with 256 chunk size
    {"model": {"chunk_size": 256, "overlap": 50}, "description": "chunk_256_overlap_50"},
    {"model": {"chunk_size": 256, "overlap": 100}, "description": "chunk_256_overlap_100"},

    # Test overlap with 512 chunk size
    {"model": {"chunk_size": 512, "overlap": 100}, "description": "chunk_512_overlap_100"},
    {"model": {"chunk_size": 512, "overlap": 200}, "description": "chunk_512_overlap_200"},

    # Test larger chunks with overlap
    {"model": {"chunk_size": 768, "overlap": 150}, "description": "chunk_768_overlap_150"},
]

def run_all_experiments():
    results = []

    for i, config in enumerate(CONFIGS):
        print(f"\n{'='*80}")
        print(f"Running experiment {i+1}/{len(CONFIGS)}: {config.get('description', 'unnamed')}")
        print(f"Config: {config['model']}")
        print(f"{'='*80}\n")

        try:
            start_time = time.time()
            rag = evaluate.run_evaluate_retrieval(config=config)
            elapsed = time.time() - start_time

            print(f"\n✓ Completed in {elapsed:.2f}s")
            results.append({
                "config": config,
                "status": "success",
                "time": elapsed
            })
        except Exception as e:
            print(f"\n✗ Failed: {e}")
            results.append({
                "config": config,
                "status": "failed",
                "error": str(e)
            })

        # Brief pause between experiments
        time.sleep(2)

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for i, result in enumerate(results):
        status = "✓" if result["status"] == "success" else "✗"
        print(f"{status} Experiment {i+1}: {CONFIGS[i].get('description', 'unnamed')}")
    print(f"\nRun 'mlflow ui' and navigate to http://localhost:5000 to view detailed results")

    return results

if __name__ == "__main__":
    results = run_all_experiments()

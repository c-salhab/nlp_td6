"""
Test différents modèles d'embedding pour optimiser le MRR

Modèles à tester :
- BAAI/bge-small-en-v1.5 : Plus rapide, moins précis
- BAAI/bge-base-en-v1.5 : Baseline (actuellement utilisé)
- BAAI/bge-large-en-v1.5 : Plus lent, plus précis
"""
from src_rag import evaluate
import time

# Meilleure config trouvée jusqu'à présent
BEST_CONFIG_BASE = {
    "chunk_size": 512,
    "overlap": 0,
    "top_k": 5
}

EMBEDDING_MODELS = [
    {
        "name": "bge-small",
        "model": "BAAI/bge-small-en-v1.5",
        "description": "Petit modèle - rapide mais moins précis"
    },
    {
        "name": "bge-base",
        "model": "BAAI/bge-base-en-v1.5",
        "description": "Modèle de base - bon compromis (baseline)"
    },
    {
        "name": "bge-large",
        "model": "BAAI/bge-large-en-v1.5",
        "description": "Grand modèle - lent mais très précis"
    },
]

def test_embedding_models():
    """Test différents modèles d'embedding avec la meilleure config"""

    results = []

    print(f"\n{'='*100}")
    print("TESTING EMBEDDING MODELS")
    print(f"Base config: {BEST_CONFIG_BASE}")
    print(f"{'='*100}\n")

    for i, emb_config in enumerate(EMBEDDING_MODELS):
        print(f"\n{'-'*100}")
        print(f"Test {i+1}/{len(EMBEDDING_MODELS)}: {emb_config['name']}")
        print(f"Model: {emb_config['model']}")
        print(f"Description: {emb_config['description']}")
        print(f"{'-'*100}\n")

        config = {
            "model": {
                **BEST_CONFIG_BASE,
                "embedding_model": emb_config['model']
            },
            "description": f"embedding_{emb_config['name']}_chunk512"
        }

        try:
            start_time = time.time()
            rag = evaluate.run_evaluate_retrieval(config=config)
            elapsed = time.time() - start_time

            results.append({
                "name": emb_config['name'],
                "model": emb_config['model'],
                "status": "✓ SUCCESS",
                "time_seconds": round(elapsed, 2)
            })

            print(f"\n✓ Completed in {elapsed:.2f}s")

        except Exception as e:
            print(f"\n✗ Failed: {e}")
            results.append({
                "name": emb_config['name'],
                "model": emb_config['model'],
                "status": f"✗ FAILED: {str(e)[:50]}",
                "time_seconds": 0
            })

        # Pause entre tests
        time.sleep(3)

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY - EMBEDDING MODELS")
    print(f"{'='*100}")

    for result in results:
        print(f"{result['status']:15} | {result['name']:15} | {result['time_seconds']}s")

    print(f"\n💡 View detailed results in MLflow UI:")
    print(f"   mlflow ui")
    print(f"   Navigate to http://localhost:5000")
    print(f"   Compare MRR across different embedding models")
    print(f"{'='*100}\n")

    return results


if __name__ == "__main__":
    test_embedding_models()

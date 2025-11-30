"""
Advanced RAG Optimization - Systematic Testing Framework

Tests multiples configurations pour maximiser le MRR :
1. Différents embeddings models
2. Différentes tailles de chunks + overlap
3. Stratégie Small2Big avec variations
4. Nombre de chunks retournés (top_k)
"""
from src_rag import evaluate
import time
import pandas as pd

# ============================================================================
# EXPÉRIMENTATIONS AVANCÉES
# ============================================================================

EXPERIMENTS = {
    # Phase 1: Optimisation chunk size + overlap
    "chunk_optimization": [
        {"model": {"chunk_size": 384, "overlap": 0}, "description": "chunk_384_no_overlap"},
        {"model": {"chunk_size": 384, "overlap": 96}, "description": "chunk_384_overlap_25pct"},
        {"model": {"chunk_size": 512, "overlap": 128}, "description": "chunk_512_overlap_25pct"},
        {"model": {"chunk_size": 640, "overlap": 160}, "description": "chunk_640_overlap_25pct"},
        {"model": {"chunk_size": 768, "overlap": 192}, "description": "chunk_768_overlap_25pct"},
        {"model": {"chunk_size": 1024, "overlap": 256}, "description": "chunk_1024_overlap_25pct"},
    ],

    # Phase 2: Small2Big avec différentes combinaisons
    "small2big_optimization": [
        {"model": {"type": "small2big", "small_chunk_size": 128, "large_chunk_size": 512, "overlap": 0},
         "description": "s2b_128_512_no_overlap"},
        {"model": {"type": "small2big", "small_chunk_size": 128, "large_chunk_size": 512, "overlap": 64},
         "description": "s2b_128_512_overlap_64"},
        {"model": {"type": "small2big", "small_chunk_size": 256, "large_chunk_size": 768, "overlap": 0},
         "description": "s2b_256_768_no_overlap"},
        {"model": {"type": "small2big", "small_chunk_size": 256, "large_chunk_size": 768, "overlap": 128},
         "description": "s2b_256_768_overlap_128"},
        {"model": {"type": "small2big", "small_chunk_size": 256, "large_chunk_size": 1024, "overlap": 128},
         "description": "s2b_256_1024_overlap_128"},
        {"model": {"type": "small2big", "small_chunk_size": 192, "large_chunk_size": 768, "overlap": 96},
         "description": "s2b_192_768_overlap_96"},
    ],

    # Phase 3: Overlap ratio testing
    "overlap_ratio_testing": [
        {"model": {"chunk_size": 512, "overlap": 51}, "description": "chunk_512_overlap_10pct"},
        {"model": {"chunk_size": 512, "overlap": 102}, "description": "chunk_512_overlap_20pct"},
        {"model": {"chunk_size": 512, "overlap": 154}, "description": "chunk_512_overlap_30pct"},
        {"model": {"chunk_size": 512, "overlap": 205}, "description": "chunk_512_overlap_40pct"},
        {"model": {"chunk_size": 512, "overlap": 256}, "description": "chunk_512_overlap_50pct"},
    ],

    # Phase 4: Best candidates avec top_k variations
    "top_k_optimization": [
        {"model": {"chunk_size": 768, "overlap": 192, "top_k": 3}, "description": "chunk_768_overlap_25pct_top3"},
        {"model": {"chunk_size": 768, "overlap": 192, "top_k": 5}, "description": "chunk_768_overlap_25pct_top5"},
        {"model": {"chunk_size": 768, "overlap": 192, "top_k": 7}, "description": "chunk_768_overlap_25pct_top7"},
        {"model": {"chunk_size": 768, "overlap": 192, "top_k": 10}, "description": "chunk_768_overlap_25pct_top10"},
    ],
}


def run_experiment_phase(phase_name, configs):
    """Run a specific phase of experiments"""
    print(f"\n{'='*100}")
    print(f"PHASE: {phase_name.upper().replace('_', ' ')}")
    print(f"{'='*100}\n")

    results = []

    for i, config in enumerate(configs):
        print(f"\n{'-'*100}")
        print(f"Experiment {i+1}/{len(configs)}: {config['description']}")
        print(f"Config: {config['model']}")
        print(f"{'-'*100}\n")

        try:
            start_time = time.time()
            rag = evaluate.run_evaluate_retrieval(config=config)
            elapsed = time.time() - start_time

            results.append({
                "phase": phase_name,
                "description": config['description'],
                "config": str(config['model']),
                "status": "✓ SUCCESS",
                "time_seconds": round(elapsed, 2)
            })

            print(f"\n✓ Completed in {elapsed:.2f}s")

        except Exception as e:
            print(f"\n✗ Failed: {e}")
            results.append({
                "phase": phase_name,
                "description": config['description'],
                "config": str(config['model']),
                "status": f"✗ FAILED: {str(e)[:50]}",
                "time_seconds": 0
            })

        # Pause entre expériences
        time.sleep(2)

    return results


def run_all_phases(phases_to_run=None):
    """Run all or selected experimental phases"""

    if phases_to_run is None:
        phases_to_run = list(EXPERIMENTS.keys())

    all_results = []

    print(f"\n{'#'*100}")
    print(f"# ADVANCED RAG OPTIMIZATION")
    print(f"# Running {len(phases_to_run)} phases")
    print(f"{'#'*100}\n")

    for phase in phases_to_run:
        if phase in EXPERIMENTS:
            results = run_experiment_phase(phase, EXPERIMENTS[phase])
            all_results.extend(results)
        else:
            print(f"⚠ Warning: Phase '{phase}' not found")

    # Summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")

    df_results = pd.DataFrame(all_results)
    print(df_results.to_string(index=False))

    success_count = len(df_results[df_results['status'] == '✓ SUCCESS'])
    total_count = len(df_results)
    total_time = df_results['time_seconds'].sum()

    print(f"\n{'='*100}")
    print(f"Success rate: {success_count}/{total_count} ({100*success_count/total_count:.1f}%)")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\n💡 View results in MLflow UI: mlflow ui")
    print(f"   Then navigate to http://localhost:5000")
    print(f"{'='*100}\n")

    return all_results


def run_quick_test():
    """Run a quick test with promising configs"""
    quick_configs = [
        {"model": {"chunk_size": 768, "overlap": 192}, "description": "quick_chunk_768_overlap_25pct"},
        {"model": {"type": "small2big", "small_chunk_size": 256, "large_chunk_size": 768, "overlap": 128},
         "description": "quick_s2b_256_768_overlap_128"},
    ]

    print("\n🚀 QUICK TEST - Testing 2 promising configurations\n")
    return run_experiment_phase("quick_test", quick_configs)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        phase = sys.argv[1]

        if phase == "quick":
            run_quick_test()
        elif phase == "all":
            run_all_phases()
        elif phase in EXPERIMENTS:
            run_all_phases([phase])
        else:
            print(f"Usage: python advanced_optimize.py [phase]")
            print(f"\nAvailable phases:")
            for p in EXPERIMENTS.keys():
                print(f"  - {p}")
            print(f"  - quick (run 2 best configs)")
            print(f"  - all (run all phases)")
    else:
        print("🎯 Advanced RAG Optimization")
        print("\nUsage: python advanced_optimize.py [phase]")
        print(f"\nAvailable phases:")
        for p in EXPERIMENTS.keys():
            print(f"  - {p}: {len(EXPERIMENTS[p])} experiments")
        print(f"  - quick: 2 promising configs")
        print(f"  - all: {sum(len(v) for v in EXPERIMENTS.values())} total experiments")
        print("\nExample: python advanced_optimize.py chunk_optimization")

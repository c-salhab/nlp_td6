"""
Analyse des résultats MLflow - Trouve les meilleures configurations

Ce script analyse toutes les expériences MLflow et affiche :
- Top 10 meilleures configurations par MRR
- Analyse par chunk_size
- Analyse par overlap
- Analyse par embedding model
- Recommandations
"""
import mlflow
import pandas as pd
from pathlib import Path

def get_all_experiments():
    """Récupère toutes les expériences MLflow"""

    # Set experiment
    mlflow.set_experiment("RAG_Movies_clean")

    # Get all runs
    runs = mlflow.search_runs(
        experiment_names=["RAG_Movies_clean"],
        order_by=["metrics.mrr DESC"]
    )

    return runs

def analyze_results():
    """Analyse complète des résultats"""

    print("\n" + "="*100)
    print("ANALYSE DES RÉSULTATS MLflow - RAG Movies")
    print("="*100 + "\n")

    runs = get_all_experiments()

    if len(runs) == 0:
        print("⚠ Aucune expérience trouvée dans MLflow!")
        print("Lancez d'abord des expériences avec optimize_rag.py ou advanced_optimize.py")
        return

    print(f"📊 Total d'expériences: {len(runs)}\n")

    # ========================================================================
    # TOP 10 MEILLEURES CONFIGURATIONS
    # ========================================================================
    print("="*100)
    print("🏆 TOP 10 MEILLEURES CONFIGURATIONS (par MRR)")
    print("="*100 + "\n")

    top_10 = runs.head(10)

    for i, row in top_10.iterrows():
        mrr = row.get('metrics.mrr', 0)
        nb_chunks = row.get('metrics.nb_chunks', 0)
        chunk_size = row.get('params.chunk_size', 'N/A')
        overlap = row.get('params.overlap', 'N/A')
        top_k = row.get('params.top_k', 'N/A')
        model_type = row.get('params.model_type', 'N/A')
        description = row.get('tags.description', 'N/A')

        print(f"#{i+1} - MRR: {mrr:.4f} | Chunks: {int(nb_chunks)}")
        print(f"    Config: chunk_size={chunk_size}, overlap={overlap}, top_k={top_k}, type={model_type}")
        print(f"    Description: {description}")
        print()

    # ========================================================================
    # ANALYSE PAR CHUNK SIZE
    # ========================================================================
    print("="*100)
    print("📏 ANALYSE PAR CHUNK SIZE")
    print("="*100 + "\n")

    chunk_analysis = runs.groupby('params.chunk_size').agg({
        'metrics.mrr': ['mean', 'max', 'count']
    }).round(4)
    chunk_analysis.columns = ['MRR Moyen', 'MRR Max', 'Nb Tests']
    chunk_analysis = chunk_analysis.sort_values('MRR Moyen', ascending=False)
    print(chunk_analysis)
    print()

    # ========================================================================
    # ANALYSE PAR OVERLAP
    # ========================================================================
    print("="*100)
    print("🔄 ANALYSE PAR OVERLAP")
    print("="*100 + "\n")

    overlap_analysis = runs.groupby('params.overlap').agg({
        'metrics.mrr': ['mean', 'max', 'count']
    }).round(4)
    overlap_analysis.columns = ['MRR Moyen', 'MRR Max', 'Nb Tests']
    overlap_analysis = overlap_analysis.sort_values('MRR Moyen', ascending=False)
    print(overlap_analysis)
    print()

    # ========================================================================
    # ANALYSE PAR TOP_K
    # ========================================================================
    if 'params.top_k' in runs.columns:
        print("="*100)
        print("🔝 ANALYSE PAR TOP_K (nombre de chunks retournés)")
        print("="*100 + "\n")

        topk_analysis = runs.groupby('params.top_k').agg({
            'metrics.mrr': ['mean', 'max', 'count']
        }).round(4)
        topk_analysis.columns = ['MRR Moyen', 'MRR Max', 'Nb Tests']
        topk_analysis = topk_analysis.sort_values('MRR Moyen', ascending=False)
        print(topk_analysis)
        print()

    # ========================================================================
    # ANALYSE PAR MODEL TYPE
    # ========================================================================
    if 'params.model_type' in runs.columns:
        print("="*100)
        print("🤖 ANALYSE PAR TYPE DE MODÈLE")
        print("="*100 + "\n")

        model_analysis = runs.groupby('params.model_type').agg({
            'metrics.mrr': ['mean', 'max', 'count']
        }).round(4)
        model_analysis.columns = ['MRR Moyen', 'MRR Max', 'Nb Tests']
        model_analysis = model_analysis.sort_values('MRR Moyen', ascending=False)
        print(model_analysis)
        print()

    # ========================================================================
    # ANALYSE PAR EMBEDDING MODEL
    # ========================================================================
    if 'params.embedding_model' in runs.columns:
        print("="*100)
        print("🧠 ANALYSE PAR EMBEDDING MODEL")
        print("="*100 + "\n")

        emb_analysis = runs.groupby('params.embedding_model').agg({
            'metrics.mrr': ['mean', 'max', 'count']
        }).round(4)
        emb_analysis.columns = ['MRR Moyen', 'MRR Max', 'Nb Tests']
        emb_analysis = emb_analysis.sort_values('MRR Moyen', ascending=False)
        print(emb_analysis)
        print()

    # ========================================================================
    # STATISTIQUES GLOBALES
    # ========================================================================
    print("="*100)
    print("📈 STATISTIQUES GLOBALES")
    print("="*100 + "\n")

    print(f"MRR Minimum:  {runs['metrics.mrr'].min():.4f}")
    print(f"MRR Maximum:  {runs['metrics.mrr'].max():.4f}")
    print(f"MRR Moyen:    {runs['metrics.mrr'].mean():.4f}")
    print(f"MRR Médian:   {runs['metrics.mrr'].median():.4f}")
    print()

    # ========================================================================
    # RECOMMANDATIONS
    # ========================================================================
    print("="*100)
    print("💡 RECOMMANDATIONS")
    print("="*100 + "\n")

    best_run = runs.iloc[0]
    best_mrr = best_run['metrics.mrr']
    best_chunk = best_run.get('params.chunk_size', 'N/A')
    best_overlap = best_run.get('params.overlap', 'N/A')
    best_topk = best_run.get('params.top_k', 'N/A')
    best_type = best_run.get('params.model_type', 'N/A')

    print(f"🥇 MEILLEURE CONFIGURATION:")
    print(f"   MRR: {best_mrr:.4f}")
    print(f"   chunk_size: {best_chunk}")
    print(f"   overlap: {best_overlap}")
    print(f"   top_k: {best_topk}")
    print(f"   model_type: {best_type}")
    print()

    # Config Python à utiliser
    print("📋 Configuration Python à utiliser:")
    print("```python")
    print("config = {")
    print("    'model': {")
    if best_type == 'small2big':
        print(f"        'type': '{best_type}',")
        print(f"        'small_chunk_size': {best_run.get('params.small_chunk_size', 'N/A')},")
        print(f"        'large_chunk_size': {best_run.get('params.large_chunk_size', 'N/A')},")
    else:
        print(f"        'chunk_size': {best_chunk},")
    print(f"        'overlap': {best_overlap},")
    print(f"        'top_k': {best_topk},")
    print("    }")
    print("}")
    print("```")
    print()

    # Amélioration vs baseline
    baseline_mrr = 0.184  # Premier baseline
    improvement = ((best_mrr - baseline_mrr) / baseline_mrr) * 100
    print(f"📊 Amélioration vs baseline: +{improvement:.1f}%")
    print(f"   (Baseline: {baseline_mrr:.4f} → Meilleur: {best_mrr:.4f})")
    print()

    print("="*100)
    print("💻 Pour voir les détails graphiques, lancez: mlflow ui")
    print("   Puis ouvrez: http://localhost:5000")
    print("="*100 + "\n")


def export_results_csv():
    """Exporte tous les résultats en CSV"""
    runs = get_all_experiments()

    output_file = "mlflow_results.csv"
    runs.to_csv(output_file, index=False)

    print(f"✅ Résultats exportés dans: {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "export":
        export_results_csv()
    else:
        analyze_results()

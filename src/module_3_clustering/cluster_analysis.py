# =============================================================================
# MODULE 3 — CLUSTERING 
# =============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import warnings
import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# Configuration MLflow
mlflow.set_tracking_uri("mlruns")  
mlflow.set_experiment("eco_smart_clustering")

CLEAN_DATA_PATH = "data/processed/clean_data.csv"
PROCESSED_DIR = "data/processed"
FIGURES_DIR = "reports/figures"
MODELS_DIR = "models"
os.makedirs(FIGURES_DIR, exist_ok=True)

NUM_COLS = ["Poids", "Volume", "Conductivite", "Opacite", "Rigidite"]


def run_clustering(data_path: str = CLEAN_DATA_PATH,
                   k_max: int = 10) -> dict:
    """
    Pipeline de clustering non-supervisé avec tracking MLflow.
    """
    print("=" * 60)
    print("MODULE 3 — CLUSTERING")
    print("=" * 60)

    # Démarrer un run MLflow
    with mlflow.start_run(run_name="KMeans_Clustering"):
        
        df = pd.read_csv(data_path)
        df_num = df[NUM_COLS].dropna().copy()
        print(f"Échantillons pour le clustering : {df_num.shape[0]}")
        
        # Log des paramètres
        mlflow.log_param("n_samples", df_num.shape[0])
        mlflow.log_param("n_features", len(NUM_COLS))
        mlflow.log_param("k_max", k_max)
        
        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_num)
        
        # -------------------------------------------------------------------
        # 1. MÉTHODE DU COUDE (Elbow Method)
        # -------------------------------------------------------------------
        print("\n--- Méthode du coude ---")
        inertias = []
        silhouettes = []
        K_range = range(2, k_max + 1)
        
        for k in K_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            sil = silhouette_score(X_scaled, km.labels_)
            silhouettes.append(sil)
            print(f"  k={k} — Inertie: {km.inertia_:.1f} | Silhouette: {sil:.4f}")
        
        # Log des métriques pour chaque k
        for k, inert, sil in zip(K_range, inertias, silhouettes):
            mlflow.log_metric(f"inertia_k{k}", inert)
            mlflow.log_metric(f"silhouette_k{k}", sil)
        
        # Tracé Elbow
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(list(K_range), inertias, "bo-")
        ax1.set_xlabel("Nombre de clusters (k)")
        ax1.set_ylabel("Inertie")
        ax1.set_title("Méthode du coude")
        ax2.plot(list(K_range), silhouettes, "rs-")
        ax2.set_xlabel("Nombre de clusters (k)")
        ax2.set_ylabel("Score de silhouette")
        ax2.set_title("Score de silhouette")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "fig08_elbow_method.png"), dpi=120)
        plt.close()
        print("  ✓ fig08_elbow_method.png sauvegardé")
        
        # Log de la figure
        mlflow.log_artifact(os.path.join(FIGURES_DIR, "fig08_elbow_method.png"))
        
        # Choix du k optimal
        best_k = list(K_range)[np.argmax(silhouettes)]
        mlflow.log_param("best_k", best_k)
        mlflow.log_metric("best_silhouette", max(silhouettes))
        print(f"\n  → k optimal retenu : {best_k} (silhouette max)")
        
        # -------------------------------------------------------------------
        # 2. KMEANS FINAL
        # -------------------------------------------------------------------
        print(f"\n--- KMeans avec k={best_k} ---")
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        print(f"  Distribution des clusters : {np.bincount(cluster_labels)}")
        
        # Log de la distribution des clusters
        for i, count in enumerate(np.bincount(cluster_labels)):
            mlflow.log_metric(f"cluster_{i}_size", count)
        
        # Sauvegarde
        df_num = df_num.copy()
        df_num["cluster"] = cluster_labels
        df_num.to_csv(os.path.join(PROCESSED_DIR, "clean_data_with_clusters.csv"), index=False)
        
        # -------------------------------------------------------------------
        # 3. VISUALISATION PCA 2D
        # -------------------------------------------------------------------
        print("\n--- Réduction PCA 2D ---")
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        
        explained = pca.explained_variance_ratio_
        mlflow.log_metric("pca_explained_variance_ratio_1", explained[0])
        mlflow.log_metric("pca_explained_variance_ratio_2", explained[1])
        mlflow.log_metric("pca_total_explained_variance", sum(explained))
        
        print(f"  Variance expliquée : PC1={explained[0]:.2%} | PC2={explained[1]:.2%}")
        print(f"  Total expliqué     : {sum(explained):.2%}")
        
        plt.figure(figsize=(9, 7))
        cmap = plt.get_cmap("tab10")
        for k in range(best_k):
            mask = cluster_labels == k
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        c=[cmap(k)], label=f"Cluster {k}",
                        alpha=0.7, edgecolors="white", linewidths=0.3, s=40)
        plt.xlabel(f"PC1 ({explained[0]:.1%})")
        plt.ylabel(f"PC2 ({explained[1]:.1%})")
        plt.title(f"Clusters KMeans (k={best_k}) — projection PCA 2D")
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "fig9_pca_clusters.png"), dpi=120)
        plt.close()
        print("  ✓ fig9_pca_clusters.png sauvegardé")
        
        mlflow.log_artifact(os.path.join(FIGURES_DIR, "fig9_pca_clusters.png"))
        
        # -------------------------------------------------------------------
        # 4. SAUVEGARDE DES MODÈLES
        # -------------------------------------------------------------------
        joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
        joblib.dump(pca, os.path.join(MODELS_DIR, "pca_model.pkl"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, "clustering_scaler.pkl"))
        print("  ✓ kmeans_model.pkl, pca_model.pkl, clustering_scaler.pkl sauvegardés")
        
        # Log des artefacts
        mlflow.log_artifact(os.path.join(MODELS_DIR, "kmeans_model.pkl"))
        mlflow.log_artifact(os.path.join(MODELS_DIR, "pca_model.pkl"))
        mlflow.log_artifact(os.path.join(MODELS_DIR, "clustering_scaler.pkl"))
        
        # Sauvegarde PCA data
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["cluster"] = cluster_labels
        pca_df.to_csv(os.path.join(PROCESSED_DIR, "pca_data.csv"), index=False)
        
        print("\n" + "=" * 60)
        print("MODULE 3 TERMINÉ")
        print("=" * 60)
        
        # Log du modèle KMeans dans MLflow
        mlflow.sklearn.log_model(kmeans, "kmeans_model")
        
        return {
            "kmeans": kmeans,
            "pca": pca,
            "scaler": scaler,
            "cluster_labels": cluster_labels,
            "best_k": best_k,
            "X_pca": X_pca,
        }


if __name__ == "__main__":
    run_clustering()
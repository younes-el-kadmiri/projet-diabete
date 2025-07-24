import sys
import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

# Pour trouver le module core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.clusterer import Clusterer

def test_clusterer_workflow():
    # Générer des données synthétiques
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    df = pd.DataFrame(X, columns=["Feature1", "Feature2"])

    clusterer = Clusterer(df)
    
    # Test du scale
    X_scaled = clusterer.scale_data()
    assert X_scaled.shape == df.shape, "Erreur de mise à l'échelle"

    # Test de la méthode du coude (ne plante pas)
    clusterer.elbow_method(max_k=5)

    # Test du fit kmeans
    clustered_df = clusterer.fit_kmeans(n_clusters=3)
    assert 'Cluster' in clustered_df.columns, "Clustering non effectué"

    # Test des moyennes par cluster
    means = clusterer.cluster_means()
    assert means.shape[0] == 3, "Nombre de clusters incorrect"

    print("Tous les tests ont réussi.")

if __name__ == "__main__":
    test_clusterer_workflow()

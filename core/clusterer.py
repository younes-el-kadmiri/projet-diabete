from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # ⚠️ à placer AVANT pyplot
import matplotlib.pyplot as plt


class Clusterer:
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.kmeans = None
        self.X_scaled = None

    def scale_data(self):
        self.X_scaled = self.scaler.fit_transform(self.df)
        return self.X_scaled

    def elbow_method(self, max_k=10, save_path=None):
        if self.X_scaled is None:
            raise ValueError("Les données doivent être normalisées avec `scale_data()` avant de continuer.")
        
        inertias = []
        for k in range(1, max_k + 1):
            model = KMeans(n_clusters=k, random_state=42)
            model.fit(self.X_scaled)
            inertias.append(model.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, max_k + 1), inertias, marker='o')
        plt.xlabel('Nombre de clusters (k)')
        plt.ylabel('Inertie (Within-cluster sum of squares)')
        plt.title('Méthode du coude')
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig("cluster_plot.png")
        plt.close()

    def fit_kmeans(self, n_clusters):
        if self.X_scaled is None:
            raise ValueError("Les données doivent être normalisées avec `scale_data()` avant de continuer.")

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Cluster'] = self.kmeans.fit_predict(self.X_scaled)
        return self.df

    def cluster_means(self):
        if 'Cluster' not in self.df.columns:
            raise ValueError("Veuillez exécuter `fit_kmeans()` d'abord.")
        return self.df.groupby('Cluster').mean()

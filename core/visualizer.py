import matplotlib
matplotlib.use("Agg")  # Pour éviter les erreurs d'affichage dans un environnement sans GUI
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Visualizer:
    @staticmethod
    def plot_clusters_2d(pca_components, labels, save_path=None):
        df_plot = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
        df_plot['Cluster'] = labels

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_plot, x='PC1', y='PC2', hue='Cluster', palette='Set1')
        plt.title('Clusters visualisés après PCA')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

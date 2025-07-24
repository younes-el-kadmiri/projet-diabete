from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PCA_Reducer:
    def __init__(self, df):
        self.df = df.copy()
        self.pca = None
    
    def fit_transform(self, n_components=2):
        self.pca = PCA(n_components=n_components)
        components = self.pca.fit_transform(self.df)
        return components
    
    def plot_variance(self):
        plt.plot(range(1, len(self.pca.explained_variance_ratio_)+1),
                 self.pca.explained_variance_ratio_.cumsum(),
                 marker='o')
        plt.xlabel('Nombre de composants')
        plt.ylabel('Variance cumulée expliquée')
        plt.title('Variance expliquée par PCA')
        plt.show()

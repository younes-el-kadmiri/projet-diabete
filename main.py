import joblib
from core.loader import DataLoader
from core.preprocessor import Preprocessor
from core.clusterer import Clusterer
from core.pca_reducer import PCA_Reducer
from core.classifier import Classifier



# Chargement
loader = DataLoader("data/processed/diabetes_clean_no_outliers.csv")
df = loader.load()

# Prétraitement
preproc = Preprocessor(df)
preproc.handle_missing()
preproc.remove_outliers()
preproc.select_features(["Pregnancies", "Glucose", "BMI", "Age", "DiabetesPedigreeFunction"])
df_processed = preproc.get_data()

# Clustering
clusterer = Clusterer(df_processed)
clusterer.scale_data()
clusterer.elbow_method()
df_clustered = clusterer.fit_kmeans(n_clusters=2)
print(clusterer.cluster_means())

# PCA pour visualiser
pca = PCA_Reducer(df_processed)
components = pca.fit_transform()
pca.plot_variance()

# Classification supervisée (avec clusters comme cibles)
classifier = Classifier(df_clustered)
classifier.prepare_data(df_processed.columns.tolist())
classifier.train_and_compare()


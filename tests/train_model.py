import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # ⚠️ Avant pyplot !
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# 1. Chargement données
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'processed', 'diabetes_clean_no_outliers.csv')

df = pd.read_csv(DATA_PATH)
print(f"Données chargées : {df.shape}")

# 2. Sélection features importantes (sans Outcome)
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age"]
X = df[features]

# 3. Mise à l'échelle
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Méthode du coude pour choisir k optimal
inertias = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Nombre de clusters k')
plt.ylabel('Inertie')
plt.title('Méthode du coude pour choisir k')
plt.grid(True)

output_dir = os.path.join(BASE_DIR, '..', 'outputs')
os.makedirs(output_dir, exist_ok=True)

plt.savefig(os.path.join(output_dir, "cluster_elbow_plot.png"))
plt.close()

# 5. Clustering avec k choisi (exemple k=2)
k_optimal = 2
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Ajout des clusters dans le DataFrame
df['Cluster'] = clusters

# 6. Réduction dimensionnelle pour visualiser les clusters
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['Cluster'] = clusters

plt.figure(figsize=(8,6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='Cluster', palette='Set1')
plt.title('Visualisation des clusters avec PCA')
plt.savefig(os.path.join(output_dir, "PCA_clusters.png"))
plt.close()

# 7. Analyse des clusters
print("Moyennes des caractéristiques par cluster:")
print(df.groupby('Cluster')[features].mean())
print("\nEffectifs par cluster:")
print(df['Cluster'].value_counts())

# 8. Sauvegarde des modèles scaler et kmeans + PCA
model_dir = os.path.join(BASE_DIR, '..', 'model')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
joblib.dump(kmeans, os.path.join(model_dir, "kmeans_model.pkl"))
joblib.dump(pca, os.path.join(model_dir, "pca_model.pkl"))

print("Modèles scaler, kmeans et PCA sauvegardés avec succès.")

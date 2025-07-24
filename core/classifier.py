from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
import pandas as pd

class Classifier:
    def __init__(self, df, target_col='Cluster'):
        self.df = df.copy()
        self.target_col = target_col
    
    def prepare_data(self, feature_cols):
        X = self.df[feature_cols]
        y = self.df[self.target_col]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
    
    def train_and_compare(self):
        models = {
            'RandomForest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42),
            'GradientBoosting': GradientBoostingClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }
        results = []
        for name, model in models.items():
            print(f"--- Entraînement et évaluation du modèle : {name} ---")
            model.fit(self.X_train, self.y_train)
            preds = model.predict(self.X_test)
            print(classification_report(self.y_test, preds))
            f1 = f1_score(self.y_test, preds, average='weighted')
            results.append({'model': name, 'f1_score': f1})
        
        df_results = pd.DataFrame(results).sort_values(by='f1_score', ascending=False)
        print("Classement des modèles selon le F1-score :")
        print(df_results)
        return df_results

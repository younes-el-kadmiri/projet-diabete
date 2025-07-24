import pandas as pd
from scipy import stats

class Preprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def drop_columns(self, cols):
        self.df.drop(columns=cols, inplace=True)
    
    def handle_missing(self):
        print("Valeurs manquantes par colonne :")
        print(self.df.isnull().sum())
        # Si tu veux : imputer, ou drop NA
        self.df.dropna(inplace=True)
    
    def remove_outliers(self, z_thresh=3):
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        z_scores = stats.zscore(self.df[numeric_cols])
        abs_z = abs(z_scores)
        filtered_entries = (abs_z < z_thresh).all(axis=1)
        self.df = self.df[filtered_entries]
        print(f"Après suppression des outliers : {self.df.shape[0]} lignes")
    
    def select_features(self, features):
        self.df = self.df[features]
    
    def get_data(self):
        return self.df

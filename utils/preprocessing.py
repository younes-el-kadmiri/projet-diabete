import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

def scale_features(df, features, scaler=None):
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])
    else:
        scaled_data = scaler.transform(df[features])
    return scaled_data


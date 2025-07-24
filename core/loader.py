import pandas as pd

class DataLoader:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None

    def load(self):
        self.df = pd.read_csv(self.filepath)
        return self.df

    def drop_columns(self, cols):
        if self.df is not None:
            self.df = self.df.drop(columns=cols)
        return self.df

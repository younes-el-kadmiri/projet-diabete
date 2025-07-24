import unittest
import pandas as pd
from core.preprocessor import Preprocessor

class TestPreprocessor(unittest.TestCase):
    def test_remove_outliers_iqr(self):
        data = {'A': [1,2,3,1000,5], 'B':[10,15,10,20,25]}
        df = pd.DataFrame(data)
        preproc = Preprocessor(df)
        df_clean = preproc.remove_outliers_iqr(['A'])
        self.assertTrue(df_clean['A'].max() < 1000)

if __name__ == '__main__':
    unittest.main()

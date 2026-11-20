import pandas as pd
import numpy as np

def encode_categorical(df):
    for col in df.columns:
        if df[col].dtype == object:
            # Convert string column to category codes (NaNs become -1)
            codes = df[col].astype('category').cat.codes
            # Put NaNs back to NaN so the network's NaN-handling logic still works
            df[col] = np.where(codes == -1, np.nan, codes)
    return df

print("Fixing clinical.csv...")
clinical = pd.read_csv('data/raw/clinical.csv', index_col=0)
clinical = encode_categorical(clinical)
clinical.to_csv('data/raw/clinical.csv')

print("Fixing cognitive.csv...")
cognitive = pd.read_csv('data/raw/cognitive.csv', index_col=0)
cognitive = encode_categorical(cognitive)
cognitive.to_csv('data/raw/cognitive.csv')

print("Done! Strings converted to numbers.")

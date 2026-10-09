import pandas as pd
import numpy as np 

def load_data(file_path):
    df = pd.read_csv(file_path)

    return df

file_path = "Data/insurance_claims.csv"
df = load_data(file_path)

print(df.head())
print(df.shape)
import pandas as pd
df = pd.read_pickle(r"Data\SYNTHETIC_DATA\data_10min_1.pkl")
df.to_csv("test.csv")
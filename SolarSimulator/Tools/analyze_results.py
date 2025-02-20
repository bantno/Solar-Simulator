import pandas as pd
import datetime

if __name__ == "__main__":
    directory = r"C:\Users\bepstein8\OneDrive - Georgia Institute of Technology\Documents\Research\Solar-Simulator\Data\SYNTHETIC_DATA\lat30\combined_synthetic_data_lat30_lon-90_15min.pkl"
    df = pd.read_pickle(directory)
    print(df)

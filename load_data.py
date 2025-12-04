import pandas as pd

def load_data(file_name):
    csv_folder = "csvs"
    file_path = f"{csv_folder}/{file_name}.csv"
    data = pd.read_csv(file_path)
    return data
import pandas as pd
import os
from collections import defaultdict

def get_all_codes(file_path: str):
    files = os.listdir(file_path)
    codes_files = defaultdict(list)
    for f in files:
        if not f.endswith(".csv") or "-" not in f:
            continue
        code = f.split("-")[1].strip(".csv")
        codes_files[code].append(f)
    return codes_files

def rename_data(file_path: str):
    codes_files = get_all_codes(file_path)
    for code, files in codes_files.items():
        latest_file = None
        latest_date = None
        for f in files:
            try:
                df = pd.read_csv(os.path.join(file_path, f), parse_dates=True, index_col=0)
            except pd.errors.EmptyDataError:
                print(f"{f} is empty, skip")
                continue
                
            if latest_date is None or df.index.max() > latest_date:
                latest_date = df.index.max()
                latest_file = f
        if latest_file:
            os.rename(
                os.path.join(file_path, latest_file),
                os.path.join(file_path, f"{code}.csv")
            )
if __name__ == "__main__":
    data_path = "./data/etf_data"
    rename_data(data_path)

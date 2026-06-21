import os
from dotenv import load_dotenv

load_dotenv()


def _default_data_dir() -> str:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "data")

class DataPath:
    DATA_DIR: str = os.getenv("DATA_DIR") or _default_data_dir()
    DEFAULT_PATH: str = os.getenv("DEFAULT_PATH") or os.path.join(DATA_DIR, "etf_data")
    INDEX_PATH: str = os.getenv("INDEX_PATH") or os.path.join(DATA_DIR, "index")
    BAK_PATH: str = os.getenv("BAK_PATH") or os.path.join(DATA_DIR, "data-bak")
    DEFAULT_WINDOWS_PATH: str = os.getenv("DEFAULT_WINDOWS_PATH") or ""
    CALANDAR_DF: str = os.getenv("CALANDAR_DF") or os.path.join(DATA_DIR, "const", "calandar_df.csv")
    CLUSTERING_DF: str = os.getenv("CLUSTERING_DF") or os.path.join(DATA_DIR, "const", "clustering_with_corr_valid_20260507.xlsx")
    ETF_NAME_LIST_DF: str = os.getenv("ETF_NAME_LIST_DF") or os.path.join(DATA_DIR, "const", "etf_name_list.xlsx")
    INDEX_NAME_LIST_DF: str = os.getenv("INDEX_NAME_LIST_DF") or os.path.join(DATA_DIR, "const", "index_name_list.csv")
    ETF_INDEX_MAP_CSV: str = os.getenv("ETF_INDEX_MAP_CSV") or os.path.join(DATA_DIR, "const", "etf_index_map.csv")

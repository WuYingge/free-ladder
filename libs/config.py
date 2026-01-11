import os
from dotenv import load_dotenv

load_dotenv()

class DataPath:
    DEFAULT_PATH: str = os.getenv("DEFAULT_PATH") # type: ignore
    BAK_PATH: str = os.getenv("BAK_PATH") # type: ignore
    DEFAULT_WINDOWS_PATH: str = os.getenv("DEFAULT_WINDOWS_PATH") # type: ignore
    DATA_DIR: str = os.getenv("DATA_DIR") # type: ignore
    CALANDAR_DF: str = os.getenv("CALANDAR_DF") # type: ignore
    CLUSTERING_DF: str = os.getenv("CLUSTERING_DF") # type: ignore

import os
from dotenv import load_dotenv

load_dotenv()

class DataPath:
    DEFAULT_PATH: str = os.getenv("DEFAULT_PATH")
    BAK_PATH: str = os.getenv("BAK_PATH")
    DEFAULT_WINDOWS_PATH: str = os.getenv("DEFAULT_WINDOWS_PATH")
    DATA_DIR: str = os.getenv("DATA_DIR")
    CALANDAR_DF: str = os.getenv("CALANDAR_DF")

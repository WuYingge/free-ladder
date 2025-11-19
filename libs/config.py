import os
from dotenv import load_dotenv

load_dotenv()

class DataPath:
    DEFAULT_PATH = os.getenv("DEFAULT_PATH")
    BAK_PATH = os.getenv("BAK_PATH")
    DEFAULT_WINDOWS_PATH = os.getenv("DEFAULT_WINDOWS_PATH")

import os

def get_symbol_name_from_fp(fp: str) -> str:
    return os.path.splitext(os.path.basename(fp))[0]

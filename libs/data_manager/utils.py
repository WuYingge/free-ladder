import os

def get_symbol_name_from_fp(fp: str) -> tuple[str, str]:
    pre = os.path.splitext(os.path.basename(fp))[0].split("-")
    name, symbol = pre
    return symbol, name

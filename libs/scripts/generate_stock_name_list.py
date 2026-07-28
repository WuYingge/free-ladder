#!/usr/bin/env python
"""CLI entry point — delegates to data_manager.stock_data_manager.generate_stock_name_list."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_manager.stock_data_manager import generate_stock_name_list

if __name__ == "__main__":
    generate_stock_name_list()
    print("Done.")

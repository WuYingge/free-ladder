"""rebalance_log 后处理工具：将 symbol 映射为 ETF 中文名称。

用法：

    # 作为模块导入
    from libs.scripts.add_etf_names_to_log import add_etf_names_to_rebalance_log
    df = add_etf_names_to_rebalance_log("path/to/rebalance_log.csv")
    print(df)

    # 命令行
    cd /home/gouzi/projects/invest
    uv run python libs/scripts/add_etf_names_to_log.py path/to/rebalance_log.csv

"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ── 确保 libs/ 内的包可以被直接运行脚本时导入 ──
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "libs"))

import pandas as pd

from data_manager.providers.etf_list_provider import ETF_LIST


_VECTORIZED_COLUMNS = ("selected_symbols",)
_DICT_COLUMNS = ("target_weights", "current_weights", "executed_weights_open")


def _parse_json_cell(value) -> list | dict | object:
    """将 JSON 字符串还原为 Python 对象；已是非字符串或解析失败则原样返回。"""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return value


def _symbol_to_name(symbol: str) -> str:
    """映射单个 symbol → ETF 名称，查不到时回退为 symbol 本身。"""
    name = ETF_LIST.get_name(symbol)
    return name or symbol


def _translate_list(value: object) -> object:
    """将列表中的 symbol 替换为名称。非列表原样返回。"""
    if not isinstance(value, list):
        return value
    return [_symbol_to_name(s) for s in value]


def _translate_dict_keys(value: object) -> object:
    """将字典的 key symbol 替换为名称。非字典原样返回。"""
    if not isinstance(value, dict):
        return value
    return {_symbol_to_name(k): v for k, v in value.items()}


def add_etf_names_to_rebalance_log(
    path_or_df: str | Path | pd.DataFrame,
    *,
    map_vector_columns: bool = True,
    map_dict_keys: bool = True,
) -> pd.DataFrame:
    """读取 rebalance_log.csv，将 symbol 映射为 ETF 中文名称。

    Parameters
    ----------
    path_or_df
        rebalance_log.csv 路径或已加载的 DataFrame。
    map_vector_columns
        是否转换 selected_symbols 列表中的 symbol 为名称（新增 selected_names 列）。
    map_dict_keys
        是否转换 target_weights/current_weights/executed_weights_open 中的 key 为名称
        （新增 _named 后缀列）。

    Returns
    -------
    pd.DataFrame
        包含原始列 + selected_names（及 _named 列）的 DataFrame。
    """
    if isinstance(path_or_df, (str, Path)):
        df = pd.read_csv(path_or_df, dtype=str, encoding="utf-8-sig")
    elif isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        raise TypeError(f"expect str/path/DataFrame, got {type(path_or_df)}")

    if df.empty:
        return df

    # ── 解析 JSON 单元格 ──
    for col in _VECTORIZED_COLUMNS + _DICT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(_parse_json_cell)

    # ── selected_symbols → selected_names ──
    if map_vector_columns and "selected_symbols" in df.columns:
        df["selected_names"] = df["selected_symbols"].apply(_translate_list)

    # ── 字典列的 key 替换为名称 ──
    if map_dict_keys:
        for col in _DICT_COLUMNS:
            if col not in df.columns:
                continue
            named_col = f"{col}_named"
            df[named_col] = df[col].apply(_translate_dict_keys)

    return df


# ====================================================================
# CLI 入口
# ====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="为 rebalance_log.csv 添加 ETF 中文名称列",
    )
    parser.add_argument(
        "path",
        help="rebalance_log.csv 文件路径",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出 CSV 路径（默认: 在输入文件名后追加 _named 后缀）",
    )
    parser.add_argument(
        "--no-dict-keys",
        action="store_true",
        help="不转换 target_weights 等字典列的 key（仅转换 selected_symbols）",
    )
    args = parser.parse_args()

    input_path = Path(args.path)
    if not input_path.exists():
        raise SystemExit(f"文件不存在: {input_path}")

    output_path = Path(args.output) if args.output else input_path.with_stem(input_path.stem + "_named")

    df = add_etf_names_to_rebalance_log(
        input_path,
        map_dict_keys=not args.no_dict_keys,
    )
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"已输出 {len(df)} 行 → {output_path}")


if __name__ == "__main__":
    main()

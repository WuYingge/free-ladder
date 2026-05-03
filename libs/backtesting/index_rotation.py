from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data_manager.etf_data_manager import get_etf_data_by_symbol
from data_manager.providers.etf_list_provider import ETF_LIST
from data_manager.utils import (
    INDEX_KEYWORDS,
    extract_index_tokens,
    extract_tracked_index_name,
    normalize_tracked_index_name,
)
from factors.portfolio.low_correlation import validate_and_prepare_correlation_matrix


@dataclass(slots=True)
class IndexRotationDiscoveryResult:
    etf_profile_frame: pd.DataFrame
    keyword_selection_frame: pd.DataFrame
    extension_candidate_frame: pd.DataFrame
    base_conflict_frame: pd.DataFrame
    base_pairwise_correlation_frame: pd.DataFrame
    base_is_feasible: bool
    correlation_matrix: pd.DataFrame
    combo_frame: pd.DataFrame
    combo_symbol_lists: list[list[str]]
    correlation_start: pd.Timestamp | None
    correlation_end: pd.Timestamp | None
    correlation_bar_count: int


@dataclass(slots=True)
class KeywordRepresentativeResult:
    etf_profile_frame: pd.DataFrame
    keyword_selection_frame: pd.DataFrame
    representative_etf_frame: pd.DataFrame


def match_index_keywords(
    index_name: str,
    keywords: Sequence[str] = INDEX_KEYWORDS,
) -> tuple[str, ...]:
    normalized_index = normalize_tracked_index_name(index_name)
    if not normalized_index:
        return tuple()

    normalized_tokens = {
        normalize_tracked_index_name(token)
        for token in extract_index_tokens(index_name)
        if normalize_tracked_index_name(token)
    }

    matched: list[str] = []
    for keyword in keywords:
        normalized_keyword = normalize_tracked_index_name(keyword)
        if not normalized_keyword:
            continue
        if (
            normalized_keyword in normalized_index
            or normalized_keyword in normalized_tokens
        ):
            matched.append(keyword)
    return tuple(dict.fromkeys(matched))


def select_keyword_representatives(
    etf_profile_frame: pd.DataFrame,
    *,
    keywords: Sequence[str] = INDEX_KEYWORDS,
    min_bar_count: int = 600,
) -> pd.DataFrame:
    if etf_profile_frame.empty:
        return pd.DataFrame(
            {
                "keyword": list(keywords),
                "candidate_count": [0] * len(keywords),
                "selected_symbol": [None] * len(keywords),
            }
        )

    required_cols = {
        "symbol",
        "name",
        "type",
        "tracked_index",
        "matched_keywords",
        "bar_count",
        "avg_value",
        "first_date",
        "last_date",
    }
    missing_cols = sorted(required_cols - set(etf_profile_frame.columns))
    if missing_cols:
        raise ValueError(f"etf_profile_frame missing required columns: {missing_cols}")

    eligible = etf_profile_frame.copy()
    eligible = eligible[
        eligible["bar_count"].fillna(0).astype(int) > int(min_bar_count)
    ].copy()
    eligible["avg_value"] = pd.to_numeric(eligible["avg_value"], errors="coerce")
    eligible = eligible.dropna(subset=["avg_value"])

    rows: list[dict[str, Any]] = []
    for keyword in keywords:
        keyword_mask = eligible["matched_keywords"].apply(
            lambda values: keyword in _coerce_tuple(values)
        )
        candidates = eligible[keyword_mask].copy()
        candidates = candidates.sort_values(
            ["avg_value", "bar_count", "symbol"],
            ascending=[False, False, True],
        )

        row: dict[str, Any] = {
            "keyword": keyword,
            "candidate_count": int(len(candidates)),
            "selected_symbol": None,
            "selected_name": None,
            "selected_type": None,
            "tracked_index": None,
            "matched_keywords": tuple(),
            "bar_count": None,
            "avg_value": None,
            "first_date": None,
            "last_date": None,
        }
        if not candidates.empty:
            best = candidates.iloc[0]
            row.update(
                {
                    "selected_symbol": str(best["symbol"]),
                    "selected_name": best["name"],
                    "selected_type": best["type"],
                    "tracked_index": best["tracked_index"],
                    "matched_keywords": _coerce_tuple(best["matched_keywords"]),
                    "bar_count": int(best["bar_count"]),
                    "avg_value": float(best["avg_value"]),
                    "first_date": best["first_date"],
                    "last_date": best["last_date"],
                }
            )
        rows.append(row)

    return pd.DataFrame(rows)


def build_representative_etf_frame(
    keyword_selection_frame: pd.DataFrame,
    *,
    exclude_symbols: Sequence[str] = (),
) -> pd.DataFrame:
    selected = keyword_selection_frame.dropna(subset=["selected_symbol"]).copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "type",
                "tracked_index",
                "bar_count",
                "avg_value",
                "first_date",
                "last_date",
                "source_keywords",
                "source_keyword_count",
                "source_keywords_text",
            ]
        )

    excluded = {str(symbol) for symbol in exclude_symbols}
    if excluded:
        selected = selected[~selected["selected_symbol"].astype(str).isin(excluded)].copy()
    if selected.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "name",
                "type",
                "tracked_index",
                "bar_count",
                "avg_value",
                "first_date",
                "last_date",
                "source_keywords",
                "source_keyword_count",
                "source_keywords_text",
            ]
        )

    grouped_rows: list[dict[str, Any]] = []
    for symbol, group in selected.groupby(selected["selected_symbol"].astype(str), sort=False):
        first = group.iloc[0]
        source_keywords = tuple(group["keyword"].astype(str).tolist())
        grouped_rows.append(
            {
                "symbol": symbol,
                "name": first["selected_name"],
                "type": first["selected_type"],
                "tracked_index": first["tracked_index"],
                "bar_count": first["bar_count"],
                "avg_value": first["avg_value"],
                "first_date": first["first_date"],
                "last_date": first["last_date"],
                "source_keywords": source_keywords,
                "source_keyword_count": len(source_keywords),
                "source_keywords_text": ", ".join(source_keywords),
            }
        )

    frame = pd.DataFrame(grouped_rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["avg_value", "bar_count", "symbol"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def build_pairwise_correlation_frame(
    corr_matrix: pd.DataFrame,
    *,
    symbols: Sequence[str],
    corr_threshold: float | None = None,
    corr_lower_bound: float | None = None,
) -> pd.DataFrame:
    prepared, _ = validate_and_prepare_correlation_matrix(corr_matrix)
    ordered_symbols = list(dict.fromkeys(map(str, symbols)))
    missing_symbols = [symbol for symbol in ordered_symbols if symbol not in prepared.index]
    if missing_symbols:
        raise ValueError(f"symbols not found in correlation matrix: {missing_symbols}")

    columns = ["left_symbol", "right_symbol", "correlation"]
    if corr_threshold is not None:
        columns.append("passes_threshold")

    if len(ordered_symbols) < 2:
        return pd.DataFrame(columns=columns)

    pairwise_frame = pd.DataFrame(
        _pairwise_corr_records(
            prepared.loc[ordered_symbols, ordered_symbols],
            ordered_symbols,
        )
    )
    pairwise_frame["correlation"] = pd.to_numeric(
        pairwise_frame["correlation"],
        errors="coerce",
    )
    if corr_threshold is not None:
        pairwise_frame["passes_threshold"] = pairwise_frame["correlation"].apply(
            lambda corr_value: _is_correlation_compatible(
                corr_value,
                corr_threshold=corr_threshold,
                corr_lower_bound=corr_lower_bound,
            )
        )

    return pairwise_frame.sort_values(
        ["correlation", "left_symbol", "right_symbol"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def discover_index_keyword_representatives(
    *,
    keyword_list: Sequence[str] = INDEX_KEYWORDS,
    candidate_symbols: Sequence[str] | None = None,
    min_bar_count: int = 600,
    exclude_symbols: Sequence[str] = (),
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> KeywordRepresentativeResult:
    all_candidate_symbols = (
        list(candidate_symbols)
        if candidate_symbols is not None
        else ETF_LIST.get_all_symbol()
    )
    profile_frame, _ = _load_profile_frame_and_close_map(
        all_candidate_symbols,
        keywords=keyword_list,
        progress_callback=progress_callback,
    )
    keyword_selection_frame = select_keyword_representatives(
        profile_frame,
        keywords=keyword_list,
        min_bar_count=min_bar_count,
    )
    representative_etf_frame = build_representative_etf_frame(
        keyword_selection_frame,
        exclude_symbols=exclude_symbols,
    )
    return KeywordRepresentativeResult(
        etf_profile_frame=profile_frame,
        keyword_selection_frame=keyword_selection_frame,
        representative_etf_frame=representative_etf_frame,
    )


def search_low_correlation_combos(
    corr_matrix: pd.DataFrame,
    *,
    base_symbols: Sequence[str],
    extension_symbols: Sequence[str] | None = None,
    corr_threshold: float = 0.3,
    corr_lower_bound: float | None = None,
    max_results: int | None = 5,
    extension_metric_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[list[str]], pd.DataFrame]:
    prepared, _ = validate_and_prepare_correlation_matrix(corr_matrix)
    base_symbols = [str(symbol) for symbol in base_symbols]
    missing_base = [symbol for symbol in base_symbols if symbol not in prepared.index]
    if missing_base:
        raise ValueError(f"base symbols not found in correlation matrix: {missing_base}")

    if extension_symbols is None:
        extension_symbols = [
            symbol for symbol in prepared.index.tolist() if symbol not in set(base_symbols)
        ]
    else:
        extension_symbols = [
            str(symbol)
            for symbol in extension_symbols
            if str(symbol) in prepared.index and str(symbol) not in set(base_symbols)
        ]

    base_pairwise_frame = build_pairwise_correlation_frame(
        prepared,
        symbols=base_symbols,
        corr_threshold=corr_threshold,
        corr_lower_bound=corr_lower_bound,
    )
    incompatible_base_pairs = []
    if not base_pairwise_frame.empty:
        incompatible_base_pairs = base_pairwise_frame.loc[
            ~base_pairwise_frame["passes_threshold"].astype(bool)
        ].to_dict("records")

    base_conflict_frame = _build_base_conflict_frame(
        prepared,
        base_symbols=base_symbols,
        extension_symbols=extension_symbols,
        corr_threshold=corr_threshold,
        corr_lower_bound=corr_lower_bound,
        extension_metric_frame=extension_metric_frame,
    )

    if incompatible_base_pairs:
        combo_frame = pd.DataFrame(
            {
                "combo_label": [],
                "symbols": [],
                "symbols_text": [],
                "symbol_count": [],
                "base_symbols": [],
                "extension_symbols": [],
                "extension_symbol_count": [],
                "avg_pairwise_correlation": [],
                "max_pairwise_correlation": [],
                "min_pairwise_correlation": [],
                "extension_avg_value": [],
                "extension_total_avg_value": [],
                "source_keywords": [],
                "source_keywords_text": [],
            }
        )
        return combo_frame, [], base_conflict_frame

    base_compatible_extensions = base_conflict_frame.loc[
        base_conflict_frame["compatible_with_base"], "symbol"
    ].astype(str).tolist()

    adjacency = _build_compatibility_graph(
        prepared,
        base_compatible_extensions,
        corr_threshold=corr_threshold,
        corr_lower_bound=corr_lower_bound,
    )
    extension_sets = _find_max_compatible_sets(base_compatible_extensions, adjacency)

    metric_lookup = None
    if extension_metric_frame is not None and not extension_metric_frame.empty:
        if "symbol" in extension_metric_frame.columns:
            metric_lookup = extension_metric_frame.set_index("symbol", drop=False)
        else:
            metric_lookup = extension_metric_frame.copy()

    combo_rows: list[dict[str, Any]] = []
    combo_symbol_lists: list[list[str]] = []
    for combo_index, extension_set in enumerate(extension_sets):
        ordered_extensions = _order_extension_symbols(extension_set, metric_lookup)
        symbols = [*base_symbols, *ordered_extensions]
        combo_symbol_lists.append(symbols)
        summary = _summarize_combo(
            prepared,
            symbols=symbols,
            base_symbols=base_symbols,
            extension_symbols=ordered_extensions,
            metric_lookup=metric_lookup,
        )
        summary["_combo_index"] = combo_index
        combo_rows.append(summary)

    combo_frame = pd.DataFrame(combo_rows)
    if combo_frame.empty:
        return combo_frame, combo_symbol_lists, base_conflict_frame

    combo_frame = combo_frame.sort_values(
        ["symbol_count", "avg_pairwise_correlation", "extension_avg_value", "symbols_text"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    combo_frame.insert(
        0,
        "combo_label",
        [f"combo_{idx:02d}" for idx in range(1, len(combo_frame) + 1)],
    )

    if max_results is not None:
        combo_frame = combo_frame.head(int(max_results)).reset_index(drop=True)
    combo_symbol_lists = [
        combo_symbol_lists[int(combo_idx)]
        for combo_idx in combo_frame["_combo_index"].tolist()
    ]
    combo_frame = combo_frame.drop(columns=["_combo_index"])
    return combo_frame, combo_symbol_lists, base_conflict_frame


def discover_index_rotation_candidates(
    *,
    base_symbols: Sequence[str],
    keyword_list: Sequence[str] = INDEX_KEYWORDS,
    candidate_symbols: Sequence[str] | None = None,
    min_bar_count: int = 600,
    corr_threshold: float = 0.3,
    corr_lower_bound: float | None = None,
    correlation_source: str = "close",
    start_date: str | None = None,
    end_date: str | None = None,
    max_results: int | None = 5,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> IndexRotationDiscoveryResult:
    all_candidate_symbols = list(candidate_symbols) if candidate_symbols is not None else ETF_LIST.get_all_symbol()
    profile_frame, close_map = _load_profile_frame_and_close_map(
        all_candidate_symbols,
        keywords=keyword_list,
        progress_callback=progress_callback,
    )
    keyword_selection_frame = select_keyword_representatives(
        profile_frame,
        keywords=keyword_list,
        min_bar_count=min_bar_count,
    )
    extension_candidate_frame = build_representative_etf_frame(
        keyword_selection_frame,
        exclude_symbols=base_symbols,
    )

    missing_base_symbols = [symbol for symbol in base_symbols if symbol not in close_map]
    if missing_base_symbols:
        base_profile_frame, base_close_map = _load_profile_frame_and_close_map(
            missing_base_symbols,
            keywords=keyword_list,
            progress_callback=None,
        )
        close_map.update(base_close_map)
        if not base_profile_frame.empty:
            profile_frame = pd.concat([profile_frame, base_profile_frame], axis=0, ignore_index=True)
            profile_frame = profile_frame.drop_duplicates(subset=["symbol"], keep="first")

    symbols_for_corr = [*map(str, base_symbols), *extension_candidate_frame["symbol"].astype(str).tolist()]
    symbols_for_corr = list(dict.fromkeys(symbols_for_corr))
    correlation_frame, corr_start, corr_end, corr_bar_count = _build_common_correlation_frame(
        symbols_for_corr,
        close_map,
        start_date=start_date,
        end_date=end_date,
        correlation_source=correlation_source,
    )
    corr_matrix = correlation_frame.corr(method="pearson")
    corr_matrix, _ = validate_and_prepare_correlation_matrix(corr_matrix)
    base_pairwise_correlation_frame = build_pairwise_correlation_frame(
        corr_matrix,
        symbols=base_symbols,
        corr_threshold=corr_threshold,
        corr_lower_bound=corr_lower_bound,
    )
    base_is_feasible = bool(
        base_pairwise_correlation_frame["passes_threshold"].all()
    ) if not base_pairwise_correlation_frame.empty else True

    combo_frame, combo_symbol_lists, base_conflict_frame = search_low_correlation_combos(
        corr_matrix,
        base_symbols=base_symbols,
        extension_symbols=extension_candidate_frame["symbol"].astype(str).tolist(),
        corr_threshold=corr_threshold,
        corr_lower_bound=corr_lower_bound,
        max_results=max_results,
        extension_metric_frame=extension_candidate_frame,
    )

    return IndexRotationDiscoveryResult(
        etf_profile_frame=profile_frame,
        keyword_selection_frame=keyword_selection_frame,
        extension_candidate_frame=extension_candidate_frame,
        base_conflict_frame=base_conflict_frame,
        base_pairwise_correlation_frame=base_pairwise_correlation_frame,
        base_is_feasible=base_is_feasible,
        correlation_matrix=corr_matrix,
        combo_frame=combo_frame,
        combo_symbol_lists=combo_symbol_lists,
        correlation_start=corr_start,
        correlation_end=corr_end,
        correlation_bar_count=corr_bar_count,
    )


def _load_profile_frame_and_close_map(
    symbols: Sequence[str],
    *,
    keywords: Sequence[str],
    progress_callback: Callable[[int, int, str], None] | None,
) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    rows: list[dict[str, Any]] = []
    close_map: dict[str, pd.Series] = {}
    total = len(symbols)
    for idx, raw_symbol in enumerate(symbols, start=1):
        symbol = str(raw_symbol).zfill(6)
        try:
            etf_data = get_etf_data_by_symbol(symbol)
            data = etf_data.data.copy()
            close_series = _extract_price_series(data, "close")
            value_series = pd.to_numeric(data.get("value"), errors="coerce")
            avg_value = float(value_series.dropna().mean()) if value_series is not None and not value_series.dropna().empty else float("nan")
            tracked_index = extract_tracked_index_name(etf_data.name or ETF_LIST.get_name(symbol))
            matched_keywords = match_index_keywords(tracked_index, keywords)
            close_map[symbol] = close_series
            rows.append(
                {
                    "symbol": symbol,
                    "name": etf_data.name or ETF_LIST.get_name(symbol),
                    "type": ETF_LIST.get_type(symbol),
                    "tracked_index": tracked_index,
                    "matched_keywords": matched_keywords,
                    "matched_keyword_count": len(matched_keywords),
                    "bar_count": int(len(close_series)),
                    "avg_value": avg_value,
                    "first_date": close_series.index.min() if not close_series.empty else pd.NaT,
                    "last_date": close_series.index.max() if not close_series.empty else pd.NaT,
                }
            )
        except Exception:
            pass
        if progress_callback is not None:
            progress_callback(idx, total, symbol)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["avg_value", "bar_count", "symbol"], ascending=[False, False, True]).reset_index(drop=True)
    return frame, close_map


def _build_common_close_frame(
    symbols: Sequence[str],
    close_map: dict[str, pd.Series],
    *,
    start_date: str | None,
    end_date: str | None,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None, int]:
    missing_symbols = [symbol for symbol in symbols if symbol not in close_map]
    if missing_symbols:
        raise ValueError(f"missing close price series for symbols: {missing_symbols}")
    if not symbols:
        raise ValueError("at least one symbol is required to build close frame")

    start_ts = pd.to_datetime(start_date) if start_date else None
    end_ts = pd.to_datetime(end_date) if end_date else None
    common_index: pd.DatetimeIndex | None = None
    prepared_series: dict[str, pd.Series] = {}
    for symbol in symbols:
        series = close_map[symbol].copy()
        if start_ts is not None:
            series = series[series.index >= start_ts]
        if end_ts is not None:
            series = series[series.index <= end_ts]
        if series.empty:
            raise ValueError(f"close price series is empty after date filtering for symbol {symbol}")
        prepared_series[symbol] = series
        common_index = series.index if common_index is None else common_index.intersection(series.index)

    assert common_index is not None
    common_index = common_index.sort_values()
    if len(common_index) < 2:
        raise ValueError("common close price window has fewer than 2 bars")

    frame = pd.DataFrame(
        {symbol: prepared_series[symbol].reindex(common_index) for symbol in symbols},
        index=common_index,
    )
    if frame.isna().any().any():
        raise ValueError("common close price frame contains NaN values")
    return frame, common_index.min(), common_index.max(), int(len(common_index))


def _build_base_conflict_frame(
    corr_matrix: pd.DataFrame,
    *,
    base_symbols: Sequence[str],
    extension_symbols: Sequence[str],
    corr_threshold: float,
    corr_lower_bound: float | None,
    extension_metric_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    metric_lookup = None
    if extension_metric_frame is not None and not extension_metric_frame.empty:
        metric_lookup = extension_metric_frame.set_index("symbol", drop=False)

    rows: list[dict[str, Any]] = []
    for symbol in extension_symbols:
        corr_to_base = corr_matrix.loc[symbol, list(base_symbols)].astype(float)
        conflicting = corr_to_base[
            corr_to_base.apply(
                lambda corr_value: not _is_correlation_compatible(
                    corr_value,
                    corr_threshold=corr_threshold,
                    corr_lower_bound=corr_lower_bound,
                )
            )
        ]
        row: dict[str, Any] = {
            "symbol": symbol,
            "compatible_with_base": bool(conflicting.empty),
            "conflicting_base_symbols": tuple(conflicting.index.astype(str).tolist()),
            "max_correlation_to_base": float(corr_to_base.max()),
            "min_correlation_to_base": float(corr_to_base.min()),
        }
        if metric_lookup is not None and symbol in metric_lookup.index:
            metric_row = metric_lookup.loc[symbol]
            row.update(
                {
                    "name": metric_row.get("name"),
                    "tracked_index": metric_row.get("tracked_index"),
                    "avg_value": metric_row.get("avg_value"),
                    "source_keywords": _coerce_tuple(metric_row.get("source_keywords")),
                }
            )
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["compatible_with_base", "max_correlation_to_base", "symbol"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def _build_compatibility_graph(
    corr_matrix: pd.DataFrame,
    symbols: Sequence[str],
    *,
    corr_threshold: float,
    corr_lower_bound: float | None,
) -> dict[str, set[str]]:
    ordered_symbols = list(dict.fromkeys(map(str, symbols)))
    adjacency: dict[str, set[str]] = {symbol: set() for symbol in ordered_symbols}
    for idx, left_symbol in enumerate(ordered_symbols):
        for right_symbol in ordered_symbols[idx + 1 :]:
            corr_value = corr_matrix.loc[left_symbol, right_symbol]
            if _is_correlation_compatible(
                corr_value,
                corr_threshold=corr_threshold,
                corr_lower_bound=corr_lower_bound,
            ):
                adjacency[left_symbol].add(right_symbol)
                adjacency[right_symbol].add(left_symbol)
    return adjacency


def _build_common_correlation_frame(
    symbols: Sequence[str],
    close_map: dict[str, pd.Series],
    *,
    start_date: str | None,
    end_date: str | None,
    correlation_source: str,
) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None, int]:
    close_frame, _, _, _ = _build_common_close_frame(
        symbols,
        close_map,
        start_date=start_date,
        end_date=end_date,
    )

    if correlation_source == "close":
        return close_frame, close_frame.index.min(), close_frame.index.max(), int(len(close_frame))

    if correlation_source == "return":
        return_frame = close_frame.pct_change().dropna(how="any")
        if len(return_frame) < 2:
            raise ValueError("common return frame has fewer than 2 bars")
        return (
            return_frame,
            return_frame.index.min(),
            return_frame.index.max(),
            int(len(return_frame)),
        )

    raise ValueError(f"unsupported correlation_source: {correlation_source}")


def _is_correlation_compatible(
    corr_value: Any,
    *,
    corr_threshold: float,
    corr_lower_bound: float | None,
) -> bool:
    if not pd.notna(corr_value):
        return False

    corr_float = float(corr_value)
    if corr_lower_bound is not None and corr_float <= float(corr_lower_bound):
        return False
    return corr_float < float(corr_threshold)


def _find_max_compatible_sets(
    symbols: Sequence[str],
    adjacency: dict[str, set[str]],
) -> list[tuple[str, ...]]:
    ordered_symbols = sorted(dict.fromkeys(map(str, symbols)))
    if not ordered_symbols:
        return [tuple()]

    best_size = 0
    best_sets: set[tuple[str, ...]] = set()

    def visit(clique: tuple[str, ...], candidates: tuple[str, ...]) -> None:
        nonlocal best_size, best_sets
        if len(clique) + len(candidates) < best_size:
            return
        if not candidates:
            clique_key = tuple(sorted(clique))
            if len(clique_key) > best_size:
                best_size = len(clique_key)
                best_sets = {clique_key}
            elif len(clique_key) == best_size:
                best_sets.add(clique_key)
            return

        for idx, symbol in enumerate(candidates):
            remaining = tuple(
                candidate
                for candidate in candidates[idx + 1 :]
                if candidate in adjacency.get(symbol, set())
            )
            visit((*clique, symbol), remaining)
            if len(clique) + len(candidates) - idx - 1 < best_size:
                break

        clique_key = tuple(sorted(clique))
        if len(clique_key) > best_size:
            best_size = len(clique_key)
            best_sets = {clique_key}
        elif len(clique_key) == best_size:
            best_sets.add(clique_key)

    visit(tuple(), tuple(ordered_symbols))
    return sorted(best_sets)


def _order_extension_symbols(
    symbols: Sequence[str],
    metric_lookup: pd.DataFrame | None,
) -> list[str]:
    if metric_lookup is None or metric_lookup.empty:
        return sorted(map(str, symbols))
    return sorted(
        map(str, symbols),
        key=lambda symbol: (
            -float(metric_lookup.loc[symbol, "avg_value"]) if symbol in metric_lookup.index and pd.notna(metric_lookup.loc[symbol, "avg_value"]) else float("inf"),
            symbol,
        ),
    )


def _summarize_combo(
    corr_matrix: pd.DataFrame,
    *,
    symbols: Sequence[str],
    base_symbols: Sequence[str],
    extension_symbols: Sequence[str],
    metric_lookup: pd.DataFrame | None,
) -> dict[str, Any]:
    sub_corr = corr_matrix.loc[list(symbols), list(symbols)]
    pairwise = _pairwise_values(sub_corr)
    if pairwise.empty:
        avg_pairwise = 0.0
        max_pairwise = 0.0
        min_pairwise = 0.0
    else:
        avg_pairwise = float(pairwise.mean())
        max_pairwise = float(pairwise.max())
        min_pairwise = float(pairwise.min())

    extension_values: list[float] = []
    source_keywords: list[str] = []
    if metric_lookup is not None and not metric_lookup.empty:
        for symbol in extension_symbols:
            if symbol not in metric_lookup.index:
                continue
            avg_value = metric_lookup.loc[symbol, "avg_value"]
            if pd.notna(avg_value):
                extension_values.append(float(avg_value))
            source_keywords.extend(map(str, _coerce_tuple(metric_lookup.loc[symbol].get("source_keywords"))))

    unique_keywords = tuple(dict.fromkeys(source_keywords))
    extension_avg_value = float(sum(extension_values) / len(extension_values)) if extension_values else 0.0
    extension_total_avg_value = float(sum(extension_values)) if extension_values else 0.0

    return {
        "symbols": list(symbols),
        "symbols_text": ", ".join(symbols),
        "symbol_count": len(symbols),
        "base_symbols": list(base_symbols),
        "extension_symbols": list(extension_symbols),
        "extension_symbol_count": len(extension_symbols),
        "avg_pairwise_correlation": avg_pairwise,
        "max_pairwise_correlation": max_pairwise,
        "min_pairwise_correlation": min_pairwise,
        "extension_avg_value": extension_avg_value,
        "extension_total_avg_value": extension_total_avg_value,
        "source_keywords": unique_keywords,
        "source_keywords_text": ", ".join(unique_keywords),
    }


def _pairwise_values(corr_matrix: pd.DataFrame) -> pd.Series:
    if corr_matrix.empty or len(corr_matrix) < 2:
        return pd.Series(dtype=float)
    values = corr_matrix.to_numpy(dtype=float)
    upper = values[np.triu_indices(len(corr_matrix), 1)]
    return pd.Series(upper, dtype=float)


def _pairwise_corr_records(
    corr_matrix: pd.DataFrame,
    symbols: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, left_symbol in enumerate(symbols):
        for right_symbol in symbols[idx + 1 :]:
            rows.append(
                {
                    "left_symbol": left_symbol,
                    "right_symbol": right_symbol,
                    "correlation": corr_matrix.loc[left_symbol, right_symbol],
                }
            )
    return rows


def _extract_price_series(data: pd.DataFrame, price_col: str) -> pd.Series:
    if price_col not in data.columns:
        raise ValueError(f"missing required price column: {price_col}")
    if "date" in data.columns:
        date_index = pd.to_datetime(data["date"], errors="coerce")
    else:
        date_index = pd.to_datetime(data.index, errors="coerce")
    series = pd.Series(
        pd.to_numeric(data[price_col], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(date_index, name="date"),
        dtype=float,
    )
    series = series[series.index.notna()].sort_index()
    series = series[~series.index.duplicated(keep="last")].dropna()
    if series.empty:
        raise ValueError("price series is empty after normalization")
    return series


def _coerce_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return tuple()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if pd.isna(value):
        return tuple()
    return (value,)


__all__ = [
    "IndexRotationDiscoveryResult",
    "KeywordRepresentativeResult",
    "build_representative_etf_frame",
    "discover_index_keyword_representatives",
    "discover_index_rotation_candidates",
    "match_index_keywords",
    "search_low_correlation_combos",
    "select_keyword_representatives",
]
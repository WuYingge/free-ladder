from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Mapping
import pandas as pd
from core.models.etf_daily_data import EtfData
from data_manager.etf_data_manager import get_etf_data_by_symbol
from factors.average_true_range import AverageTrueRange
from factors.portfolio.correlation import CorrelationFactor

class Portfolio:
    
    def __init__(self, all_money) -> None:
        self._positions: dict[str, Position] = {}
        self._all_money = all_money
        self._postiion_value: Optional[float] = None
    
    @property
    def all_money(self) -> float:
        return self._all_money
    
    @all_money.setter
    def all_money(self, value: float) -> None:
        self._all_money = value
    
    @property
    def positions(self) -> dict[str, Position]:
        return self._positions
    
    def add_position(self, position: Position) -> None:
        self._positions[position.symbol] = position
        self.position_value += position.market_value
        
    def has_symbol(self, symbol: str) -> bool:
        return symbol in self._positions
        
    def add_position_with_symbol_quantities(self, *args: tuple[str, int]) -> None:
        for symbol, quantity in args:
            position = Position(symbol=symbol, quantity=quantity)
            self.add_position(position)
        
    def get_position(self, symbol: str) -> Optional[Position]:
        return self._positions.get(symbol)
        
    @property
    def position_value(self) -> float:
        if self._postiion_value is None:
            self._postiion_value = sum(
                pos.market_value for pos in self._positions.values() if pos.market_value is not None
            )
        return self._postiion_value
    
    @position_value.setter
    def position_value(self, value: float) -> None:
        self._postiion_value = value
        
    def validate(self):
        total_value = sum(
            pos.market_value for pos in self._positions.values() if pos.market_value is not None
        )
        
        # todo use backward-adjustment price may lead to error
        if total_value > self._all_money:
            raise ValueError("Portfolio value exceeds available funds.")
        
    def caculate_unallocated_funds(self) -> float:
        allocated_value = sum(
            pos.market_value for pos in self._positions.values() if pos.market_value is not None
        )
        return self._all_money - allocated_value
    
    def max_money_for_symbol_by_ATR(
        self, 
        symbol: str,
        tolerance: float = 0.003,
        atr_multiplier: float = 1
        ) -> float:
        etf_data = get_etf_data_by_symbol(symbol)
        atr_calculator = AverageTrueRange(window=50)
        etf_data.add_factors(atr_calculator)
        etf_data.calc_factors()
        res = etf_data.factor_results.get(atr_calculator.name)
        latest_atr = res.iloc[-1] if res is not None else None
        if latest_atr is None or latest_atr <= 0:
            raise ValueError("Invalid ATR value.")
        return self.all_money * tolerance / (latest_atr * atr_multiplier)
    
    def calc_corr_with_current_position(self, symbol: str, name_dict: Mapping[str, str]|None = None) -> pd.DataFrame:
        etf_data = get_etf_data_by_symbol(symbol)
        idx = []
        corr = CorrelationFactor()
        core_matrix = corr(*( [pos.data for pos in self._positions.values()] + [etf_data]))
        core_matrix["name"] = name_dict.get(symbol, symbol) if name_dict else symbol
        core_matrix["symbol"] = symbol
        return core_matrix.set_index(["symbol"])
    
    def calc_corrs_with_current_position(self, symbols: list[str], name_dict: Mapping[str, str]|None = None) -> pd.DataFrame:
        corr = CorrelationFactor()
        corr_analysis: list[pd.DataFrame] = []
        idx = []
        for symbol in symbols:
            corr_analysis.append(self.calc_corr_with_current_position(symbol, name_dict))

        return pd.concat(corr_analysis, axis=0).sort_values(by="effective_diversification_number", ascending=False)

@dataclass
class Position:
    symbol: str
    quantity: int
    _current_price: Optional[float] = field(default=None, init=False)
    _data: Optional[EtfData] = None

    @property
    def market_value(self) -> float:
        return self.current_price * self.quantity
    
    @property
    def data(self) -> EtfData:
        if self._data is None:
            self._data = get_etf_data_by_symbol(self.symbol)
        return self._data
    
    @property
    def current_price(self) -> float:
        if self._current_price is None:
            self._current_price = self.data.data.iloc[-1]['close']
        return self._current_price # type: ignore
    
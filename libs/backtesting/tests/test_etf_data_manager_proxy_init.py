"""Focused regression tests for ETF update proxy warm-up.

Run from project root:
    PYTHONPATH=libs pytest libs/backtesting/tests/test_etf_data_manager_proxy_init.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import data_manager.etf_data_manager as etf_data_manager


def test_get_proxy_init_delay_staggers_by_worker_identity(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        etf_data_manager,
        "current_process",
        lambda: SimpleNamespace(name="ForkPoolWorker-4", _identity=(4,)),
    )

    assert etf_data_manager._get_proxy_init_delay(0.75) == pytest.approx(2.25)


def test_initialize_etf_update_worker_sleeps_before_proxy_warmup(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[str, float | None]] = []

    monkeypatch.setattr(
        etf_data_manager,
        "current_process",
        lambda: SimpleNamespace(name="ForkPoolWorker-3", _identity=(3,)),
    )
    monkeypatch.setattr(etf_data_manager, "intervals", lambda seconds=1.0: calls.append(("sleep", seconds)))
    monkeypatch.setattr(etf_data_manager, "initialize_proxy_pool", lambda: calls.append(("init", None)))

    etf_data_manager._initialize_etf_update_worker(0.5)

    assert calls == [("sleep", 1.0), ("init", None)]
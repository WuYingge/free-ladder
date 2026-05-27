"""Focused regression tests for proxy provider integration.

Run from project root:
    PYTHONPATH=libs pytest libs/proxy/tests/test_proxy_pool.py -v
"""

from __future__ import annotations

import json

import pytest

import proxy.proxy as proxy_module
from proxy.proxy import PROXY_PROVIDERS, ProxyAccount, ProxyPool, ProxySettings


class _ResponseStub:
    def __init__(self, payload: dict):
        self._payload = payload
        self.content = json.dumps(payload).encode("utf-8")
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class _RequesterStub:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[tuple[str, int]] = []

    def get(self, url: str, timeout: int):
        self.calls.append((url, timeout))
        return _ResponseStub(self.payload)


class _SequenceRequesterStub:
    def __init__(self, payloads: list[dict]):
        self.payloads = payloads
        self.calls: list[tuple[str, int]] = []

    def get(self, url: str, timeout: int):
        self.calls.append((url, timeout))
        payload_index = min(len(self.calls) - 1, len(self.payloads) - 1)
        return _ResponseStub(self.payloads[payload_index])


def test_proxy_account_prefers_generic_api_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PROXY_PROVIDER", "luotuo")
    monkeypatch.setenv("PROXY_API_URL", "https://proxy.example/luotuo")
    monkeypatch.delenv("PROXY_ENCRYPT_URL", raising=False)
    monkeypatch.setenv("PROXY_UNBIND_TIME", "900")

    settings = ProxyAccount.load()

    assert settings.provider.name == "luotuo"
    assert settings.api_url == "https://proxy.example/luotuo"
    assert settings.unbind_time == 900


def test_proxy_account_keeps_legacy_encrypt_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PROXY_PROVIDER", raising=False)
    monkeypatch.delenv("PROXY_API_URL", raising=False)
    monkeypatch.setenv("PROXY_ENCRYPT_URL", "https://proxy.example/hailiang")
    monkeypatch.delenv("PROXY_UNBIND_TIME", raising=False)

    settings = ProxyAccount.load()

    assert settings.provider.name == "hailiang"
    assert settings.api_url == "https://proxy.example/hailiang"
    assert settings.unbind_time == 600


def test_proxy_account_rejects_provider_api_url_family_mismatch(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PROXY_PROVIDER", "hailiang")
    monkeypatch.setenv(
        "PROXY_API_URL",
        "https://www.lthttp.com/iplist?key=test&count=1&isAuth=false",
    )
    monkeypatch.delenv("PROXY_ENCRYPT_URL", raising=False)

    with pytest.raises(EnvironmentError, match="does not match PROXY_API_URL"):
        ProxyAccount.load()


def test_hailiang_payload_is_parsed_to_requests_proxy_dicts():
    requester = _RequesterStub(
        {
            "code": 0,
            "data": [
                {
                    "ip": "1.2.3.4",
                    "port": "8080",
                }
            ],
        }
    )

    pool = ProxyPool(
        settings=ProxySettings(
            provider=PROXY_PROVIDERS["hailiang"],
            api_url="https://proxy.example/hailiang",
            unbind_time=600,
            static_ttl_seconds=900,
            static_failure_cooldown_seconds=30.0,
        ),
        requester=requester,
    )

    assert requester.calls == [("https://proxy.example/hailiang", 8)]
    assert [lease.proxy for lease in pool.pool] == [
        {"http": "http://1.2.3.4:8080", "https": "http://1.2.3.4:8080"}
    ]


def test_luotuo_payload_accepts_string_code():
    requester = _RequesterStub(
        {
            "code": "0",
            "msg": "",
            "data": [
                {
                    "ip": "222.186.21.101",
                    "port": "10268",
                    "outIp": "222.186.21.101",
                    "isp": "dx",
                    "pro": "云南",
                    "city": "昭通",
                    "expire": "2024-10-23 17:49:05",
                }
            ],
        }
    )

    pool = ProxyPool(
        settings=ProxySettings(
            provider=PROXY_PROVIDERS["luotuo"],
            api_url="https://proxy.example/luotuo",
            unbind_time=600,
            static_ttl_seconds=900,
            static_failure_cooldown_seconds=30.0,
        ),
        requester=requester,
    )

    assert requester.calls == [("https://proxy.example/luotuo", 8)]
    assert [lease.proxy for lease in pool.pool] == [
        {"http": "http://222.186.21.101:10268", "https": "http://222.186.21.101:10268"}
    ]


def test_generic_string_payload_is_supported():
    requester = _RequesterStub(
        {
            "code": "0",
            "data": "10.0.0.1:80\nhttp://10.0.0.2:81",
        }
    )

    pool = ProxyPool(
        settings=ProxySettings(
            provider=PROXY_PROVIDERS["luotuo"],
            api_url="https://proxy.example/generic",
            unbind_time=600,
            static_ttl_seconds=900,
            static_failure_cooldown_seconds=30.0,
        ),
        requester=requester,
    )

    assert [lease.proxy for lease in pool.pool] == [
        {"http": "http://10.0.0.1:80", "https": "http://10.0.0.1:80"},
        {"http": "http://10.0.0.2:81", "https": "http://10.0.0.2:81"},
    ]


def test_proxy_pool_reuses_last_successful_proxy_until_it_fails():
    requester = _RequesterStub(
        {
            "code": "0",
            "data": [
                {"ip": "10.0.0.1", "port": "80"},
                {"ip": "10.0.0.2", "port": "81"},
            ],
        }
    )

    pool = ProxyPool(
        settings=ProxySettings(
            provider=PROXY_PROVIDERS["luotuo"],
            api_url="https://proxy.example/reuse",
            unbind_time=600,
            static_ttl_seconds=900,
            static_failure_cooldown_seconds=30.0,
        ),
        requester=requester,
    )

    first_proxy = pool.get_proxy()
    pool.mark_success(first_proxy)

    assert pool.get_proxy() == first_proxy

    pool.mark_failure(first_proxy)

    assert pool.get_proxy() != first_proxy


def test_static_proxy_pool_reuses_ip_until_expiry(monkeypatch: pytest.MonkeyPatch):
    fake_now = {"value": 1_000.0}
    monkeypatch.setattr(proxy_module.time, "time", lambda: fake_now["value"])
    requester = _RequesterStub(
        {
            "code": "0",
            "data": [
                {"ip": "121.31.233.38", "port": 11031, "outIp": "117.162.195.69"},
            ],
        }
    )

    pool = ProxyPool(
        settings=ProxySettings(
            provider=PROXY_PROVIDERS["luotuo_static"],
            api_url="https://proxy.example/static",
            unbind_time=600,
            static_ttl_seconds=900,
            static_failure_cooldown_seconds=30.0,
        ),
        requester=requester,
    )

    first_proxy = pool.get_proxy()
    pool.mark_success(first_proxy)
    second_proxy = pool.get_proxy()

    assert first_proxy == second_proxy
    assert len(pool.pool) == 1
    assert pool.pool[0].expires_at == pytest.approx(1_900.0)


def test_static_proxy_pool_refreshes_only_after_expiry(monkeypatch: pytest.MonkeyPatch):
    fake_now = {"value": 1_000.0}
    monkeypatch.setattr(proxy_module.time, "time", lambda: fake_now["value"])
    requester = _SequenceRequesterStub(
        [
            {"code": "0", "data": [{"ip": "121.31.233.38", "port": 11031}]},
            {"code": "0", "data": [{"ip": "121.31.233.47", "port": 12548}]},
        ]
    )

    pool = ProxyPool(
        settings=ProxySettings(
            provider=PROXY_PROVIDERS["luotuo_static"],
            api_url="https://proxy.example/static-refresh",
            unbind_time=600,
            static_ttl_seconds=10,
            static_failure_cooldown_seconds=30.0,
        ),
        requester=requester,
    )

    first_proxy = pool.get_proxy()
    fake_now["value"] += 11
    second_proxy = pool.get_proxy()

    assert first_proxy != second_proxy
    assert requester.calls == [
        ("https://proxy.example/static-refresh", 8),
        ("https://proxy.example/static-refresh", 8),
    ]


def test_static_proxy_pool_reuses_shared_cache_across_instances(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
):
    fake_now = {"value": 1_000.0}
    monkeypatch.setattr(proxy_module.time, "time", lambda: fake_now["value"])
    requester = _SequenceRequesterStub(
        [
            {"code": "0", "data": [{"ip": "121.31.233.38", "port": 11031}]},
            {"code": "0", "data": [{"ip": "121.31.233.47", "port": 12548}]},
        ]
    )
    settings = ProxySettings(
        provider=PROXY_PROVIDERS["luotuo_static"],
        api_url="https://proxy.example/shared-static",
        unbind_time=600,
        static_ttl_seconds=900,
        static_failure_cooldown_seconds=30.0,
        shared_cache_dir=str(tmp_path),
    )

    first_pool = ProxyPool(settings=settings, requester=requester)
    second_pool = ProxyPool(settings=settings, requester=requester)

    assert first_pool.get_proxy() == second_pool.get_proxy()
    assert requester.calls == [("https://proxy.example/shared-static", 8)]
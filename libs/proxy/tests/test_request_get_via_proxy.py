"""Focused regression tests for proxy request handling.

Run from project root:
    PYTHONPATH=libs pytest libs/proxy/tests/test_request_get_via_proxy.py -v
"""

from __future__ import annotations

from curl_cffi import requests

import fetcher.utils as fetcher_utils


class _ResponseStub:
    def raise_for_status(self) -> None:
        return None


def test_request_get_via_proxy_marks_bad_proxy_and_reuses_good_proxy(monkeypatch):
    proxy_calls = iter(
        [
            {"http": "http://bad-proxy", "https": "http://bad-proxy"},
            {"http": "http://good-proxy", "https": "http://good-proxy"},
        ]
    )
    events: list[tuple[str, dict[str, str]]] = []
    request_timeouts: list[tuple[float, int]] = []
    request_impersonates: list[str] = []

    monkeypatch.setattr(fetcher_utils, "get_proxy", lambda: next(proxy_calls))
    monkeypatch.setattr(fetcher_utils, "mark_proxy_failure", lambda proxy: events.append(("failure", proxy)))
    monkeypatch.setattr(fetcher_utils, "mark_proxy_success", lambda proxy: events.append(("success", proxy)))

    def fake_get(url: str, timeout, proxies, **kwargs):
        del url
        request_timeouts.append(timeout)
        request_impersonates.append(kwargs["impersonate"])
        if proxies["http"] == "http://bad-proxy":
            raise requests.exceptions.ProxyError("boom")
        return _ResponseStub()

    monkeypatch.setattr(fetcher_utils.requests, "get", fake_get)

    response = fetcher_utils.request_get_via_proxy("https://example.test", timeout=15, max_proxy_retries=2)

    assert isinstance(response, _ResponseStub)
    assert request_timeouts == [(3.0, 15), (3.0, 15)]
    assert request_impersonates == [fetcher_utils.PROXY_IMPERSONATE_BROWSER, fetcher_utils.PROXY_IMPERSONATE_BROWSER]
    assert events == [
        ("failure", {"http": "http://bad-proxy", "https": "http://bad-proxy"}),
        ("success", {"http": "http://good-proxy", "https": "http://good-proxy"}),
    ]
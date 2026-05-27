import hashlib
import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from random import randint
from typing import Any
from urllib.parse import urlparse

import dotenv
import requests

try:
    import fcntl
except ImportError:
    fcntl = None


dotenv.load_dotenv()

DEFAULT_HAILIANG_URL = (
    "https://api.hailiangip.com:8522/api/getIpEncrypt?dataType=0&encryptParam="
    "9f9A9DgV7OXgVxE3tqtmm6j6rS8I99d%2Fg9hvk%2BVqKqRpFV7sxse1Mptjrxe1uPHv4"
    "yIotsaqoUafihPVbud4vZSmgdAVfPHJipH%2F50AvUmCxI5kSrBPVhP7GpXIAOqbIv%2B"
    "W0vWDVclnDvojQXkyrA02C59MOHdgFvQvJ4XXIuLzBOMWZSZhMyHtk6dQPjykAleo%2F"
    "E0ZLu1LLR9Afir%2BLdk%2BO7UXupg6Lh8sV7xFl1H8%3D"
)

DEFAULT_LUOTUO_STATIC_URL = (
    "https://www.lthttp.com/iplist?key=ec806fa51e736d2c&count=50&protocol=0&type=0"
    "&isp=0&distinct=0&os=1&cs=0&is=0&es=0&textSep=1&isAuth=false&province=&city="
)
DEFAULT_STATIC_PROXY_TTL_SECONDS = 15 * 60
DEFAULT_STATIC_PROXY_FAILURE_COOLDOWN_SECONDS = 30.0
DEFAULT_SHARED_PROXY_CACHE_DIR = os.path.join(tempfile.gettempdir(), "invest_proxy_cache")


@dataclass(frozen=True)
class ProxyProvider:
    name: str
    default_url: str | None = None
    code_key: str = "code"
    success_codes: tuple[str, ...] = ("0",)
    data_key: str = "data"
    ip_key: str = "ip"
    port_key: str = "port"
    expire_key: str | None = "expire"
    reuse_until_expired: bool = False
    default_ttl_seconds: int | None = None


@dataclass(frozen=True)
class ProxySettings:
    provider: ProxyProvider
    api_url: str
    unbind_time: int
    static_ttl_seconds: int
    static_failure_cooldown_seconds: float
    shared_cache_dir: str | None = None


@dataclass
class ProxyLease:
    proxy: dict[str, str]
    expires_at: float | None = None
    cooldown_until: float = 0.0

    @property
    def key(self) -> str:
        return _proxy_key(self.proxy)


PROXY_PROVIDERS: dict[str, ProxyProvider] = {
    "hailiang": ProxyProvider(name="hailiang", default_url=DEFAULT_HAILIANG_URL),
    "luotuo": ProxyProvider(name="luotuo"),
    "luotuo_static": ProxyProvider(
        name="luotuo_static",
        default_url=DEFAULT_LUOTUO_STATIC_URL,
        reuse_until_expired=True,
        default_ttl_seconds=DEFAULT_STATIC_PROXY_TTL_SECONDS,
    ),
}

PROXY_PROVIDER_FAMILIES: dict[str, str] = {
    "hailiang": "hailiang",
    "luotuo": "luotuo",
    "luotuo_static": "luotuo",
}


def _detect_provider_family_from_api_url(api_url: str) -> str | None:
    hostname = (urlparse(api_url).hostname or "").strip().lower()
    if hostname.endswith("hailiangip.com"):
        return "hailiang"
    if hostname.endswith("lthttp.com"):
        return "luotuo"
    return None


def _validate_provider_api_url(provider: ProxyProvider, api_url: str):
    configured_family = PROXY_PROVIDER_FAMILIES.get(provider.name)
    detected_family = _detect_provider_family_from_api_url(api_url)
    if configured_family is None or detected_family is None or configured_family == detected_family:
        return
    raise EnvironmentError(
        f"proxy provider '{provider.name}' does not match PROXY_API_URL '{api_url}'. "
        f"The URL looks like the '{detected_family}' provider family; "
        "update PROXY_PROVIDER or PROXY_API_URL so they target the same backend."
    )


class ProxyAccount:
    @classmethod
    def load(cls) -> ProxySettings:
        provider_name = (os.getenv("PROXY_PROVIDER") or "hailiang").strip().lower()
        provider = PROXY_PROVIDERS.get(provider_name) or ProxyProvider(name=provider_name)
        api_url = (
            os.getenv("PROXY_API_URL")
            or os.getenv("PROXY_ENCRYPT_URL")
            or provider.default_url
        )
        if not api_url:
            raise EnvironmentError(
                f"no proxy url configured for provider '{provider.name}', "
                "set PROXY_API_URL or PROXY_ENCRYPT_URL"
            )
        _validate_provider_api_url(provider, api_url)
        unbind_time = int(os.getenv("PROXY_UNBIND_TIME") or "600")
        static_ttl_seconds = int(
            os.getenv("PROXY_STATIC_TTL_SECONDS")
            or provider.default_ttl_seconds
            or DEFAULT_STATIC_PROXY_TTL_SECONDS
        )
        static_failure_cooldown_seconds = float(
            os.getenv("PROXY_STATIC_FAILURE_COOLDOWN_SECONDS")
            or DEFAULT_STATIC_PROXY_FAILURE_COOLDOWN_SECONDS
        )
        shared_cache_dir = (os.getenv("PROXY_SHARED_CACHE_DIR") or "").strip() or DEFAULT_SHARED_PROXY_CACHE_DIR
        return ProxySettings(
            provider=provider,
            api_url=api_url,
            unbind_time=unbind_time,
            static_ttl_seconds=max(static_ttl_seconds, 1),
            static_failure_cooldown_seconds=max(static_failure_cooldown_seconds, 0.0),
            shared_cache_dir=shared_cache_dir,
        )


def _normalize_response_code(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_proxy_url(value: str) -> str:
    proxy_url = value.strip()
    if proxy_url.startswith(("http://", "https://")):
        return proxy_url
    return f"http://{proxy_url}"


def _build_proxy_mapping(ip: str, port: str) -> dict[str, str]:
    proxy_url = _normalize_proxy_url(f"{ip}:{port}")
    return {"http": proxy_url, "https": proxy_url}


def _build_string_proxy_mapping(proxy_value: str) -> dict[str, str]:
    proxy_url = _normalize_proxy_url(proxy_value)
    return {"http": proxy_url, "https": proxy_url}


def _parse_expire_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric_value = float(value)
        return numeric_value / 1000 if numeric_value > 1e11 else numeric_value

    text = str(value).strip()
    if not text:
        return None

    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue

    try:
        numeric_value = float(text)
    except ValueError:
        return None
    return numeric_value / 1000 if numeric_value > 1e11 else numeric_value


def _proxy_key(proxy: dict[str, str] | None) -> str:
    if not proxy:
        return ""
    return proxy.get("https") or proxy.get("http") or ""


def _build_proxy_lease(
    item: Any,
    provider: ProxyProvider,
    *,
    fetched_at: float,
    static_ttl_seconds: int,
) -> ProxyLease:
    expires_at = None

    if isinstance(item, str):
        proxy = _build_string_proxy_mapping(item)
    else:
        if not isinstance(item, dict):
            raise ValueError(f"proxy item has unsupported type: {item}")

        ip = item.get(provider.ip_key)
        port = item.get(provider.port_key)
        if not ip or not port:
            raise ValueError(f"proxy item missing ip or port: {item}")
        proxy = _build_proxy_mapping(str(ip), str(port))
        if provider.expire_key:
            expires_at = _parse_expire_timestamp(item.get(provider.expire_key))

    if provider.reuse_until_expired and expires_at is None:
        expires_at = fetched_at + static_ttl_seconds

    return ProxyLease(proxy=proxy, expires_at=expires_at)


def _parse_proxy_payload(
    payload: dict[str, Any],
    provider: ProxyProvider,
    *,
    fetched_at: float,
    static_ttl_seconds: int,
) -> list[ProxyLease]:
    success_code = _normalize_response_code(payload.get(provider.code_key))
    if success_code not in provider.success_codes:
        raise ValueError(f"proxy API returned non-zero code: {payload}")

    raw_data = payload.get(provider.data_key, [])
    if isinstance(raw_data, str):
        data_items: list[Any] = [item.strip() for item in raw_data.splitlines() if item.strip()]
    elif isinstance(raw_data, list):
        data_items = raw_data
    else:
        raise ValueError(f"proxy API returned unsupported data payload: {payload}")

    proxy_map: dict[str, ProxyLease] = {}
    for item in data_items:
        lease = _build_proxy_lease(
            item,
            provider,
            fetched_at=fetched_at,
            static_ttl_seconds=static_ttl_seconds,
        )
        if not lease.key:
            continue
        existing = proxy_map.get(lease.key)
        if existing is None or (lease.expires_at or 0.0) > (existing.expires_at or 0.0):
            proxy_map[lease.key] = lease

    proxies = list(proxy_map.values())
    if not proxies:
        raise ValueError(f"proxy API returned empty data: {payload}")
    return proxies


class ProxyPool:
    def __init__(self, settings: ProxySettings | None = None, requester: Any = requests):
        self.settings = settings or ProxyAccount.load()
        self.unbindTime = self.settings.unbind_time
        self.requester = requester
        self.start_time = int(time.time())
        self.active_proxy: ProxyLease | None = None
        self.bad_proxy_keys: set[str] = set()
        try:
            self.pool = self._get_proxies()
        except Exception as err:
            print(f"ProxyPool init failed for {self.settings.provider.name}: {err}")
            self.pool = []

    def _decode_payload(self, response: Any) -> dict[str, Any]:
        try:
            return response.json()
        except ValueError:
            return json.loads(response.content)

    def _is_static_pool(self) -> bool:
        return self.settings.provider.reuse_until_expired

    def _get_shared_cache_path(self) -> str | None:
        if not self._is_static_pool() or not self.settings.shared_cache_dir:
            return None
        cache_key = hashlib.sha1(
            f"{self.settings.provider.name}|{self.settings.api_url}".encode("utf-8")
        ).hexdigest()[:16]
        return os.path.join(self.settings.shared_cache_dir, f"{self.settings.provider.name}_{cache_key}.json")

    @contextmanager
    def _locked_cache_file(self, cache_path: str):
        cache_dir = os.path.dirname(cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        with open(cache_path, "a+", encoding="utf-8") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            handle.seek(0)
            try:
                yield handle
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _load_cached_leases(self, handle) -> list[ProxyLease]:
        handle.seek(0)
        raw_payload = handle.read().strip()
        if not raw_payload:
            return []

        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return []

        leases: list[ProxyLease] = []
        for item in payload.get("leases", []):
            proxy_payload = item.get("proxy")
            if not isinstance(proxy_payload, dict):
                continue
            proxy = {str(name): str(value) for name, value in proxy_payload.items()}
            if not _proxy_key(proxy):
                continue
            leases.append(
                ProxyLease(
                    proxy=proxy,
                    expires_at=_parse_expire_timestamp(item.get("expires_at")),
                )
            )
        return leases

    def _write_cached_leases(self, handle, leases: list[ProxyLease]):
        payload = {
            "provider": self.settings.provider.name,
            "api_url": self.settings.api_url,
            "updated_at": time.time(),
            "leases": [
                {
                    "proxy": lease.proxy,
                    "expires_at": lease.expires_at,
                }
                for lease in leases
            ],
        }
        handle.seek(0)
        handle.truncate()
        json.dump(payload, handle, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())

    def _fetch_proxies_from_api(self) -> list[ProxyLease]:
        max_retries = 5
        retry_wait_sec = 15
        for _ in range(max_retries):
            response = None
            try:
                fetched_at = time.time()
                response = self.requester.get(self.settings.api_url, timeout=8)
                js_res = self._decode_payload(response)
                return _parse_proxy_payload(
                    js_res,
                    self.settings.provider,
                    fetched_at=fetched_at,
                    static_ttl_seconds=self.settings.static_ttl_seconds,
                )
            except Exception as err:
                print(f"Can't get proxy due to {err}, response is {response.text if response else 'None'}")
                time.sleep(retry_wait_sec)
                continue
        raise RuntimeError("Failed to acquire proxies after max retries")

    def _get_shared_static_proxies(self) -> list[ProxyLease]:
        cache_path = self._get_shared_cache_path()
        if not cache_path:
            return self._fetch_proxies_from_api()

        with self._locked_cache_file(cache_path) as handle:
            cached_leases = self._load_cached_leases(handle)
            valid_cached = [lease for lease in cached_leases if not self._is_expired(lease)]
            if valid_cached:
                return valid_cached

            fresh_leases = self._fetch_proxies_from_api()
            self._write_cached_leases(handle, fresh_leases)
            return fresh_leases

    def _find_lease(self, proxy_key: str) -> ProxyLease | None:
        if self.active_proxy is not None and self.active_proxy.key == proxy_key:
            return self.active_proxy
        for lease in self.pool:
            if lease.key == proxy_key:
                return lease
        return None

    def _is_expired(self, lease: ProxyLease, now: float | None = None) -> bool:
        if lease.expires_at is None:
            return False
        current_time = time.time() if now is None else now
        return lease.expires_at <= current_time

    def _purge_expired(self, now: float | None = None):
        current_time = time.time() if now is None else now
        self.pool = [lease for lease in self.pool if not self._is_expired(lease, current_time)]
        valid_keys = {lease.key for lease in self.pool}
        self.bad_proxy_keys.intersection_update(valid_keys)
        if self.active_proxy is not None and self._is_expired(self.active_proxy, current_time):
            self.active_proxy = None

    def _get_active_proxy(self, now: float | None = None) -> ProxyLease | None:
        current_time = time.time() if now is None else now
        if self.active_proxy is None:
            return None
        if self._is_expired(self.active_proxy, current_time):
            self.active_proxy = None
            return None
        if self._is_static_pool() and self.active_proxy.cooldown_until > current_time:
            self.active_proxy = None
            return None
        if not self._is_static_pool() and self.active_proxy.key in self.bad_proxy_keys:
            self.active_proxy = None
            return None
        return self.active_proxy

    def _available_leases(self, now: float | None = None) -> list[ProxyLease]:
        current_time = time.time() if now is None else now
        leases = [lease for lease in self.pool if not self._is_expired(lease, current_time)]
        if self._is_static_pool():
            ready = [lease for lease in leases if lease.cooldown_until <= current_time]
            return ready or leases
        return [lease for lease in leases if lease.key not in self.bad_proxy_keys]

    def _get_proxies(self) -> list[ProxyLease]:
        if self._is_static_pool():
            return self._get_shared_static_proxies()
        return self._fetch_proxies_from_api()

    def get_proxy(self):
        now = time.time()
        self._purge_expired(now)
        if self._check_timeout() or (not self.pool and self.active_proxy is None):
            self.refresh()
            now = time.time()
            self._purge_expired(now)

        active_proxy = self._get_active_proxy(now)
        if active_proxy is not None:
            return active_proxy.proxy

        candidates = self._available_leases(now)
        if not candidates:
            self.refresh()
            now = time.time()
            self._purge_expired(now)
            active_proxy = self._get_active_proxy(now)
            if active_proxy is not None:
                return active_proxy.proxy
            candidates = self._available_leases(now)
        if candidates:
            lease = candidates[randint(0, len(candidates) - 1)]
            if not self._is_static_pool():
                self.pool.remove(lease)
            return lease.proxy
        raise RuntimeError("Proxy pool is empty")

    def _check_timeout(self):
        if self._is_static_pool():
            self._purge_expired()
            return not self.pool and self.active_proxy is None
        return int(time.time()) - self.start_time + 20 > self.unbindTime

    def refresh(self):
        self.start_time = int(time.time())
        self.active_proxy = None
        self.bad_proxy_keys.clear()
        self.pool = self._get_proxies()

    def mark_success(self, proxy: dict[str, str] | None):
        proxy_key = _proxy_key(proxy)
        if not proxy_key:
            return
        lease = self._find_lease(proxy_key)
        if lease is None:
            if proxy is None:
                return
            lease = ProxyLease(proxy={name: value for name, value in proxy.items()})
            if self._is_static_pool():
                lease.expires_at = time.time() + self.settings.static_ttl_seconds
                self.pool.append(lease)
        if self._is_static_pool():
            lease.cooldown_until = 0.0
        self.bad_proxy_keys.discard(proxy_key)
        self.active_proxy = lease

    def mark_failure(self, proxy: dict[str, str] | None):
        proxy_key = _proxy_key(proxy)
        if not proxy_key:
            return
        if self._is_static_pool():
            lease = self._find_lease(proxy_key)
            if lease is not None:
                lease.cooldown_until = time.time() + self.settings.static_failure_cooldown_seconds
            if self.active_proxy is not None and self.active_proxy.key == proxy_key:
                self.active_proxy = None
            return
        self.bad_proxy_keys.add(proxy_key)
        if self.active_proxy is not None and self.active_proxy.key == proxy_key:
            self.active_proxy = None


class _LazyProxyPool:
    def __init__(self):
        self._pool: ProxyPool | None = None

    def _get_pool(self) -> ProxyPool:
        if self._pool is None:
            self._pool = ProxyPool()
        return self._pool

    def ensure_initialized(self) -> ProxyPool:
        return self._get_pool()

    def get_proxy(self):
        return self._get_pool().get_proxy()

    def refresh(self):
        return self._get_pool().refresh()

    def mark_success(self, proxy: dict[str, str] | None):
        return self._get_pool().mark_success(proxy)

    def mark_failure(self, proxy: dict[str, str] | None):
        return self._get_pool().mark_failure(proxy)

    def __getattr__(self, name: str):
        return getattr(self._get_pool(), name)


PROXY_POOL = _LazyProxyPool()


def initialize_proxy_pool() -> ProxyPool:
    return PROXY_POOL.ensure_initialized()


def mark_proxy_success(proxy: dict[str, str] | None):
    PROXY_POOL.mark_success(proxy)


def mark_proxy_failure(proxy: dict[str, str] | None):
    PROXY_POOL.mark_failure(proxy)


def get_proxy():
    return PROXY_POOL.get_proxy()

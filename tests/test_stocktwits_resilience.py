"""StockTwits fetch: transport-error resilience (#1024), crypto symbol
mapping (#1113), and CDN TLS-fingerprint fallback.

StockTwits lists crypto under ``<BASE>.X`` (Yahoo's ``BTC-USD`` 404s). The
public stream is still unauthenticated, but the CDN challenges Python's
default TLS fingerprint, so the fetcher prefers ``curl_cffi`` impersonation
and degrades to a placeholder on any remaining transport error.
"""

from __future__ import annotations

import http.client
import json
from unittest.mock import patch
from urllib.error import HTTPError

import pytest

from tradingagents.dataflows import stocktwits


def _raise(exc):
    class _Resp:
        def __enter__(self_inner):
            return self_inner

        def __exit__(self_inner, *a):
            return False

        def read(self_inner):
            raise exc
    return _Resp()


class _FakeCFResp:
    def __init__(self, status, body, content_type="application/json"):
        self.status_code = status
        self.content = body if isinstance(body, bytes) else body.encode()
        self.headers = {"content-type": content_type}


@pytest.mark.unit
class TestStockTwitsResilience:
    @pytest.mark.parametrize(
        "exc",
        [
            http.client.IncompleteRead(b""),
            HTTPError("url", 503, "down", {}, None),
            TimeoutError("slow"),
        ],
    )
    def test_transport_errors_return_placeholder(self, exc):
        with patch.object(stocktwits, "_get_bytes", side_effect=exc):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert "unavailable" in out.lower()
        assert out.startswith("<stocktwits unavailable")

    def test_httperror_placeholder_includes_status_code(self):
        with patch.object(
            stocktwits, "_get_bytes", side_effect=HTTPError("url", 403, "Forbidden", {}, None)
        ):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert out == "<stocktwits unavailable: HTTPError 403>"

    def test_urlopen_used_when_curl_cffi_missing(self):
        with (
            patch.object(stocktwits, "_cf_requests", None),
            patch.object(stocktwits, "urlopen", return_value=_raise(TimeoutError("slow"))),
        ):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert out.startswith("<stocktwits unavailable")

    def test_impersonate_403_falls_back_to_next_profile(self):
        calls = []

        def fake_get(url, **kwargs):
            profile = kwargs.get("impersonate")
            calls.append(profile)
            if profile == "chrome124":
                return _FakeCFResp(403, b"<html>Just a moment...</html>", "text/html")
            payload = json.dumps({"messages": []}).encode()
            return _FakeCFResp(200, payload)

        fake_cf = type("CF", (), {"get": staticmethod(fake_get)})
        with patch.object(stocktwits, "_cf_requests", fake_cf):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert out.startswith("<no StockTwits messages found")
        assert calls == ["chrome124", "safari180"]

    def test_formats_bullish_bearish_summary(self):
        payload = {
            "messages": [
                {
                    "created_at": "2026-01-01T00:00:00Z",
                    "user": {"username": "bull"},
                    "entities": {"sentiment": {"basic": "Bullish"}},
                    "body": "buying the dip &#39;now&#39;",
                },
                {
                    "created_at": "2026-01-01T00:01:00Z",
                    "user": {"username": "bear"},
                    "entities": {"sentiment": {"basic": "Bearish"}},
                    "body": "overextended",
                },
            ]
        }
        with patch.object(stocktwits, "_get_bytes", return_value=json.dumps(payload).encode()):
            out = stocktwits.fetch_stocktwits_messages("NVDA")
        assert "Bullish: 1 (50%)" in out
        assert "Bearish: 1 (50%)" in out
        assert "@bull · Bullish" in out
        assert "buying the dip 'now'" in out
        assert "@bear · Bearish" in out


@pytest.mark.unit
class TestStockTwitsCryptoSymbols:
    @pytest.mark.parametrize(
        ("ticker", "expected"),
        [
            ("BTC-USD", "BTC.X"),
            ("eth-usd", "ETH.X"),
            ("SOL-USD", "SOL.X"),
            ("BTCUSD", "BTC.X"),      # undashed broker form
            ("BTC-USDT", "BTC.X"),    # stablecoin quote
            ("AMD", "AMD"),
            ("BRK-B", "BRK-B"),       # dashed class share: untouched
            ("GOLD", "GOLD"),         # real equity (aliases elsewhere): untouched here
            ("XYZ-USD", "XYZ-USD"),   # unknown base: not treated as crypto
        ],
    )
    def test_symbol_mapping(self, ticker, expected):
        assert stocktwits._stocktwits_symbol(ticker) == expected

    def test_crypto_pair_requests_dot_x_endpoint(self):
        seen = {}

        def fake_get_bytes(url, timeout=None):
            seen["url"] = url
            raise TimeoutError("stop after capturing the URL")

        with patch.object(stocktwits, "_get_bytes", side_effect=fake_get_bytes):
            stocktwits.fetch_stocktwits_messages("BTC-USD")
        assert "/symbol/BTC.X.json" in seen["url"]

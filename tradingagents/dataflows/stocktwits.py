"""StockTwits public symbol-stream fetcher.

StockTwits exposes a per-symbol message stream at
``api.stocktwits.com/api/2/streams/symbol/{ticker}.json`` that requires no
API key, no OAuth, and no registration. Each message includes a
user-labeled sentiment field (``Bullish``/``Bearish``/null), the message
body, timestamp, and posting user.

The CDN in front of that endpoint challenges Python's default TLS
fingerprint (``urllib`` / ``requests`` return HTTP 403 HTML). ``yfinance``
already depends on ``curl_cffi`` for the same reason on Yahoo, so when it
is installed we reuse it with browser impersonation profiles the CDN
currently accepts, then fall back to ``urlopen``.

The function is deliberately self-contained: short timeout, graceful
degradation on any HTTP or parse failure, and a string return type so
the calling agent gets a uniform interface regardless of whether the
network call succeeded.
"""

from __future__ import annotations

import html
import http.client
import json
import logging
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .symbol_utils import crypto_base

try:
    from curl_cffi import requests as _cf_requests
except ImportError:  # pragma: no cover - optional; yfinance pulls it in
    _cf_requests = None

logger = logging.getLogger(__name__)

_API = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
# Profiles verified live against the StockTwits CDN. Newer Chrome impersonations
# (chrome131+, generic "chrome") currently get a JS challenge; these two do not.
_IMPERSONATE_PROFILES = ("chrome124", "safari180")


def _stocktwits_symbol(ticker: str) -> str:
    """Map a crypto pair to StockTwits' ``<BASE>.X`` convention.

    StockTwits lists crypto as ``BTC.X`` (Yahoo's ``BTC-USD`` form 404s), so any
    crypto symbol resolves to its base plus ``.X``; other symbols pass through
    upper-cased.
    """
    base = crypto_base(ticker)
    return f"{base}.X" if base else ticker.strip().upper()


def _is_json_body(status: int, content_type: str | None, body: bytes) -> bool:
    if status != 200 or not body:
        return False
    if content_type and "json" in content_type.lower():
        return True
    return body.lstrip()[:1] == b"{"


def _curl_cffi_get(url: str, timeout: float, profile: str) -> bytes | None:
    """GET ``url`` impersonating ``profile``. Return the body on JSON 200, else None."""
    resp = _cf_requests.get(
        url,
        impersonate=profile,
        timeout=timeout,
        headers={"Accept": "application/json"},
    )
    if _is_json_body(resp.status_code, resp.headers.get("content-type"), resp.content):
        return resp.content
    if resp.status_code == 429:
        logger.warning("StockTwits 429 via %s — retrying once", profile)
        time.sleep(2.0)
        resp = _cf_requests.get(
            url,
            impersonate=profile,
            timeout=timeout,
            headers={"Accept": "application/json"},
        )
        if _is_json_body(resp.status_code, resp.headers.get("content-type"), resp.content):
            return resp.content
    logger.warning(
        "StockTwits %s via %s returned HTTP %s (%s)",
        url,
        profile,
        resp.status_code,
        resp.headers.get("content-type") or "unknown",
    )
    return None


def _urlopen_get(url: str, timeout: float) -> bytes:
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _get_bytes(url: str, timeout: float) -> bytes:
    """Fetch the symbol-stream body, preferring a CDN-accepted TLS fingerprint."""
    if _cf_requests is not None:
        for profile in _IMPERSONATE_PROFILES:
            try:
                body = _curl_cffi_get(url, timeout, profile)
            except (OSError, http.client.HTTPException) as exc:
                logger.warning("StockTwits %s via %s failed: %s", url, profile, exc)
                continue
            if body is not None:
                return body
    return _urlopen_get(url, timeout)


def _unavailable(exc: BaseException) -> str:
    if isinstance(exc, HTTPError):
        return f"<stocktwits unavailable: HTTPError {exc.code}>"
    return f"<stocktwits unavailable: {type(exc).__name__}>"


def fetch_stocktwits_messages(ticker: str, limit: int = 30, timeout: float = 10.0) -> str:
    """Fetch recent StockTwits messages for ``ticker`` and return them as a
    formatted plaintext block ready for prompt injection.

    Returns a placeholder string when the endpoint is unreachable, the
    symbol has no messages, or the response shape is unexpected — the
    caller never has to special-case None or exceptions.
    """
    url = _API.format(ticker=_stocktwits_symbol(ticker))
    if limit:
        url = f"{url}?limit={int(limit)}"
    try:
        data = json.loads(_get_bytes(url, timeout))
    except (OSError, http.client.HTTPException, json.JSONDecodeError) as exc:
        # OSError covers URLError/TimeoutError/connection resets/HTTPError;
        # HTTPException covers chunked-transfer errors (IncompleteRead, #1024).
        logger.warning("StockTwits fetch failed for %s: %s", ticker, exc)
        return _unavailable(exc)

    messages = data.get("messages", []) if isinstance(data, dict) else []
    if not messages:
        return f"<no StockTwits messages found for ${ticker.upper()}>"

    lines = []
    bullish = bearish = unlabeled = 0
    for m in messages[:limit]:
        created = m.get("created_at", "")
        user = (m.get("user") or {}).get("username", "?")
        entities = m.get("entities") or {}
        sentiment_obj = entities.get("sentiment") or {}
        sentiment = sentiment_obj.get("basic") if isinstance(sentiment_obj, dict) else None
        body = html.unescape((m.get("body") or "").replace("\n", " ").strip())
        if len(body) > 280:
            body = body[:280] + "…"

        if sentiment == "Bullish":
            bullish += 1
            tag = "Bullish"
        elif sentiment == "Bearish":
            bearish += 1
            tag = "Bearish"
        else:
            unlabeled += 1
            tag = "no-label"
        lines.append(f"[{created} · @{user} · {tag}] {body}")

    total = bullish + bearish + unlabeled
    bull_pct = round(100 * bullish / total) if total else 0
    bear_pct = round(100 * bearish / total) if total else 0
    summary = (
        f"Bullish: {bullish} ({bull_pct}%) · "
        f"Bearish: {bearish} ({bear_pct}%) · "
        f"Unlabeled: {unlabeled} · "
        f"Total: {total} most-recent messages"
    )
    return summary + "\n\n" + "\n".join(lines)

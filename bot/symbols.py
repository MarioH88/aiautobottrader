"""Symbol classification and normalization for stocks and crypto."""

from __future__ import annotations

import re

_CRYPTO_QUOTE_SUFFIXES = ("USD", "USDT", "USDC", "EUR", "GBP")
_CRYPTO_SLASH_PATTERN = re.compile(r"^[A-Z0-9]{2,10}/[A-Z]{3,5}$")
_KNOWN_CRYPTO_BASES = {
    "BTC", "ETH", "SOL", "DOGE", "XRP", "ADA", "AVAX", "DOT", "LINK",
    "MATIC", "SHIB", "UNI", "LTC", "BCH", "ATOM", "NEAR", "PEPE", "ZEC",
}


def is_crypto_symbol(symbol: str) -> bool:
    """Return True when the symbol looks like a crypto pair (e.g. BTC/USD or BTCUSD)."""
    normalized = symbol.strip().upper().replace("-", "/")
    if "/" in normalized:
        return True
    if normalized in _KNOWN_CRYPTO_BASES:
        return True
    for suffix in _CRYPTO_QUOTE_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            return True
    return False


def normalize_crypto_symbol(symbol: str) -> str:
    """Normalize crypto tickers to Alpaca's slash format (e.g. BTCUSD -> BTC/USD)."""
    normalized = symbol.strip().upper().replace("-", "/")
    if normalized in _KNOWN_CRYPTO_BASES:
        return f"{normalized}/USD"
    if _CRYPTO_SLASH_PATTERN.match(normalized):
        return normalized
    if "/" in normalized:
        return normalized
    for suffix in _CRYPTO_QUOTE_SUFFIXES:
        if normalized.endswith(suffix) and len(normalized) > len(suffix):
            base = normalized[: -len(suffix)]
            return f"{base}/{suffix}"
    return normalized


def normalize_symbol(symbol: str) -> str:
    """Normalize any supported symbol for Alpaca market-data calls."""
    if is_crypto_symbol(symbol):
        return normalize_crypto_symbol(symbol)
    return symbol.strip().upper()


def to_trading_symbol(symbol: str) -> str:
    """Return the symbol form expected by Alpaca Trading API."""
    return normalize_symbol(symbol)


def news_aliases(symbol: str) -> list[str]:
    """Return news lookup aliases for a symbol (crypto often tagged as BTC/BTCUSD)."""
    raw = symbol.strip().upper().replace("-", "/")
    aliases = {raw, raw.replace("/", "")}
    if raw in _KNOWN_CRYPTO_BASES:
        aliases.update({raw, f"{raw}/USD", f"{raw}USD"})
    canonical = normalize_symbol(symbol)
    aliases.add(canonical)
    aliases.add(canonical.replace("/", ""))
    if is_crypto_symbol(canonical) and "/" in canonical:
        base, quote = canonical.split("/", 1)
        aliases.update({base, f"{base}USD", f"{base}/{quote}"})
    return sorted(aliases)


def canonical_symbol_key(symbol: str) -> str:
    """Stable key used for in-memory sentiment / position maps."""
    return normalize_symbol(symbol)


def partition_symbols(symbols: list[str]) -> tuple[list[str], list[str]]:
    """Split symbols into stock and crypto lists."""
    stocks: list[str] = []
    crypto: list[str] = []
    for symbol in symbols:
        if is_crypto_symbol(symbol):
            crypto.append(normalize_crypto_symbol(symbol))
        else:
            stocks.append(symbol.strip().upper())
    return stocks, crypto

# backend/brokers/exchange_config.py
"""
Exchange configuration and management for CCXT integration.

Supports multiple exchanges with different authentication methods,
testnet/sandbox modes, and trading features.
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExchangeType(str, Enum):
    """Supported exchange types."""
    BINANCE = "binance"
    BINANCE_US = "binanceus"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    OKX = "okx"
    KUCOIN = "kucoin"
    GATEIO = "gateio"
    HUOBI = "huobi"
    BITGET = "bitget"
    MEXC = "mexc"
    CRYPTOCOM = "cryptocom"


@dataclass
class ExchangeConfig:
    """Configuration for a single exchange."""
    exchange_id: str
    display_name: str
    has_spot: bool = True
    has_futures: bool = False
    has_margin: bool = False
    requires_password: bool = False  # For exchanges like KuCoin
    testnet_available: bool = False
    testnet_url: Optional[str] = None
    default_type: str = "spot"  # spot, future, margin
    rate_limit: int = 1200  # requests per minute
    min_order_value_usd: float = 10.0
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.001  # 0.1%
    withdrawal_enabled: bool = True
    supported_quote_currencies: List[str] = field(default_factory=lambda: ["USDT", "USD", "BTC", "ETH"])
    notes: str = ""


# Exchange configurations
EXCHANGE_CONFIGS: Dict[str, ExchangeConfig] = {
    ExchangeType.BINANCE: ExchangeConfig(
        exchange_id="binance",
        display_name="Binance",
        has_spot=True,
        has_futures=True,
        has_margin=True,
        testnet_available=True,
        testnet_url="https://testnet.binance.vision",
        rate_limit=1200,
        maker_fee=0.001,
        taker_fee=0.001,
        supported_quote_currencies=["USDT", "BUSD", "BTC", "ETH", "BNB"],
        notes="Largest exchange by volume. Not available in US."
    ),
    ExchangeType.BINANCE_US: ExchangeConfig(
        exchange_id="binanceus",
        display_name="Binance US",
        has_spot=True,
        has_futures=False,
        has_margin=False,
        testnet_available=False,
        rate_limit=1200,
        maker_fee=0.001,
        taker_fee=0.001,
        supported_quote_currencies=["USD", "USDT", "BTC", "ETH"],
        notes="US-compliant version of Binance."
    ),
    ExchangeType.COINBASE: ExchangeConfig(
        exchange_id="coinbase",
        display_name="Coinbase Pro",
        has_spot=True,
        has_futures=False,
        has_margin=False,
        testnet_available=True,
        testnet_url="https://api-public.sandbox.pro.coinbase.com",
        rate_limit=600,
        maker_fee=0.004,  # 0.4%
        taker_fee=0.006,  # 0.6%
        supported_quote_currencies=["USD", "USDT", "BTC", "ETH"],
        notes="US-friendly. Higher fees but high liquidity."
    ),
    ExchangeType.KRAKEN: ExchangeConfig(
        exchange_id="kraken",
        display_name="Kraken",
        has_spot=True,
        has_futures=True,
        has_margin=True,
        testnet_available=False,
        rate_limit=600,
        maker_fee=0.0016,  # 0.16%
        taker_fee=0.0026,  # 0.26%
        supported_quote_currencies=["USD", "EUR", "BTC", "ETH", "USDT"],
        notes="US-friendly. Good for fiat pairs."
    ),
    ExchangeType.BYBIT: ExchangeConfig(
        exchange_id="bybit",
        display_name="Bybit",
        has_spot=True,
        has_futures=True,
        has_margin=True,
        testnet_available=True,
        testnet_url="https://api-testnet.bybit.com",
        rate_limit=600,
        maker_fee=0.001,
        taker_fee=0.001,
        supported_quote_currencies=["USDT", "BTC", "ETH"],
        notes="Popular for derivatives trading."
    ),
    ExchangeType.OKX: ExchangeConfig(
        exchange_id="okx",
        display_name="OKX",
        has_spot=True,
        has_futures=True,
        has_margin=True,
        requires_password=True,
        testnet_available=True,
        rate_limit=600,
        maker_fee=0.0008,
        taker_fee=0.001,
        supported_quote_currencies=["USDT", "BTC", "ETH"],
        notes="Requires passphrase for API. Good derivatives platform."
    ),
    ExchangeType.KUCOIN: ExchangeConfig(
        exchange_id="kucoin",
        display_name="KuCoin",
        has_spot=True,
        has_futures=True,
        has_margin=True,
        requires_password=True,
        testnet_available=True,
        testnet_url="https://openapi-sandbox.kucoin.com",
        rate_limit=1200,
        maker_fee=0.001,
        taker_fee=0.001,
        supported_quote_currencies=["USDT", "BTC", "ETH", "KCS"],
        notes="Requires passphrase. Wide variety of altcoins."
    ),
    ExchangeType.GATEIO: ExchangeConfig(
        exchange_id="gateio",
        display_name="Gate.io",
        has_spot=True,
        has_futures=True,
        has_margin=True,
        testnet_available=False,
        rate_limit=600,
        maker_fee=0.002,
        taker_fee=0.002,
        supported_quote_currencies=["USDT", "BTC", "ETH"],
        notes="Large selection of altcoins."
    ),
    ExchangeType.CRYPTOCOM: ExchangeConfig(
        exchange_id="cryptocom",
        display_name="Crypto.com",
        has_spot=True,
        has_futures=False,
        has_margin=False,
        testnet_available=True,
        rate_limit=600,
        maker_fee=0.004,
        taker_fee=0.004,
        supported_quote_currencies=["USDT", "USD", "BTC", "CRO"],
        notes="Popular retail exchange with mobile app."
    ),
}


def get_exchange_config(exchange_id: str) -> Optional[ExchangeConfig]:
    """Get configuration for an exchange."""
    exchange_id = exchange_id.lower()
    return EXCHANGE_CONFIGS.get(exchange_id)


def get_all_exchanges() -> List[Dict[str, Any]]:
    """Get list of all supported exchanges."""
    return [
        {
            "id": config.exchange_id,
            "name": config.display_name,
            "has_spot": config.has_spot,
            "has_futures": config.has_futures,
            "has_margin": config.has_margin,
            "testnet_available": config.testnet_available,
            "maker_fee": config.maker_fee,
            "taker_fee": config.taker_fee,
            "notes": config.notes
        }
        for config in EXCHANGE_CONFIGS.values()
    ]


def get_configured_exchanges() -> List[str]:
    """Get list of exchanges that have credentials configured."""
    configured = []

    for exchange_id in EXCHANGE_CONFIGS.keys():
        prefix = exchange_id.upper()
        api_key = os.getenv(f'{prefix}_API_KEY')
        secret = os.getenv(f'{prefix}_SECRET')

        if api_key and secret:
            configured.append(exchange_id)

    return configured


def get_exchange_credentials(exchange_id: str) -> Dict[str, str]:
    """
    Get credentials for an exchange from environment variables.

    Returns empty dict if credentials not found.
    """
    exchange_id = exchange_id.lower()
    prefix = exchange_id.upper()

    credentials = {}

    api_key = os.getenv(f'{prefix}_API_KEY')
    secret = os.getenv(f'{prefix}_SECRET')

    if api_key and secret:
        credentials['apiKey'] = api_key
        credentials['secret'] = secret

        # Check for password/passphrase
        password = os.getenv(f'{prefix}_PASSWORD') or os.getenv(f'{prefix}_PASSPHRASE')
        if password:
            credentials['password'] = password

    return credentials


def is_testnet_enabled(exchange_id: str) -> bool:
    """Check if testnet/sandbox mode is enabled for an exchange."""
    exchange_id = exchange_id.lower()
    prefix = exchange_id.upper()
    return os.getenv(f'{prefix}_SANDBOX', 'false').lower() == 'true'


def get_ccxt_config(exchange_id: str) -> Dict[str, Any]:
    """
    Get full CCXT configuration for an exchange.

    Returns configuration dict ready to pass to CCXT exchange constructor.
    """
    exchange_id = exchange_id.lower()
    config = get_exchange_config(exchange_id)

    if not config:
        return {}

    ccxt_config = {
        'enableRateLimit': True,
        'rateLimit': 60000 / config.rate_limit,  # Convert to ms per request
        'options': {
            'defaultType': config.default_type,
        }
    }

    # Add credentials
    credentials = get_exchange_credentials(exchange_id)
    ccxt_config.update(credentials)

    # Check for sandbox mode
    if is_testnet_enabled(exchange_id) and config.testnet_available:
        ccxt_config['sandbox'] = True
        if config.testnet_url:
            ccxt_config['urls'] = {'api': config.testnet_url}

    return ccxt_config


# Common trading pairs across exchanges
COMMON_PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "ADA/USDT",
    "DOGE/USDT",
    "MATIC/USDT",
    "DOT/USDT",
    "AVAX/USDT",
    "LINK/USDT",
]


def normalize_symbol(symbol: str, exchange_id: str) -> str:
    """
    Normalize a trading symbol for a specific exchange.

    Different exchanges use different formats:
    - Binance: BTCUSDT
    - Coinbase: BTC-USD
    - Most CCXT: BTC/USDT
    """
    # Convert to standard CCXT format first
    symbol = symbol.upper().replace("-", "/").replace("_", "/")

    if "/" not in symbol:
        # Try to detect quote currency
        for quote in ["USDT", "USD", "BTC", "ETH", "BUSD", "EUR"]:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                symbol = f"{base}/{quote}"
                break

    return symbol


# Export all
__all__ = [
    'ExchangeType',
    'ExchangeConfig',
    'EXCHANGE_CONFIGS',
    'get_exchange_config',
    'get_all_exchanges',
    'get_configured_exchanges',
    'get_exchange_credentials',
    'is_testnet_enabled',
    'get_ccxt_config',
    'COMMON_PAIRS',
    'normalize_symbol',
]

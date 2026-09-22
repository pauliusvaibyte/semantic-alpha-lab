from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    db_path: str = os.getenv("SEMANTIC_ALPHA_DB", "./data/semantic_alpha.db")
    typesafe_api_key: str = os.getenv("TYPESAFE_API_KEY", "")
    x_api_key: str = os.getenv("X_API_KEY", "")
    bybit_base_url: str = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
    x_api_url: str = os.getenv(
        "X_API_URL", "https://api.twitterapi.io/twitter/tweet/advanced_search"
    )
    x_base_url: str = os.getenv("X_BASE_URL", "https://api.twitterapi.io")
    x_ws_url: str = os.getenv("X_WS_URL", "wss://ws.twitterapi.io/twitter/tweet/websocket")
    bybit_ws_url: str = os.getenv("BYBIT_WS_URL", "wss://stream.bybit.com/v5/public/linear")
    jev_model: str = os.getenv("JEV_MODEL", "jev-latest")
    jev_input_usd_per_m: float = float(os.getenv("JEV_INPUT_USD_PER_M", "0.042"))
    x_cost_per_post_usd: float = float(os.getenv("X_COST_PER_POST_USD", "0.00015"))
    fixed_monthly_data_usd: float = float(os.getenv("FIXED_MONTHLY_DATA_USD", "0"))
    # Accounts treated as deterministic official sources regardless of what the
    # semantic engine infers (prompt-injection-safe official_source signal).
    official_accounts: frozenset[str] = frozenset(
        a.strip().lower() for a in os.getenv("OFFICIAL_X_ACCOUNTS", "").split(",") if a.strip()
    )
    x_query_langs: str = os.getenv("X_QUERY_LANGS", "en")
    binance_base_url: str = os.getenv("BINANCE_BASE_URL", "https://fapi.binance.com")
    news_feeds: str = os.getenv("NEWS_FEEDS", "")


settings = Settings()

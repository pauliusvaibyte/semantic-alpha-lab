from __future__ import annotations

import asyncio
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

from ..schema import MarketSnapshot


def _f(x: Any) -> float | None:
    if x in (None, ""):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _topic_family(topic: str) -> str | None:
    for fam in ("tickers", "orderbook", "publicTrade", "allLiquidation"):
        if topic.startswith(fam + "."):
            return fam
    return None


@dataclass
class MicrostructureAccumulator:
    """Rolling microstructure state with per-topic source freshness.

    Every applied message records (a) the exchange event timestamp and (b) the
    local receive time. A snapshot is only *fresh* when the ticker and orderbook
    topics have been updated recently — a disconnected feed keeps its last state
    but can no longer produce fresh-looking rows.
    """

    symbol: str
    ticker: dict[str, Any] = field(default_factory=dict)
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    trades: deque[tuple[datetime, str, float, float]] = field(default_factory=lambda: deque(maxlen=10000))
    liquidations: deque[tuple[datetime, str, float, float]] = field(default_factory=lambda: deque(maxlen=5000))
    # Per-topic source liveness: family -> last local receive time.
    last_update: dict[str, datetime] = field(default_factory=dict)
    # Latest exchange event timestamp observed per family (message.ts / data ts).
    last_exchange_ts: dict[str, datetime] = field(default_factory=dict)
    # Connection telemetry maintained by the reader loop.
    connected: bool = False
    last_connected_at: datetime | None = None
    last_disconnect_at: datetime | None = None
    reconnects: int = 0
    messages_received: int = 0
    last_message_at: datetime | None = None

    def note_connected(self, ts: datetime | None = None) -> None:
        ts = ts or datetime.now(timezone.utc)
        if not self.connected:
            self.reconnects += 1 if self.last_connected_at is not None else 0
        self.connected = True
        self.last_connected_at = ts

    def note_disconnected(self, ts: datetime | None = None) -> None:
        self.connected = False
        self.last_disconnect_at = ts or datetime.now(timezone.utc)

    def topic_age(self, family: str, now: datetime | None = None) -> float | None:
        """Seconds since the last message for a topic family; None if never seen."""
        last = self.last_update.get(family)
        if last is None:
            return None
        return max(0.0, ((now or datetime.now(timezone.utc)) - last).total_seconds())

    def freshness(self, now: datetime | None = None) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        ages = {fam: self.topic_age(fam, now) for fam in ("tickers", "orderbook", "publicTrade", "allLiquidation")}
        return {
            "connected": self.connected,
            "reconnects": self.reconnects,
            "messages_received": self.messages_received,
            "ticker_age_s": ages["tickers"],
            "book_age_s": ages["orderbook"],
            "trade_age_s": ages["publicTrade"],
            "liquidation_age_s": ages["allLiquidation"],
            "last_exchange_ts": {k: v.isoformat() for k, v in self.last_exchange_ts.items()},
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
        }

    def is_fresh(self, now: datetime | None = None, *, max_ticker_age_s: float = 10.0, max_book_age_s: float = 30.0) -> bool:
        """Hard freshness gate for persisting a snapshot.

        Ticker carries price/funding/OI; the book carries spread/depth. Trades and
        liquidations are legitimately quiet, so they inform but do not gate.
        """
        ticker_age = self.topic_age("tickers", now)
        book_age = self.topic_age("orderbook", now)
        if ticker_age is None or book_age is None:
            return False
        return ticker_age <= max_ticker_age_s and book_age <= max_book_age_s

    def apply(self, message: dict[str, Any]) -> None:
        recv = datetime.now(timezone.utc)
        topic = str(message.get("topic") or "")
        fam = _topic_family(topic)
        if fam is None:
            return
        self.messages_received += 1
        self.last_message_at = recv
        self.last_update[fam] = recv
        ex = message.get("ts")
        if ex is not None:
            try:
                self.last_exchange_ts[fam] = datetime.fromtimestamp(float(ex) / 1000, tz=timezone.utc)
            except (TypeError, ValueError, OSError):
                pass

        if fam == "tickers":
            data = message.get("data") or {}
            if isinstance(data, list):
                data = data[0] if data else {}
            if message.get("type") == "snapshot":
                self.ticker = dict(data)
            else:
                self.ticker.update(data)
            return

        if fam == "orderbook":
            data = message.get("data") or {}
            if message.get("type") == "snapshot":
                self.bids.clear(); self.asks.clear()
            self._apply_book_side(self.bids, data.get("b") or [])
            self._apply_book_side(self.asks, data.get("a") or [])
            return

        if fam == "publicTrade":
            for t in message.get("data") or []:
                ts = datetime.fromtimestamp(float(t.get("T", message.get("ts", 0))) / 1000, tz=timezone.utc)
                self.trades.append((ts, str(t.get("S") or ""), float(t.get("v") or 0), float(t.get("p") or 0)))
            return

        if fam == "allLiquidation":
            data = message.get("data") or []
            if isinstance(data, dict):
                data = [data]
            for x in data:
                ts = datetime.fromtimestamp(float(x.get("T", message.get("ts", 0))) / 1000, tz=timezone.utc)
                self.liquidations.append((ts, str(x.get("S") or ""), float(x.get("v") or 0), float(x.get("p") or 0)))

    @staticmethod
    def _apply_book_side(book: dict[float, float], rows: list[list[str]]) -> None:
        for row in rows:
            if len(row) < 2:
                continue
            p, q = float(row[0]), float(row[1])
            if q == 0:
                book.pop(p, None)
            else:
                book[p] = q

    def snapshot(self, now: datetime | None = None, horizon_seconds: int = 60) -> MarketSnapshot:
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=horizon_seconds)
        while self.trades and self.trades[0][0] < cutoff:
            self.trades.popleft()
        while self.liquidations and self.liquidations[0][0] < cutoff:
            self.liquidations.popleft()

        bid = max(self.bids) if self.bids else _f(self.ticker.get("bid1Price"))
        ask = min(self.asks) if self.asks else _f(self.ticker.get("ask1Price"))
        last = _f(self.ticker.get("lastPrice")) or ((bid + ask) / 2 if bid and ask else 0.0)
        mid = ((bid + ask) / 2) if bid and ask else last
        spread_bps = ((ask - bid) / mid * 10_000) if bid and ask and mid else None
        bid_depth = sum(p * q for p, q in self.bids.items() if mid and p >= mid * (1 - .001))
        ask_depth = sum(p * q for p, q in self.asks.items() if mid and p <= mid * (1 + .001))
        denom = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / denom if denom else None

        buy_notional = sum(q * p for _, side, q, p in self.trades if side.lower() == "buy")
        sell_notional = sum(q * p for _, side, q, p in self.trades if side.lower() == "sell")
        total_trade = buy_notional + sell_notional
        taker_buy_ratio = buy_notional / total_trade if total_trade else None

        # Bybit liquidation side denotes the liquidated position: Buy=long liquidation, Sell=short liquidation.
        long_liq = sum(q * p for _, side, q, p in self.liquidations if side.lower() == "buy")
        short_liq = sum(q * p for _, side, q, p in self.liquidations if side.lower() == "sell")

        feed = self.freshness(now)
        exchange_ts = feed["last_exchange_ts"].get("tickers")
        mark = _f(self.ticker.get("markPrice"))
        index = _f(self.ticker.get("indexPrice"))
        # Bybit V5 tickers do not carry a basis field; derive it from mark/index
        # the same way the Binance snapshot does.
        basis = _f(self.ticker.get("basis")) or _f(self.ticker.get("basisRate")) or (mark / index - 1.0 if mark and index else None)
        return MarketSnapshot(
            source="bybit_ws",
            symbol=self.symbol,
            ts=now,
            last=last,
            bid=bid,
            ask=ask,
            mark=mark,
            index=index,
            funding_rate=_f(self.ticker.get("fundingRate")),
            open_interest_qty=_f(self.ticker.get("openInterest")),
            open_interest_value=_f(self.ticker.get("openInterestValue")),
            volume_24h=_f(self.ticker.get("volume24h")),
            turnover_24h=_f(self.ticker.get("turnover24h")),
            basis=basis,
            spread_bps=spread_bps,
            book_imbalance=imbalance,
            depth_bid_10bps=bid_depth,
            depth_ask_10bps=ask_depth,
            taker_buy_ratio=taker_buy_ratio,
            trade_notional_1m=total_trade,
            signed_trade_notional_1m=buy_notional - sell_notional,
            liquidation_long_usd_1m=long_liq,
            liquidation_short_usd_1m=short_liq,
            raw={
                "trades_1m": len(self.trades), "liquidations_1m": len(self.liquidations),
                "exchange_ts": exchange_ts, "received_ts": feed["last_message_at"],
                "fresh": self.is_fresh(now), "feed": feed,
            },
        )


class BybitLinearStream:
    """Public Bybit linear WebSocket stream.

    Topics: ticker, 50-level orderbook, public trades, all liquidations.
    `messages()` is intentionally low-level; production collection can aggregate
    into fixed-interval MarketSnapshot records without storing every tick.
    Connection lifecycle events are yielded as ``{"_control": ...}`` dicts so
    consumers can track reconnects and gaps.
    """

    def __init__(self, symbols: list[str], url: str = "wss://stream.bybit.com/v5/public/linear"):
        self.symbols = [s.upper().replace("/", "") for s in symbols]
        self.url = url

    async def messages(self) -> AsyncIterator[dict[str, Any]]:
        try:
            import websockets
        except ImportError as e:
            raise RuntimeError("Install semantic-alpha-lab[stream] for WebSocket capture") from e

        args: list[str] = []
        for s in self.symbols:
            args += [f"tickers.{s}", f"orderbook.50.{s}", f"publicTrade.{s}", f"allLiquidation.{s}"]
        while True:
            try:
                async with websockets.connect(self.url, ping_interval=20, ping_timeout=20, max_size=8_000_000) as ws:
                    await ws.send(json.dumps({"op": "subscribe", "args": args}))
                    yield {"_control": "connected", "ts": datetime.now(timezone.utc).isoformat()}
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("topic"):
                            yield msg
            except asyncio.CancelledError:
                raise
            except Exception as e:
                yield {"_control": "disconnected", "ts": datetime.now(timezone.utc).isoformat(), "error": str(e)[:200]}
                await asyncio.sleep(2)

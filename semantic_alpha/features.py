from __future__ import annotations

import math
from collections import Counter
from datetime import datetime, timedelta

import numpy as np

from .clustering import batch_similarity
from .config import settings
from .schema import FeatureSnapshot, MarketSnapshot
from .storage import CANONICAL_MARKET_SOURCES, Store

INTENT = {"STRONG_LONG": 2.0, "LONG": 1.0, "NEUTRAL": 0.0, "REDUCE_LONG": -0.5, "SHORT": -1.0, "STRONG_SHORT": -2.0}

# Minimum trailing feature rows before residualization is trusted; below this
# the residual features stay 0 (warmup) rather than fitting noise.
RESIDUAL_MIN_SAMPLES = 100
RESIDUAL_HISTORY_LIMIT = 2000


def _z(x: float, arr: list[float]) -> float:
    if len(arr) < 5:
        return 0.0
    sd = float(np.std(arr, ddof=1))
    return (x - float(np.mean(arr))) / sd if sd > 1e-12 else 0.0


def _entropy_agreement(vals: list[str], weights: list[float] | None = None) -> float:
    if not vals:
        return 0.0
    if weights is None:
        c = Counter(vals)
        p = np.array(list(c.values()), dtype=float) / len(vals)
        hmax = math.log(max(2, len(c)))
    else:
        c = Counter()
        for v, w in zip(vals, weights, strict=True):
            if w > 0:
                c[v] += w
        tot = sum(c.values())
        if tot <= 0:
            return 0.0
        p = np.array(list(c.values()), dtype=float) / tot
        hmax = math.log(max(2, len(c)))
    h = -float(np.sum(p * np.log(p + 1e-12)))
    return 1.0 - min(1.0, h / hmax) if hmax > 0 else 1.0


# Jev's own relevance verdict gates how much a post's semantics contribute to
# aggregate features. The X rules match ecosystem-keyword noise (e.g. Solana
# memecoin spam is ~60% of the SOL feed); an IRRELEVANT post's
# new_information_prob is not new information about the asset.
RELEVANCE_WEIGHT = {"RELEVANT": 1.0, "MAYBE": 0.5, "IRRELEVANT": 0.0}


class FeatureEngine:
    def __init__(self, store: Store, semantic_model: str | None = None, feature_version: str = "v4"):
        self.store = store
        self.semantic_model = semantic_model
        self.feature_version = feature_version

    def build(self, symbol: str, ts: datetime, market: MarketSnapshot | None = None) -> FeatureSnapshot:
        vals: dict[str, float] = {}
        for minutes in (1, 5, 15):
            start = ts - timedelta(minutes=minutes)
            # Point-in-time correctness: a post only counts once it was actually
            # observed (first_seen_at <= ts) and its semantics classified (<= ts).
            posts = self.store.posts_between(symbol, start, ts, as_of=ts)
            sem = self.store.semantics_between(symbol, start, ts, self.semantic_model, as_of=ts)
            vals[f"posts_{minutes}m"] = float(len(posts))
            vals[f"authors_{minutes}m"] = float(len({p.author_username for p in posts if p.author_username}))
            vals[f"author_velocity_{minutes}m"] = vals[f"authors_{minutes}m"] / minutes
            # Deterministic source signals — cannot be prompt-injected.
            vals[f"verified_author_rate_{minutes}m"] = float(np.mean([bool(p.verified) for p in posts])) if posts else 0.0
            vals[f"official_account_rate_{minutes}m"] = float(np.mean([(p.author_username or "").lower() in settings.official_accounts for p in posts])) if posts else 0.0
            if posts:
                lags=[max(0.0,(p.first_seen_at-p.created_at).total_seconds()) for p in posts]
                vals[f"first_seen_lag_mean_s_{minutes}m"] = float(np.mean(lags))
                vals[f"forward_observation_rate_{minutes}m"] = float(np.mean([x <= 300 for x in lags]))
            else:
                vals[f"first_seen_lag_mean_s_{minutes}m"] = 0.0
                vals[f"forward_observation_rate_{minutes}m"] = 1.0
            if sem:
                post_lookup={p.post_id:p for p in posts}
                live=[]
                for x in sem:
                    p0=post_lookup.get(x.post_id)
                    if p0 is not None:
                        live.append((x.classified_at-p0.first_seen_at).total_seconds() <= 300)
                vals[f"semantic_live_rate_{minutes}m"] = float(np.mean(live)) if live else 0.0
                # Relevance-weighted aggregates: Jev's IRRELEVANT verdict removes
                # a post from semantic features entirely; MAYBE counts at half.
                wts = [RELEVANCE_WEIGHT.get(s.relevance, 0.0) for s in sem]
                wsum = float(np.sum(wts))
                vals[f"rel_posts_{minutes}m"] = wsum
                vals[f"rel_share_{minutes}m"] = float(wsum / len(sem))
                def wm(xs: list[float], wts: list[float] = wts, wsum: float = wsum) -> float:
                    return float(np.dot(wts, xs) / wsum) if wsum > 0 else 0.0
                intent = [INTENT.get(s.trade_intent, 0.0) for s in sem]
                vals[f"intent_mean_{minutes}m"] = wm(intent)
                vals[f"intent_sum_{minutes}m"] = float(np.dot(wts, intent))
                vals[f"explicit_signal_rate_{minutes}m"] = wm([s.explicit_recommendation_prob for s in sem])
                vals[f"new_info_rate_{minutes}m"] = wm([s.new_information_prob for s in sem])
                vals[f"reactive_rate_{minutes}m"] = wm([s.reactive_to_price_prob for s in sem])
                vals[f"promo_rate_{minutes}m"] = wm([s.promotional_prob for s in sem])
                vals[f"shill_rate_{minutes}m"] = wm([s.coordinated_shill_prob for s in sem])
                vals[f"evidence_rate_{minutes}m"] = wm([s.evidence_prob for s in sem])
                vals[f"impact_mean_{minutes}m"] = wm([s.market_impact_score for s in sem])
                vals[f"intent_agreement_{minutes}m"] = _entropy_agreement([s.trade_intent for s in sem], wts)
                vals[f"primary_source_rate_{minutes}m"] = wm([float(s.source_role == "PRIMARY_OFFICIAL") for s in sem])
                vals[f"research_source_rate_{minutes}m"] = wm([float(s.source_role == "JOURNALIST_RESEARCHER") for s in sem])
                vals[f"rumor_rate_{minutes}m"] = wm([float(s.claim_status == "RUMOR") for s in sem])
                vals[f"corroborated_rate_{minutes}m"] = wm([float(s.claim_status in {"FIRSTHAND","CORROBORATED"}) for s in sem])
                vals[f"semantic_confidence_mean_{minutes}m"] = wm([s.confidence for s in sem])
                vals[f"official_new_info_impact_{minutes}m"] = wm([s.new_information_prob * s.market_impact_score * (1.0 if s.source_role == "PRIMARY_OFFICIAL" else 0.0) for s in sem])
                vals[f"rumor_impact_{minutes}m"] = wm([s.market_impact_score * (1.0 if s.claim_status == "RUMOR" else 0.0) for s in sem])
                for cat in ("LISTING","DELISTING","SECURITY","REGULATION","ETF","PARTNERSHIP","TOKEN_UNLOCK","PROTOCOL","WHALE_FLOW","MACRO","TECHNICAL"):
                    vals[f"catalyst_{cat.lower()}_rate_{minutes}m"] = wm([float(s.catalyst_type == cat) for s in sem])
                for typ in ("EXPLICIT_TRADE_SIGNAL","CATALYST_NEWS","MARKET_ANALYSIS","PRICE_REACTION","PROMOTION"):
                    vals[f"type_{typ.lower()}_rate_{minutes}m"] = wm([float(s.communication_type == typ) for s in sem])
            else:
                vals[f"semantic_live_rate_{minutes}m"] = 1.0 if not posts else 0.0
                vals[f"rel_posts_{minutes}m"] = 0.0
                vals[f"rel_share_{minutes}m"] = 0.0
                for n in ["intent_mean","intent_sum","explicit_signal_rate","new_info_rate","reactive_rate","promo_rate","shill_rate","evidence_rate","impact_mean","intent_agreement","primary_source_rate","research_source_rate","rumor_rate","corroborated_rate","semantic_confidence_mean","official_new_info_impact","rumor_impact"]:
                    vals[f"{n}_{minutes}m"] = 0.0
                for cat in ("LISTING","DELISTING","SECURITY","REGULATION","ETF","PARTNERSHIP","TOKEN_UNLOCK","PROTOCOL","WHALE_FLOW","MACRO","TECHNICAL"):
                    vals[f"catalyst_{cat.lower()}_rate_{minutes}m"] = 0.0
                for typ in ("EXPLICIT_TRADE_SIGNAL","CATALYST_NEWS","MARKET_ANALYSIS","PRICE_REACTION","PROMOTION"):
                    vals[f"type_{typ.lower()}_rate_{minutes}m"] = 0.0

        # Attention surprise: current 5m count versus non-overlapping 5m buckets in trailing 24h.
        # Pull the trailing window once instead of issuing ~288 DB queries per feature snapshot.
        current = vals["posts_5m"]
        hist_start = ts - timedelta(hours=24)
        hist_end = ts - timedelta(minutes=5)
        historical_posts = self.store.posts_between(symbol, hist_start, hist_end, as_of=ts)
        bucket_count = max(1, int((hist_end - hist_start).total_seconds() // 300))
        counts = [0.0] * bucket_count
        for p in historical_posts:
            idx = int((p.created_at - hist_start).total_seconds() // 300)
            if 0 <= idx < bucket_count:
                counts[idx] += 1.0
        vals["attention_z_5m"] = _z(current, counts)
        vals["attention_accel"] = vals["posts_1m"] * 5.0 - vals["posts_5m"]

        self._engagement_features(symbol, ts, vals)
        self._narrative_features(symbol, ts, vals)
        self._author_features(symbol, ts, vals)

        market = market or self.store.latest_market(symbol)
        if market:
            vals.update(_market_features(market))
            self._market_history_features(symbol, ts, market, vals)
        for minutes in (1, 5, 15, 30, 60):
            vals.setdefault(f"return_{minutes}m", 0.0)
            vals.setdefault(f"oi_change_{minutes}m", 0.0)
        vals.setdefault("realized_vol_15m", 0.0)
        vals.setdefault("flow_imbalance_mean_5m", 0.0)
        vals.setdefault("book_imbalance_mean_5m", 0.0)
        vals.setdefault("spread_mean_5m", 0.0)
        vals.setdefault("spread_max_5m", 0.0)
        vals.setdefault("liquidation_imbalance_mean_5m", 0.0)

        semantic_shock = (
            vals["attention_z_5m"] * 0.25
            + vals["intent_sum_5m"] * 0.10
            + vals["new_info_rate_5m"] * 1.5
            + vals["impact_mean_5m"] * 1.0
            + vals["primary_source_rate_5m"] * 0.8
            + vals["corroborated_rate_5m"] * 0.6
            + vals["narrative_novelty_15m"] * 0.9
            + vals["diffusion_velocity_15m"] * 0.05
            - vals["reactive_rate_5m"] * 0.75
            - vals["promo_rate_5m"] * 0.5
            - vals["shill_rate_5m"] * 0.5
            - vals["rumor_rate_5m"] * 0.5
        )
        reaction = abs(vals["return_30m"]) * 100.0 + abs(vals["oi_change_30m"]) * 20.0
        vals["semantic_shock"] = float(semantic_shock)
        vals["information_price_gap_proxy"] = float(semantic_shock - reaction)
        vals["crowding_proxy"] = float(max(0.0, vals["intent_sum_5m"]) * max(0.0, vals.get("funding_rate", 0.0)) * 10_000 + max(0.0, vals["oi_change_30m"]) * 10)
        vals["panic_squeeze_proxy"] = float(max(0.0, -vals["intent_sum_5m"]) * max(0.0, -vals.get("funding_rate", 0.0)) * 10_000 + max(0.0, vals.get("liquidation_short_usd_1m", 0.0)) / 1_000_000)
        prev5 = self.store.feature_at_or_before(symbol, ts - timedelta(minutes=5), self.feature_version, max_delay_seconds=180)
        if prev5:
            vals["semantic_shock_delta_5m"] = float(vals["semantic_shock"] - prev5.values.get("semantic_shock", 0.0))
            vals["attention_z_delta_5m"] = float(vals["attention_z_5m"] - prev5.values.get("attention_z_5m", 0.0))
            vals["gap_delta_5m"] = float(vals["information_price_gap_proxy"] - prev5.values.get("information_price_gap_proxy", 0.0))
        else:
            vals["semantic_shock_delta_5m"] = 0.0
            vals["attention_z_delta_5m"] = 0.0
            vals["gap_delta_5m"] = 0.0
        vals["reaction_per_semantic_unit_5m"] = float(abs(vals.get("return_5m", 0.0)) / (abs(vals["semantic_shock"]) + 0.5))
        self._residual_features(symbol, ts, vals)
        self._venue_features(symbol, ts, vals)
        return FeatureSnapshot(symbol=symbol, ts=ts, feature_version=self.feature_version, values=vals)

    def _residual_features(self, symbol: str, ts: datetime, vals: dict[str, float]) -> None:
        """Social activity unexplained by the market state that caused it.

        Reverse-causality control (price→posts): fit E[social|market] on the
        trailing feature history strictly before ts, then emit the residual.
        Posts echoing a 4% move produce residual ≈ 0; only genuinely surprising
        social shocks remain.
        """
        targets = {
            "attention_residual_5m": ("attention_z_5m", None),
            "posts_residual_5m": ("posts_5m", math.log1p),
            "intent_residual_5m": ("intent_sum_5m", None),
            "semantic_shock_residual": ("semantic_shock", None),
        }
        for out in targets:
            vals[out] = 0.0
        rows = self.store.features_before(symbol, ts, self.feature_version, limit=RESIDUAL_HISTORY_LIMIT)
        if len(rows) < RESIDUAL_MIN_SAMPLES:
            return

        def mvec(v: dict[str, float]) -> list[float]:
            return [
                abs(v.get("return_1m", 0.0)), abs(v.get("return_5m", 0.0)), abs(v.get("return_15m", 0.0)),
                math.log1p(max(0.0, v.get("trade_notional_1m", 0.0))),
                math.log1p(max(0.0, v.get("liquidation_long_usd_1m", 0.0)) + max(0.0, v.get("liquidation_short_usd_1m", 0.0))),
                abs(v.get("oi_change_5m", 0.0)), abs(v.get("realized_vol_15m", 0.0)),
            ]

        X = np.array([mvec(r.values) for r in rows], dtype=float)
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-9] = 1.0
        Xb = np.c_[np.ones(len(X)), (X - mu) / sd]
        A = Xb.T @ Xb
        A[1:, 1:] += 1.0 * np.eye(Xb.shape[1] - 1)  # ridge, no penalty on intercept
        cur = np.r_[1.0, (np.array(mvec(vals)) - mu) / sd]
        for out, (src, tf) in targets.items():
            y = np.array([tf(r.values.get(src, 0.0)) if tf else r.values.get(src, 0.0) for r in rows], dtype=float)
            beta = np.linalg.solve(A, Xb.T @ y)
            obs = tf(vals.get(src, 0.0)) if tf else vals.get(src, 0.0)
            vals[out] = float(obs - cur @ beta)

    def _venue_features(self, symbol: str, ts: datetime, vals: dict[str, float]) -> None:
        """Cross-venue dispersion/lead-lag factors; zeros when only one venue exists."""
        vals.setdefault("venue_price_dispersion_bps", 0.0)
        vals.setdefault("venue_funding_dispersion", 0.0)
        vals.setdefault("venue_basis_dispersion", 0.0)
        vals.setdefault("bybit_vs_venues_return_gap_5m", 0.0)
        venues = self.store.markets_near(symbol, ts, max_delay_seconds=180)
        lasts = {s: m.last for s, m in venues.items() if m.last}
        if len(lasts) < 2:
            return
        lo, hi = min(lasts.values()), max(lasts.values())
        if lo > 0:
            vals["venue_price_dispersion_bps"] = float((hi / lo - 1.0) * 10_000)
        fundings = [m.funding_rate for m in venues.values() if m.funding_rate is not None]
        if len(fundings) >= 2:
            vals["venue_funding_dispersion"] = float(max(fundings) - min(fundings))
        bases = [m.basis for m in venues.values() if m.basis is not None]
        if len(bases) >= 2:
            vals["venue_basis_dispersion"] = float(max(bases) - min(bases))
        # The "primary" venue is whichever canonical Bybit source is present —
        # live-only operation writes bybit_ws, never source="bybit", so keying
        # on the literal name left this gap permanently zero.
        canon_src = next((s for s in CANONICAL_MARKET_SOURCES if s in venues), None)
        bybit = venues.get(canon_src) if canon_src else None
        if bybit and bybit.last:
            prev_b = self.store.nearest_market_at_or_before(symbol, ts - timedelta(minutes=5), source=canon_src)
            if prev_b and prev_b.last:
                br = bybit.last / prev_b.last - 1.0
                others = []
                for src, m in venues.items():
                    if src in CANONICAL_MARKET_SOURCES or not m.last:
                        continue
                    pm = self.store.nearest_market_at_or_before(symbol, ts - timedelta(minutes=5), source=src)
                    if pm and pm.last:
                        others.append(m.last / pm.last - 1.0)
                if others:
                    vals["bybit_vs_venues_return_gap_5m"] = float(br - float(np.mean(others)))

    def _market_history_features(self, symbol: str, ts: datetime, market: MarketSnapshot, vals: dict[str, float]) -> None:
        for minutes in (1, 5, 15, 30, 60):
            prev = self.store.nearest_market_at_or_before(symbol, ts - timedelta(minutes=minutes), max_delay_seconds=180)
            if prev and prev.last:
                vals[f"return_{minutes}m"] = float(market.last / prev.last - 1.0)
                if market.open_interest_value is not None and prev.open_interest_value:
                    vals[f"oi_change_{minutes}m"] = float(market.open_interest_value / prev.open_interest_value - 1.0)

        hist = self.store.market_between(symbol, ts - timedelta(minutes=15), ts + timedelta(microseconds=1))
        if not hist:
            return
        # Downsample to one last price per minute before realized-vol calculation so a
        # 1-second recorder does not artificially create 60x more observations.
        by_minute: dict[datetime, float] = {}
        for m in hist:
            k = m.ts.replace(second=0, microsecond=0)
            if m.last:
                by_minute[k] = float(m.last)
        prices = list(by_minute.values())
        if len(prices) >= 3:
            lr = np.diff(np.log(np.asarray(prices, dtype=float)))
            vals["realized_vol_15m"] = float(np.std(lr, ddof=1)) if len(lr) >= 2 else 0.0

        last5 = [m for m in hist if m.ts >= ts - timedelta(minutes=5)]
        if last5:
            flow = []
            book = []
            spreads = []
            liq = []
            for m in last5:
                tn = float(m.trade_notional_1m or 0.0)
                sn = float(m.signed_trade_notional_1m or 0.0)
                if tn > 0:
                    flow.append(sn / tn)
                if m.book_imbalance is not None:
                    book.append(float(m.book_imbalance))
                if m.spread_bps is not None:
                    spreads.append(float(m.spread_bps))
                long_liq = float(m.liquidation_long_usd_1m or 0.0)
                short_liq = float(m.liquidation_short_usd_1m or 0.0)
                tot = long_liq + short_liq
                if tot > 0:
                    liq.append((short_liq - long_liq) / tot)
            vals["flow_imbalance_mean_5m"] = float(np.mean(flow)) if flow else 0.0
            vals["book_imbalance_mean_5m"] = float(np.mean(book)) if book else 0.0
            vals["spread_mean_5m"] = float(np.mean(spreads)) if spreads else 0.0
            vals["spread_max_5m"] = float(np.max(spreads)) if spreads else 0.0
            vals["liquidation_imbalance_mean_5m"] = float(np.mean(liq)) if liq else 0.0

    def _engagement_features(self, symbol: str, ts: datetime, vals: dict[str, float]) -> None:
        posts = self.store.posts_between(symbol, ts - timedelta(minutes=15), ts, as_of=ts)
        velocities: list[float] = []
        early_totals: list[float] = []
        for p in posts:
            snaps = self.store.engagement_for_post(p.post_id, end=ts)
            if snaps:
                early_totals.append(float(snaps[-1].likes + 2 * snaps[-1].reposts + snaps[-1].replies))
            if len(snaps) >= 2:
                a, b = snaps[0], snaps[-1]
                dt = max((b.observed_at - a.observed_at).total_seconds() / 60.0, 1 / 60)
                e0 = a.likes + 2 * a.reposts + a.replies
                e1 = b.likes + 2 * b.reposts + b.replies
                velocities.append((e1 - e0) / dt)
        vals["engagement_velocity_mean_15m"] = float(np.mean(velocities)) if velocities else 0.0
        vals["engagement_velocity_max_15m"] = float(np.max(velocities)) if velocities else 0.0
        vals["early_engagement_mean_15m"] = float(np.mean(early_totals)) if early_totals else 0.0

    def _author_features(self, symbol: str, ts: datetime, vals: dict[str, float]) -> None:
        posts = self.store.posts_between(symbol, ts - timedelta(minutes=5), ts, as_of=ts)
        weighted: list[float] = []
        sample_counts: list[float] = []
        contrarian = 0
        known = 0
        for p in posts:
            if not p.author_username:
                continue
            rep = self.store.author_reputation_for(p.author_username, symbol, ts, "15m")
            if rep is None:
                continue
            sems = [x for x in self.store.semantics_for_post(p.post_id, self.semantic_model, symbol)
                    if x.classified_at <= ts]
            if not sems:
                continue
            sem = sems[-1]
            direction = np.sign(INTENT.get(sem.trade_intent, 0.0))
            if direction == 0 or sem.explicit_recommendation_prob < 0.5:
                continue
            known += 1
            sample_counts.append(float(rep.samples))
            weighted.append(float(direction * rep.posterior_edge * sem.explicit_recommendation_prob))
            if rep.posterior_edge < 0:
                contrarian += 1
        vals["author_alpha_signal_5m"] = float(np.sum(weighted)) if weighted else 0.0
        vals["author_alpha_mean_5m"] = float(np.mean(weighted)) if weighted else 0.0
        vals["author_history_samples_mean_5m"] = float(np.mean(sample_counts)) if sample_counts else 0.0
        vals["known_author_signal_count_5m"] = float(known)
        vals["contrarian_author_rate_5m"] = float(contrarian / known) if known else 0.0

    def _narrative_features(self, symbol: str, ts: datetime, vals: dict[str, float]) -> None:
        recent = self.store.events_between(symbol, ts - timedelta(minutes=15), ts)
        prior = self.store.events_between(symbol, ts - timedelta(hours=24), ts - timedelta(minutes=15))
        vals["events_15m"] = float(len(recent))
        vals["event_author_breadth_15m"] = float(sum(int(e.get("unique_authors", 0)) for e in recent))
        vals["event_source_domains_15m"] = float(sum(int(e.get("source_domains", 0)) for e in recent))
        velocities = []
        novelties = []
        # One vectorizer fit for the whole recent×prior comparison instead of a
        # fresh fit per pair — with ~7k prior events the pairwise version takes
        # minutes per build and starves the event loop (dropped X keepalives,
        # billed redeliveries). Prior is capped at the most recent 2k events to
        # bound the matrix as history grows.
        sim_matrix = None
        prior_capped = prior[-2000:]
        if recent and prior_capped:
            sim_matrix = batch_similarity(
                [e.get("representative_text", "") for e in recent]
                + [p.get("representative_text", "") for p in prior_capped]
            )
        for i, e in enumerate(recent):
            st = datetime.fromisoformat(str(e["started_at"]).replace("Z", "+00:00")) if isinstance(e.get("started_at"), str) else e.get("started_at")
            age = max((ts - st).total_seconds() / 60.0, 0.25) if st else 15.0
            velocities.append(float(e.get("unique_authors", 0)) / age)
            if sim_matrix is not None and sim_matrix.shape[0] == len(recent) + len(prior_capped):
                max_sim = float(sim_matrix[i, len(recent):].max()) if prior_capped else 0.0
                novelties.append(1.0 - max_sim)
            else:
                novelties.append(1.0 if recent else 0.0)
        vals["diffusion_velocity_15m"] = float(max(velocities)) if velocities else 0.0
        vals["narrative_novelty_15m"] = float(np.mean(novelties)) if novelties else 0.0


def _market_features(m: MarketSnapshot) -> dict[str, float]:
    long_liq = float(m.liquidation_long_usd_1m or 0.0)
    short_liq = float(m.liquidation_short_usd_1m or 0.0)
    liq_total = long_liq + short_liq
    return {
        "last": float(m.last),
        "spread_bps": float(m.spread_bps or 0.0),
        "funding_rate": float(m.funding_rate or 0.0),
        "open_interest_value": float(m.open_interest_value or 0.0),
        "book_imbalance": float(m.book_imbalance or 0.0),
        "depth_bid_10bps": float(m.depth_bid_10bps or 0.0),
        "depth_ask_10bps": float(m.depth_ask_10bps or 0.0),
        "taker_buy_ratio": float(m.taker_buy_ratio if m.taker_buy_ratio is not None else 0.5),
        "trade_notional_1m": float(m.trade_notional_1m or 0.0),
        "signed_trade_notional_1m": float(m.signed_trade_notional_1m or 0.0),
        "liquidation_long_usd_1m": long_liq,
        "liquidation_short_usd_1m": short_liq,
        "liquidation_imbalance_1m": float((short_liq - long_liq) / liq_total) if liq_total else 0.0,
        "basis": float(m.basis or 0.0),
        "volume_24h": float(m.volume_24h or 0.0),
    }

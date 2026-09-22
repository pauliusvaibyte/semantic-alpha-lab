from __future__ import annotations
import re
from ..schema import PostSemantics, SocialPost

LONG = {"buy", "buying", "long", "longing", "accumulate", "adding", "bullish", "breakout"}
SHORT = {"sell", "selling", "short", "shorting", "bearish", "exit", "dump", "take profit", "taking profit"}
PROMO = {"gem", "100x", "1000x", "ape", "moon", "join telegram", "referral"}
CATALYSTS = {
    "LISTING": ["listing", "listed on"], "DELISTING": ["delist"], "SECURITY": ["hack", "exploit", "breach"],
    "REGULATION": ["sec", "regulation", "lawsuit", "court"], "ETF": ["etf"],
    "PARTNERSHIP": ["partnership", "partnered"], "PROTOCOL": ["upgrade", "mainnet", "testnet"],
    "WHALE_FLOW": ["whale", "exchange inflow", "exchange outflow"], "TOKEN_UNLOCK": ["unlock"],
}

class HeuristicSemanticEngine:
    """Deterministic fallback used for tests and ablation baselines. Not intended as alpha."""
    model = "heuristic-v1"

    async def classify(self, post: SocialPost) -> PostSemantics:
        text = post.text.lower()
        tokens = set(re.findall(r"[a-z0-9$]+", text))
        long_hits = sum(1 for k in LONG if k in text or k in tokens)
        short_hits = sum(1 for k in SHORT if k in text or k in tokens)
        if long_hits > short_hits:
            intent = "LONG"
        elif short_hits > long_hits:
            intent = "SHORT" if "short" in text else "REDUCE_LONG"
        else:
            intent = "NEUTRAL"
        catalyst = "NONE"
        for name, kws in CATALYSTS.items():
            if any(k in text for k in kws):
                catalyst = name
                break
        explicit = 0.85 if any(x in text for x in ["i'm buying", "im buying", "long $", "short $", "buying $", "selling $"]) else (0.55 if intent != "NEUTRAL" else 0.05)
        promo = min(1.0, sum(1 for k in PROMO if k in text) / 2)
        new_info = 0.8 if catalyst != "NONE" else 0.2
        reactive = 0.75 if any(x in text for x in ["pumped", "dumped", "up today", "down today", "after the move"]) else 0.2
        ctype = "EXPLICIT_TRADE_SIGNAL" if explicit > 0.7 else ("CATALYST_NEWS" if catalyst != "NONE" else "MARKET_ANALYSIS")
        return PostSemantics(
            post_id=post.post_id, symbol=post.symbol, model=self.model,
            relevance="RELEVANT", communication_type=ctype, trade_intent=intent,
            catalyst_type=catalyst, horizon="HOURS",
            source_role=("PROMOTER" if promo > 0.5 else "TRADER_ANALYST"),
            claim_status=("SECONDHAND" if catalyst != "NONE" else "OPINION"),
            new_information_prob=new_info, reactive_to_price_prob=reactive,
            original_information_prob=0.5, explicit_recommendation_prob=explicit,
            evidence_prob=0.4 if post.urls else 0.15, promotional_prob=promo,
            coordinated_shill_prob=promo * 0.6, market_impact_score=0.7 if catalyst != "NONE" else 0.3,
            confidence=0.65,
        )

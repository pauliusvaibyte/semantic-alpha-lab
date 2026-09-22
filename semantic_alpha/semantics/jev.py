from __future__ import annotations
import json
import time as _time
from ..config import settings

from ..schema import PostSemantics, SocialPost


def is_provider_error(msg: str) -> bool:
    """Provider-side failure (billing, rate limit, transport, timeout) — says
    nothing about the post, so it must not count toward the per-post failure
    cap. Content/model-side errors DO count."""
    m = msg.lower()
    return any(k in m for k in (
        "402", "no available", "credits", "429", "rate limit", "timeout",
        "timed out", "503", "502", "500", "connection", "transport", "unavailable",
    ))


class JevSemanticEngine:
    """Classify a single post into independent, testable semantic variables."""

    question_version = "v2"

    def __init__(self, api_key: str, model: str = "jev-latest"):
        if not api_key:
            raise ValueError("TypeSafe API key required")
        self.api_key = api_key
        self.model = model
        # Populated after the first response: the provider's actual serving model.
        self.effective_model: str | None = None
        self._client = None

    async def _get_client(self):
        """Lazily create and reuse one SDK client. A fresh client per post pays
        a full connection/handshake on every classification, which saturates
        workers under burst load."""
        if self._client is None:
            from typesafe_sdk import AsyncTypeSafeClient
            self._client = await AsyncTypeSafeClient(api_key=self.api_key).__aenter__()
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try: await self._client.__aexit__(None, None, None)
            except Exception: pass
            self._client = None

    @property
    def query_model(self) -> str:
        """Model identity used for semantic-coverage lookups. Once the provider
        reports its effective model, coverage must check that — a post classified
        by jev-2025-01 is not covered by a query for the alias jev-latest."""
        return self.effective_model or self.model

    async def classify(self, post: SocialPost) -> PostSemantics:
        try:
            from typesafe_sdk import Choice, Noul, Score
        except ImportError as e:
            raise RuntimeError("Install semantic-alpha-lab[jev] to use Jev") from e

        state = {
            "asset": post.symbol,
            "post": {
                "text": post.text,
                "created_at": post.created_at.isoformat(),
                "author": post.author_username,
                "followers": post.followers,
                "verified": post.verified,
                "urls": post.urls,
            },
        }
        questions = {
            "relevance": Choice(
                instructions="Is this post materially about `asset` rather than incidental mention?",
                criteria={"RELEVANT":"Materially about asset.", "MAYBE":"Ambiguous/partial.", "IRRELEVANT":"Not materially about asset."},
            ),
            "communication_type": Choice(
                instructions="What is the primary information type of the post?",
                criteria={
                    "EXPLICIT_TRADE_SIGNAL":"Author expresses a concrete position/recommendation.",
                    "CATALYST_NEWS":"Reports potentially market-moving new event/news.",
                    "MARKET_ANALYSIS":"Analysis without explicit position.",
                    "PRICE_REACTION":"Mostly reacts to a move that already happened.",
                    "PROMOTION":"Promotional/shilling content.", "OTHER":"Other discussion.",
                },
            ),
            "trade_intent": Choice(
                instructions="What directional trading intent is actually expressed? Do not equate positive tone with a long recommendation.",
                criteria={
                    "STRONG_LONG":"High-conviction explicit long/buy/add.", "LONG":"Long/buy bias.",
                    "NEUTRAL":"No actionable directional intent.", "REDUCE_LONG":"Take profit/exit/reduce long exposure.",
                    "SHORT":"Short/sell bias.", "STRONG_SHORT":"High-conviction explicit short/sell.",
                },
            ),
            "catalyst_type": Choice(
                instructions="Classify the primary catalyst, if any.",
                criteria={
                    "LISTING":"Exchange listing.", "DELISTING":"Exchange delisting.", "SECURITY":"Hack/exploit/security issue.",
                    "REGULATION":"Regulatory/legal action.", "ETF":"ETF-related development.", "PARTNERSHIP":"Partnership/integration.",
                    "TOKEN_UNLOCK":"Token unlock/supply event.", "PROTOCOL":"Upgrade/mainnet/testnet/technical milestone.",
                    "WHALE_FLOW":"Large holder/exchange flow.", "MACRO":"Macro/geopolitical rates/liquidity.",
                    "TECHNICAL":"Technical-analysis setup only.", "NONE":"No catalyst.",
                },
            ),
            "horizon": Choice(
                instructions="What trading horizon does the post imply?",
                criteria={"MINUTES":"Minutes.","HOURS":"Hours/intraday.","DAYS":"Days.","WEEKS":"Weeks or longer.","NONE":"No horizon."},
            ),
            "source_role": Choice(
                instructions="What role is the author/source playing in this post? Judge the post, not follower count alone.",
                criteria={
                    "PRIMARY_OFFICIAL":"Project, exchange, regulator, company, or directly involved primary source.",
                    "JOURNALIST_RESEARCHER":"Reporter, researcher, investigator, or evidence-focused analyst.",
                    "TRADER_ANALYST":"Trader or market analyst expressing analysis/positioning.",
                    "SIGNAL_BOT":"Automated or systematic signal/news bot.",
                    "PROMOTER":"Primarily promotional/affiliate/shill role.",
                    "UNKNOWN":"Cannot determine.",
                },
            ),
            "claim_status": Choice(
                instructions="What is the epistemic status of the main claim?",
                criteria={
                    "FIRSTHAND":"Author appears to be the primary source/direct participant.",
                    "CORROBORATED":"Claim is supported by multiple/authoritative sources or direct evidence.",
                    "SECONDHAND":"Reports another source without independent corroboration.",
                    "RUMOR":"Unverified rumor/speculation/hearsay.",
                    "OPINION":"Interpretation or market opinion rather than factual claim.",
                    "NONE":"No meaningful claim.",
                },
            ),
            "new_information": Noul(instructions="Does this post contain genuinely new information rather than repetition/opinion?"),
            "reactive": Noul(instructions="Is this mostly reacting to a price move/event already visible rather than leading it?"),
            "original": Noul(instructions="Does the author appear to be an original/primary source rather than repeating another source?"),
            "explicit_recommendation": Noul(instructions="Does the author explicitly recommend or disclose a directional trade/position?"),
            "evidence": Noul(instructions="Does the post provide concrete evidence, source, data, or verifiable specifics?"),
            "promotional": Noul(instructions="Is this primarily promotional/shill content?"),
            "coordinated_shill": Noul(instructions="Does wording/style look like coordinated/template promotion rather than independent analysis?"),
            "impact": Score(
                instructions="If the claim is true, how large could its market impact on the asset be?",
                criteria=["Negligible", "Small", "Moderate", "Large", "Major"],
            ),
        }
        client = await self._get_client()
        # The configured model must actually be sent — otherwise metadata
        # claims self.model while the SDK silently runs its own default.
        try:
            response = await client.system_one(state=state, questions=questions, model=self.model)
        except TypeError:
            response = await client.system_one(state=state, questions=questions)

        # The provider reports the model that actually served the request; that
        # is the experimental identity, not the string we asked for.
        effective_model = str(
            getattr(response, "model", None)
            or getattr(response, "model_used", None)
            or self.model
        )
        self.effective_model = effective_model

        # Prefer provider-reported usage/cost when exposed by the SDK; otherwise use
        # a transparent character-based token estimate for operating-cost accounting.
        usage_obj=getattr(response,"usage",None)
        provider_tokens=None; provider_cost=None
        if usage_obj is not None:
            for name in ("input_tokens","prompt_tokens","inputTokenCount"):
                v=getattr(usage_obj,name,None)
                if v is not None:
                    try: provider_tokens=float(v); break
                    except (TypeError,ValueError): pass
            for name in ("cost","cost_usd","total_cost"):
                v=getattr(usage_obj,name,None)
                if v is not None:
                    try: provider_cost=float(v); break
                    except (TypeError,ValueError): pass
        estimated_tokens=float(provider_tokens or max(1,int((len(json.dumps(state,default=str))+2600)/4)))
        estimated_cost=float(provider_cost if provider_cost is not None else estimated_tokens*settings.jev_input_usd_per_m/1_000_000)
        # reference_id must identify one billed call: requested→effective model
        # plus call time. Re-saving the same stored record replays the same ref
        # (INSERT OR IGNORE stays idempotent); a genuine re-classification gets
        # a new ref so its spend is not silently absorbed by the first call.
        usage_meta={
            "provider":"typesafe","category":"semantic_classification",
            "reference_id":f"{post.post_id}:{post.symbol}:{self.model}->{effective_model}:{self.question_version}:{int(_time.time()*1000)}",
            "units":estimated_tokens,"unit_name":"input_token","estimated_usd":estimated_cost,
            "cost_source":"provider" if provider_cost is not None else ("provider_tokens_configured_rate" if provider_tokens is not None else "estimated_tokens_configured_rate"),
            "requested_model":self.model,"effective_model":effective_model,
        }

        probs: dict[str, dict[str, float]] = {}
        def choice(name: str, default: str) -> str:
            a = response.answers.get(name)
            p = getattr(a, "probabilities", None) or {}
            if p:
                probs[name] = {str(k): float(v) for k, v in p.items()}
            return str(getattr(a, "choice", default))
        def noul(name: str) -> float:
            return float(getattr(response.answers.get(name), "noul", 0.0))

        relevance = choice("relevance", "MAYBE")
        ctype = choice("communication_type", "OTHER")
        intent = choice("trade_intent", "NEUTRAL")
        catalyst = choice("catalyst_type", "NONE")
        horizon = choice("horizon", "NONE")
        source_role = choice("source_role", "UNKNOWN")
        claim_status = choice("claim_status", "NONE")
        impact_raw = float(getattr(response.answers.get("impact"), "score", 0.0))
        selected_conf = []
        for n in ["relevance", "communication_type", "trade_intent", "catalyst_type", "horizon", "source_role", "claim_status"]:
            a = response.answers.get(n)
            selected_conf.append(float(getattr(a, "confidence", 0.0)))
        confidence = min(selected_conf) if selected_conf else 0.0

        return PostSemantics(
            post_id=post.post_id, symbol=post.symbol, model=effective_model, question_version=self.question_version,
            relevance=relevance, communication_type=ctype, trade_intent=intent, catalyst_type=catalyst, horizon=horizon,
            source_role=source_role, claim_status=claim_status,
            new_information_prob=noul("new_information"), reactive_to_price_prob=noul("reactive"),
            original_information_prob=noul("original"), explicit_recommendation_prob=noul("explicit_recommendation"),
            evidence_prob=noul("evidence"), promotional_prob=noul("promotional"),
            coordinated_shill_prob=noul("coordinated_shill"), market_impact_score=impact_raw / 4.0 if impact_raw > 1 else impact_raw,
            confidence=confidence, probabilities=probs,
            raw={"answer_names": list(response.answers.keys()),"_usage":usage_meta,
                 "requested_model":self.model,"effective_model":effective_model},
        )

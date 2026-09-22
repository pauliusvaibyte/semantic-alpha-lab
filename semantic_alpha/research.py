from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import dynamic_costs, portfolio_metrics, stateful_trade_vectors, trade_metrics
from .labels import HORIZONS
from .storage import Store


FEATURE_GROUPS = {
    "price_derivatives": [
        "spread_bps", "funding_rate", "open_interest_value", "book_imbalance", "taker_buy_ratio", "basis",
        "return_1m", "return_5m", "return_15m", "return_30m", "return_60m",
        "oi_change_1m", "oi_change_5m", "oi_change_15m", "oi_change_30m", "oi_change_60m",
        "realized_vol_15m", "flow_imbalance_mean_5m", "book_imbalance_mean_5m",
        "spread_mean_5m", "spread_max_5m", "liquidation_imbalance_mean_5m",
        "depth_bid_10bps", "depth_ask_10bps", "trade_notional_1m",
        "signed_trade_notional_1m", "liquidation_long_usd_1m", "liquidation_short_usd_1m", "liquidation_imbalance_1m",
        "market_breadth_5m", "return_dispersion_5m", "relative_return_5m", "btc_return_5m", "eth_return_5m",
    ],
    "attention": [
        "posts_1m", "posts_5m", "posts_15m", "authors_5m", "attention_z_5m", "attention_accel",
        "engagement_velocity_mean_15m", "engagement_velocity_max_15m", "early_engagement_mean_15m",
        "cross_attention_mean", "cross_attention_std", "relative_attention_z", "attention_z_delta_5m",
    ],
    "semantics": [
        "intent_mean_5m", "intent_sum_5m", "explicit_signal_rate_5m", "new_info_rate_5m",
        "reactive_rate_5m", "promo_rate_5m", "shill_rate_5m", "evidence_rate_5m", "impact_mean_5m",
        "intent_agreement_5m", "events_15m", "event_author_breadth_15m", "event_source_domains_15m",
        "diffusion_velocity_15m", "narrative_novelty_15m", "primary_source_rate_5m", "research_source_rate_5m",
        "rumor_rate_5m", "corroborated_rate_5m", "semantic_shock",
        "verified_author_rate_5m", "official_account_rate_5m",
        "author_alpha_signal_5m", "author_alpha_mean_5m", "author_history_samples_mean_5m",
        "known_author_signal_count_5m", "contrarian_author_rate_5m",
        "cross_semantic_mean", "cross_semantic_std", "relative_semantic_z", "btc_semantic_shock", "eth_semantic_shock",
        "social_breadth_positive", "social_breadth_extreme",
        "semantic_shock_delta_5m", "semantic_confidence_mean_5m", "official_new_info_impact_5m", "rumor_impact_5m",
        "catalyst_listing_rate_5m", "catalyst_delisting_rate_5m", "catalyst_security_rate_5m", "catalyst_regulation_rate_5m",
        "catalyst_etf_rate_5m", "catalyst_partnership_rate_5m", "catalyst_token_unlock_rate_5m", "catalyst_protocol_rate_5m",
        "catalyst_whale_flow_rate_5m", "catalyst_macro_rate_5m", "catalyst_technical_rate_5m",
        "type_explicit_trade_signal_rate_5m", "type_catalyst_news_rate_5m", "type_market_analysis_rate_5m",
        "type_price_reaction_rate_5m", "type_promotion_rate_5m",
    ],
    "gap": [
        "information_price_gap_proxy", "crowding_proxy", "panic_squeeze_proxy", "gap_delta_5m", "reaction_per_semantic_unit_5m",
        # Residualized social shock: activity unexplained by the market move that may have caused it.
        "attention_residual_5m", "posts_residual_5m", "intent_residual_5m", "semantic_shock_residual",
        # Cross-venue dispersion/lead-lag (require a second market source to be nonzero).
        "venue_price_dispersion_bps", "venue_funding_dispersion", "venue_basis_dispersion", "bybit_vs_venues_return_gap_5m",
    ],
}


def dataset(store: Store, horizon: str = "15m", feature_version: str = "v4", label_version: str = "v4") -> pd.DataFrame:
    rows = []
    for f in store.all_features(feature_version):
        l = store.label_for(f.symbol, f.ts, label_version)
        if not l:
            continue
        y = l.abnormal_returns.get(horizon)
        if y is None:
            continue
        row = {
            "symbol": f.symbol, "ts": f.ts, "target_return": y,
            # Secondary v4 heads for magnitude/volatility/crowding models.
            "target_abs_move": l.abs_move.get(horizon),
            "target_range_move": l.range_move.get(horizon),
            "target_fade_ratio": l.fade_ratio.get(horizon),
        }
        row.update(f.values)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ts") if rows else pd.DataFrame()


@dataclass
class EvalResult:
    rows: int
    auc: float | None
    brier: float | None
    hit_rate: float | None
    mean_net_return: float | None
    trades: int
    net_return: float | None = None
    sharpe: float | None = None
    max_drawdown: float | None = None
    profit_factor: float | None = None
    # Portfolio-native accounting: equity-curve Sharpe/drawdown on a fixed time
    # grid, distinct from the per-trade stats above which ignore idle periods.
    portfolio_sharpe: float | None = None
    portfolio_max_drawdown: float | None = None
    portfolio_net_return: float | None = None
    avg_concurrent_positions: float | None = None


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], model_name: str = "logistic") -> np.ndarray:
    y = (train.target_return > 0).astype(int)
    if model_name == "histgb":
        model = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=17)
    else:
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(train[features], y)
    return model.predict_proba(test[features])[:, 1]


def _evaluate_split(train: pd.DataFrame, test: pd.DataFrame, features: list[str], *, model_name: str, cost_bps: float, dynamic_cost: bool, notional_usd: float, horizon_minutes: int = 15, max_positions: int = 3) -> EvalResult:
    ytr=(train.target_return>0).astype(int); yte=(test.target_return>0).astype(int)
    if ytr.nunique()<2 or yte.nunique()<2:
        return EvalResult(len(train)+len(test),None,None,None,None,0)
    prob=_fit_predict(train,test,features,model_name)
    sig=np.where(prob>=.60,1,np.where(prob<=.40,-1,0))
    gross=sig*test.target_return.to_numpy()
    costs = dynamic_costs(test, notional_usd=notional_usd) if dynamic_cost else np.full(len(test), cost_bps/10_000)
    net, mask = stateful_trade_vectors(test, sig, costs, horizon_minutes=horizon_minutes, max_positions=max_positions)
    metrics=trade_metrics(pd.Series(net[mask])) if mask.any() else trade_metrics(pd.Series(dtype=float))
    ledger = pd.DataFrame({
        "entry_ts": pd.to_datetime(test.ts, utc=True).to_numpy()[mask],
        "exit_ts": (pd.to_datetime(test.ts, utc=True) + pd.Timedelta(minutes=horizon_minutes)).to_numpy()[mask],
        "net_return": net[mask], "notional_usd": float(notional_usd),
    })
    port = portfolio_metrics(ledger, horizon_minutes=horizon_minutes)
    return EvalResult(
        rows=len(train)+len(test), auc=float(roc_auc_score(yte,prob)), brier=float(brier_score_loss(yte,prob)),
        hit_rate=float((gross[mask]>0).mean()) if mask.any() else None,
        mean_net_return=float(net[mask].mean()) if mask.any() else None, trades=int(mask.sum()),
        net_return=metrics["net_return"], sharpe=metrics["sharpe"], max_drawdown=metrics["max_drawdown"], profit_factor=metrics["profit_factor"],
        portfolio_sharpe=port["portfolio_sharpe"], portfolio_max_drawdown=port["portfolio_max_drawdown"],
        portfolio_net_return=port["portfolio_net_return"], avg_concurrent_positions=port["avg_concurrent_positions"],
    )


def evaluate_classifier(df: pd.DataFrame, features: list[str], cost_bps: float = 10.0, min_move_bps: float = 15.0,
                        dynamic_cost: bool = False, notional_usd: float = 1_000.0, model_name: str = "logistic",
                        horizon: str = "15m", max_positions: int = 3) -> EvalResult:
    del min_move_bps
    if len(df) < 50:
        return EvalResult(len(df), None, None, None, None, 0)
    d = df.dropna(subset=[*features, "target_return"]).copy()
    if len(d) < 50:
        return EvalResult(len(d), None, None, None, None, 0)
    split = int(len(d) * 0.7); train, test = d.iloc[:split], d.iloc[split:]
    return _evaluate_split(train,test,features,model_name=model_name,cost_bps=cost_bps,dynamic_cost=dynamic_cost,notional_usd=notional_usd,horizon_minutes=HORIZONS.get(horizon,15),max_positions=max_positions)


def ablation_ladder(df: pd.DataFrame, cost_bps: float = 10.0, dynamic_cost: bool = False, notional_usd: float = 1_000.0, model_name: str = "logistic", horizon: str = "15m") -> pd.DataFrame:
    stages = [
        ("M1_price_derivatives", ["price_derivatives"]),
        ("M2_plus_attention", ["price_derivatives", "attention"]),
        ("M3_plus_semantics", ["price_derivatives", "attention", "semantics"]),
        ("M4_plus_gap", ["price_derivatives", "attention", "semantics", "gap"]),
    ]
    rows = []
    for name, groups in stages:
        feats = [x for g in groups for x in FEATURE_GROUPS[g] if x in df.columns]
        r = evaluate_classifier(df, feats, cost_bps=cost_bps, dynamic_cost=dynamic_cost, notional_usd=notional_usd, model_name=model_name, horizon=horizon)
        rows.append({"stage":name, "model":model_name, "features":len(feats), **r.__dict__})
    return pd.DataFrame(rows)


def model_ladder(df: pd.DataFrame, cost_bps: float = 10.0, dynamic_cost: bool = True, notional_usd: float = 1_000.0, horizon: str = "15m") -> pd.DataFrame:
    out=[]
    for model_name in ("logistic","histgb"):
        x=ablation_ladder(df,cost_bps,dynamic_cost,notional_usd,model_name,horizon)
        out.append(x)
    return pd.concat(out,ignore_index=True) if out else pd.DataFrame()


def walk_forward_ablation(df: pd.DataFrame, cost_bps: float = 10.0, folds: int = 4, min_train: int = 100,
                          dynamic_cost: bool = False, notional_usd: float = 1_000.0, model_name: str = "logistic", horizon: str = "15m") -> pd.DataFrame:
    if df.empty or len(df) < min_train + folds * 20:
        return pd.DataFrame()
    d = df.sort_values("ts").reset_index(drop=True)
    test_size = max(20, (len(d) - min_train) // folds)
    outputs = []
    stages = [
        ("M1_price_derivatives", ["price_derivatives"]),
        ("M2_plus_attention", ["price_derivatives", "attention"]),
        ("M3_plus_semantics", ["price_derivatives", "attention", "semantics"]),
        ("M4_plus_gap", ["price_derivatives", "attention", "semantics", "gap"]),
    ]
    for fold in range(folds):
        train_end = min_train + fold * test_size; test_end = min(len(d), train_end + test_size)
        if test_end <= train_end: continue
        train = d.iloc[:train_end]; test = d.iloc[train_end:test_end]
        for stage, groups in stages:
            feats=[x for g in groups for x in FEATURE_GROUPS[g] if x in d.columns]
            tr=train.dropna(subset=[*feats,"target_return"]); te=test.dropna(subset=[*feats,"target_return"])
            if len(tr)<30 or len(te)<10: continue
            r=_evaluate_split(tr,te,feats,model_name=model_name,cost_bps=cost_bps,dynamic_cost=dynamic_cost,notional_usd=notional_usd,horizon_minutes=HORIZONS.get(horizon,15))
            outputs.append({"fold":fold,"stage":stage,"model":model_name,"train_rows":len(tr),"test_rows":len(te),**{k:v for k,v in r.__dict__.items() if k!="rows"}})
    return pd.DataFrame(outputs)


def latency_cost_sensitivity(predicted_returns: pd.Series, delay_decay_per_second: float = 0.002, costs_bps=(5,10,20,30), delays=(0,1,5,15,30,60)) -> pd.DataFrame:
    rows=[]; base=predicted_returns.dropna().astype(float)
    for delay in delays:
        decayed=base*np.exp(-delay_decay_per_second*delay)
        for cost in costs_bps:
            net=decayed-np.sign(decayed).astype(bool)*(cost/10000)
            rows.append({"delay_s":delay,"cost_bps":cost,"mean_net":float(net.mean()) if len(net) else None,"positive_share":float((net>0).mean()) if len(net) else None})
    return pd.DataFrame(rows)


def incremental_semantic_test(df: pd.DataFrame, *, model_name: str = "logistic", dynamic_cost: bool = True,
                              cost_bps: float = 10.0, notional_usd: float = 1_000.0,
                              bootstrap_samples: int = 500, seed: int = 17, horizon: str = "15m") -> dict:
    """Paired held-out test of M2 (market+attention) versus M4 (+semantics+gap).

    Uses the same chronological train/test boundary and the same test rows for both
    models, then bootstraps row indices. This is not a proof of alpha, but it makes
    incremental-value claims much harder to manufacture from incomparable samples.
    """
    m2=[x for g in ["price_derivatives","attention"] for x in FEATURE_GROUPS[g] if x in df.columns]
    m4=[x for g in ["price_derivatives","attention","semantics","gap"] for x in FEATURE_GROUPS[g] if x in df.columns]
    common=df.dropna(subset=[*m4,"target_return"]).sort_values("ts").reset_index(drop=True)
    if len(common)<100:
        return {"ok":False,"reason":"need_at_least_100_common_rows","rows":len(common)}
    split=int(len(common)*.70); te=common.iloc[split:].copy()
    if horizon not in HORIZONS:
        return {"ok":False,"reason":"unknown_horizon","horizon":horizon}
    test_start=pd.to_datetime(te.ts,utc=True).min()
    purge=pd.Timedelta(minutes=HORIZONS[horizon])
    tr=common[pd.to_datetime(common.ts,utc=True) < test_start-purge].copy()
    ytr=(tr.target_return>0).astype(int); yte=(te.target_return>0).astype(int)
    if ytr.nunique()<2 or yte.nunique()<2:
        return {"ok":False,"reason":"single_class_split","rows":len(common)}
    p2=_fit_predict(tr,te,m2,model_name); p4=_fit_predict(tr,te,m4,model_name)
    costs=dynamic_costs(te,notional_usd=notional_usd) if dynamic_cost else np.full(len(te),cost_bps/10_000)
    s2=np.where(p2>=.60,1,np.where(p2<=.40,-1,0)); s4=np.where(p4>=.60,1,np.where(p4<=.40,-1,0))
    n2,m2mask=stateful_trade_vectors(te,s2,costs,horizon_minutes=HORIZONS[horizon],max_positions=3)
    n4,m4mask=stateful_trade_vectors(te,s4,costs,horizon_minutes=HORIZONS[horizon],max_positions=3)
    base={
        "ok":True,"rows":len(common),"train_rows":len(tr),"test_rows":len(te),"model":model_name,"horizon":horizon,"purge_minutes":HORIZONS[horizon],
        "auc_m2":float(roc_auc_score(yte,p2)),"auc_m4":float(roc_auc_score(yte,p4)),
        "auc_delta":float(roc_auc_score(yte,p4)-roc_auc_score(yte,p2)),
        "brier_m2":float(brier_score_loss(yte,p2)),"brier_m4":float(brier_score_loss(yte,p4)),
        "mean_net_all_rows_m2":float(n2.mean()),"mean_net_all_rows_m4":float(n4.mean()),
        "mean_net_delta":float((n4-n2).mean()),"trades_m2":int(m2mask.sum()),"trades_m4":int(m4mask.sum()),
    }
    rng=np.random.default_rng(seed); auc_d=[]; net_d=[]
    yarr=yte.to_numpy(); block=max(4, min(24, int(np.sqrt(len(te)))))
    for _ in range(bootstrap_samples):
        idx_parts=[]
        while sum(len(x) for x in idx_parts) < len(te):
            start=int(rng.integers(0,max(1,len(te)-block+1)))
            idx_parts.append(np.arange(start,min(len(te),start+block)))
        idx=np.concatenate(idx_parts)[:len(te)]
        ys=yarr[idx]
        if len(np.unique(ys))>1:
            auc_d.append(roc_auc_score(ys,p4[idx])-roc_auc_score(ys,p2[idx]))
        net_d.append(float((n4[idx]-n2[idx]).mean()))
    def ci(x):
        return [float(np.quantile(x,.025)),float(np.quantile(x,.975))] if x else [None,None]
    base["bootstrap_block_rows"]=block; base["auc_delta_ci95"]=ci(auc_d); base["mean_net_delta_ci95"]=ci(net_d)
    base["semantic_auc_improvement_supported"] = bool(base["auc_delta_ci95"][0] is not None and base["auc_delta_ci95"][0] > 0)
    base["semantic_net_improvement_supported"] = bool(base["mean_net_delta_ci95"][0] is not None and base["mean_net_delta_ci95"][0] > 0)
    return base

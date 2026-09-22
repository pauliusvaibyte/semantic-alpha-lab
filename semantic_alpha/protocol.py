from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .backtest import dynamic_costs, portfolio_metrics, stateful_trade_vectors, trade_metrics
from .labels import HORIZONS
from .manifest import dataset_fingerprint, holdout_identity
from .models import SemanticResidualModel
from .research import FEATURE_GROUPS, dataset, incremental_semantic_test
from .storage import Store
from .sampling import decision_points
from .validation import purged_walk_forward_ablation


def expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if not mask.any():
            continue
        total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(total)


def _feature_sets(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    baseline = [x for g in ("price_derivatives", "attention") for x in FEATURE_GROUPS[g] if x in df.columns]
    full = [x for g in ("price_derivatives", "attention", "semantics", "gap") for x in FEATURE_GROUPS[g] if x in df.columns]
    return baseline, full


def locked_split(df: pd.DataFrame, horizon: str, holdout_fraction: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    if horizon not in HORIZONS:
        raise ValueError(f"unknown horizon {horizon}")
    if not 0.10 <= holdout_fraction <= 0.40:
        raise ValueError("holdout_fraction must be between 0.10 and 0.40")
    d = df.sort_values("ts").reset_index(drop=True).copy()
    d["ts"] = pd.to_datetime(d.ts, utc=True)
    if len(d) < 120:
        raise ValueError("need at least 120 rows for locked holdout protocol")
    hold_i = max(80, min(len(d) - 30, int(len(d) * (1.0 - holdout_fraction))))
    holdout = d.iloc[hold_i:].copy()
    holdout_start = pd.Timestamp(holdout.ts.min())
    purge = pd.Timedelta(minutes=HORIZONS[horizon])
    discovery = d[d.ts < holdout_start - purge].copy()
    if len(discovery) < 80 or len(holdout) < 30:
        raise ValueError("insufficient discovery/holdout rows after purge")
    return discovery, holdout, holdout_start


def _discovery_score(df: pd.DataFrame, horizon: str, model_name: str, notional_usd: float) -> dict:
    inc = incremental_semantic_test(
        df,
        model_name=model_name,
        dynamic_cost=True,
        notional_usd=notional_usd,
        bootstrap_samples=160,
        horizon=horizon,
    )
    min_train = max(80, len(df) // 3)
    wf = purged_walk_forward_ablation(
        df,
        horizon=horizon,
        folds=3,
        min_train=min_train,
        model_name=model_name,
        notional_usd=notional_usd,
    )
    m4 = wf[wf.stage == "M4_plus_gap"] if not wf.empty else pd.DataFrame()
    m2 = wf[wf.stage == "M2_plus_attention"] if not wf.empty else pd.DataFrame()
    fold_delta: list[float] = []
    if not m4.empty and not m2.empty:
        z = m4.merge(m2, on="fold", suffixes=("_m4", "_m2"))
        fold_delta = (z.mean_net_return_m4 - z.mean_net_return_m2).dropna().astype(float).tolist()
    consistency = float(np.mean(np.asarray(fold_delta) > 0)) if fold_delta else 0.0
    mean_delta = float(np.mean(fold_delta)) if fold_delta else -1.0
    auc_delta = float(inc.get("auc_delta", -1.0) or -1.0) if inc.get("ok") else -1.0
    net_delta = float(inc.get("mean_net_delta", -1.0) or -1.0) if inc.get("ok") else -1.0
    supported = int(bool(inc.get("semantic_net_improvement_supported"))) + int(bool(inc.get("semantic_auc_improvement_supported")))
    # Deliberately conservative: significance/support + fold consistency dominate tiny raw mean differences.
    score = supported * 3.0 + consistency * 2.0 + np.tanh(mean_delta * 2_000.0) + 0.5 * np.tanh(net_delta * 2_000.0) + 0.5 * np.tanh(auc_delta * 10.0)
    return {
        "horizon": horizon,
        "model_name": model_name,
        "rows": int(len(df)),
        "discovery_score": float(score),
        "fold_semantic_net_positive_rate": consistency,
        "fold_mean_net_delta": mean_delta,
        "incremental_auc_delta": auc_delta,
        "incremental_net_delta": net_delta,
        "semantic_auc_supported": bool(inc.get("semantic_auc_improvement_supported", False)),
        "semantic_net_supported": bool(inc.get("semantic_net_improvement_supported", False)),
        "incremental_test": inc,
    }


def discover_configuration(
    store: Store,
    horizons: Iterable[str] = ("5m", "15m", "30m", "1h"),
    model_names: Iterable[str] = ("logistic", "histgb"),
    *,
    holdout_fraction: float = 0.20,
    notional_usd: float = 1_000.0,
    sampling_mode: str = "event",
    feature_version: str = "v4",
) -> tuple[pd.DataFrame, dict]:
    """Search only on pre-holdout data. Holdout labels are never scored here."""
    rows: list[dict] = []
    splits: dict[tuple[str, str], dict] = {}
    for horizon in horizons:
        if horizon not in HORIZONS:
            continue
        raw = dataset(store, horizon=horizon, feature_version=feature_version)
        d = decision_points(raw) if sampling_mode == "event" else raw
        if len(d) < 120:
            continue
        try:
            discovery, holdout, holdout_start = locked_split(d, horizon, holdout_fraction)
        except ValueError:
            continue
        for model_name in model_names:
            score = _discovery_score(discovery, horizon, model_name, notional_usd)
            score.update({
                "holdout_start": holdout_start.isoformat(),
                "holdout_rows_sealed": int(len(holdout)),
                "discovery_fingerprint": dataset_fingerprint(discovery),
                "sampling_mode": sampling_mode,
                "feature_version": feature_version,
                "raw_rows": int(len(raw)),
                "sampled_rows": int(len(d)),
            })
            # Keep detailed test out of CSV-friendly table; retained in split metadata/report.
            detail = score.pop("incremental_test")
            rows.append(score)
            splits[(horizon, model_name)] = {
                "discovery": discovery,
                "holdout": holdout,
                "holdout_start": holdout_start,
                "incremental_test": detail,
            }
    table = pd.DataFrame(rows)
    if table.empty:
        return table, {}
    table = table.sort_values(["discovery_score", "semantic_net_supported", "fold_semantic_net_positive_rate"], ascending=False).reset_index(drop=True)
    winner_row = table.iloc[0]
    key = (str(winner_row.horizon), str(winner_row.model_name))
    return table, {"winner": winner_row.to_dict(), **splits[key]}


def _block_bootstrap_delta(
    y: np.ndarray,
    p_base: np.ndarray,
    p_full: np.ndarray,
    net_base: np.ndarray,
    net_full: np.ndarray,
    *,
    samples: int = 500,
    seed: int = 29,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    block = max(4, min(32, int(np.sqrt(max(1, n)))))
    auc_d: list[float] = []
    net_d: list[float] = []
    for _ in range(samples):
        chunks: list[np.ndarray] = []
        while sum(len(c) for c in chunks) < n:
            start = int(rng.integers(0, max(1, n - block + 1)))
            chunks.append(np.arange(start, min(n, start + block)))
        idx = np.concatenate(chunks)[:n]
        ys = y[idx]
        if len(np.unique(ys)) > 1:
            auc_d.append(float(roc_auc_score(ys, p_full[idx]) - roc_auc_score(ys, p_base[idx])))
        net_d.append(float((net_full[idx] - net_base[idx]).mean()))
    def ci(x: list[float]) -> list[float | None]:
        return [float(np.quantile(x, .025)), float(np.quantile(x, .975))] if x else [None, None]
    return {"block_rows": block, "auc_delta_ci95": ci(auc_d), "mean_net_delta_ci95": ci(net_d)}


def evaluate_locked_holdout(
    discovery: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    horizon: str,
    model_name: str,
    notional_usd: float = 1_000.0,
    bootstrap_samples: int = 500,
) -> tuple[dict, SemanticResidualModel]:
    baseline, full = _feature_sets(discovery)
    common_cols = [*full, "target_return"]
    tr = discovery.dropna(subset=common_cols).copy()
    te = holdout.dropna(subset=common_cols).copy()
    if len(tr) < 100 or len(te) < 30:
        raise ValueError("insufficient complete rows for locked holdout")
    model = SemanticResidualModel.fit(
        tr, baseline, full, model_name=model_name, horizon=horizon,
        calibrate_thresholds=True, notional_usd=notional_usd,
    )
    scores = model.score_frame(te)
    y = (te.target_return > 0).astype(int).to_numpy()
    pb = scores.p_baseline_up.to_numpy()
    pf = scores.p_full_up.to_numpy()
    edge = scores.semantic_edge_logodds.to_numpy()
    pt = float(model.probability_threshold)
    et = float(model.semantic_edge_threshold)
    sig_base = np.where(pb >= pt, 1, np.where(pb <= 1.0 - pt, -1, 0))
    sig_full = np.where((pf >= pt) & (edge >= et), 1, np.where((pf <= 1.0 - pt) & (edge <= -et), -1, 0))
    costs = dynamic_costs(te, notional_usd=notional_usd)
    horizon_min = HORIZONS[horizon]
    net_base, mask_base = stateful_trade_vectors(te, sig_base, costs, horizon_minutes=horizon_min, max_positions=3)
    net_full, mask_full = stateful_trade_vectors(te, sig_full, costs, horizon_minutes=horizon_min, max_positions=3)
    bm = trade_metrics(pd.Series(net_base[mask_base]))
    fm = trade_metrics(pd.Series(net_full[mask_full]))
    te_ts = pd.to_datetime(te.ts, utc=True)

    def _ledger(mask: np.ndarray, net: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({
            "entry_ts": te_ts.to_numpy()[mask],
            "exit_ts": (te_ts + pd.Timedelta(minutes=horizon_min)).to_numpy()[mask],
            "net_return": net[mask], "notional_usd": float(notional_usd),
        })
    port_base = portfolio_metrics(_ledger(mask_base, net_base), horizon_minutes=horizon_min)
    port_full = portfolio_metrics(_ledger(mask_full, net_full), horizon_minutes=horizon_min)
    auc_base = float(roc_auc_score(y, pb)) if len(np.unique(y)) > 1 else None
    auc_full = float(roc_auc_score(y, pf)) if len(np.unique(y)) > 1 else None
    brier_base = float(brier_score_loss(y, pb))
    brier_full = float(brier_score_loss(y, pf))
    ece_base = expected_calibration_error(y, pb)
    ece_full = expected_calibration_error(y, pf)
    boot = _block_bootstrap_delta(y, pb, pf, net_base, net_full, samples=bootstrap_samples)

    # Parameter/cost fragility tests use the already-frozen model. They never feed
    # back into selection or threshold calibration.
    cost_stress = []
    for mult in (1.0, 1.5, 2.0):
        stressed_costs = costs * mult
        n, mask = stateful_trade_vectors(te, sig_full, stressed_costs, horizon_minutes=horizon_min, max_positions=3)
        met = trade_metrics(pd.Series(n[mask]))
        cost_stress.append({"cost_multiplier": mult, "trades": int(mask.sum()), **met})
    threshold_stress = []
    for pdelta in (-0.02, 0.0, 0.02):
        for edelta in (-0.05, 0.0, 0.05):
            ptx = float(np.clip(pt + pdelta, 0.52, 0.80))
            etx = max(0.0, et + edelta)
            sig = np.where((pf >= ptx) & (edge >= etx), 1, np.where((pf <= 1.0 - ptx) & (edge <= -etx), -1, 0))
            n, mask = stateful_trade_vectors(te, sig, costs, horizon_minutes=horizon_min, max_positions=3)
            met = trade_metrics(pd.Series(n[mask]))
            threshold_stress.append({
                "probability_threshold": ptx, "semantic_edge_threshold": etx,
                "trades": int(mask.sum()), **met,
            })
    stress_positive = [float(x.get("net_return", 0.0) or 0.0) > 0 for x in threshold_stress if int(x.get("trades", 0) or 0) >= 10]
    threshold_positive_fraction = float(np.mean(stress_positive)) if stress_positive else 0.0

    by_symbol = []
    tmp = te[["symbol", "target_return"]].copy()
    tmp["full_net"] = net_full
    tmp["full_trade"] = mask_full
    for sym, g in tmp.groupby("symbol"):
        active = g[g.full_trade]
        if active.empty:
            continue
        met = trade_metrics(active.full_net)
        by_symbol.append({"symbol": str(sym), "rows": int(len(g)), "trades": int(len(active)), "mean_net_return": float(active.full_net.mean()), **met})
    positive_symbols = float(np.mean([r["mean_net_return"] > 0 for r in by_symbol])) if by_symbol else 0.0

    ci = boot["mean_net_delta_ci95"]
    checks = {
        "enough_holdout_rows": len(te) >= 50,
        "enough_full_trades": int(mask_full.sum()) >= 20,
        "full_net_positive": float(fm.get("net_return", 0.0) or 0.0) > 0.0,
        "semantic_net_delta_ci_positive": bool(ci[0] is not None and ci[0] > 0),
        "full_auc_above_random": bool(auc_full is not None and auc_full > 0.50),
        "auc_not_materially_worse": bool(auc_base is None or auc_full is None or auc_full >= auc_base - 0.01),
        "cross_symbol_not_concentrated": bool(len(by_symbol) >= 3 and positive_symbols >= 0.50),
        "not_outlier_dominated": bool(fm.get("top5_profit_share") is not None and float(fm["top5_profit_share"]) <= 0.60),
        "survives_2x_costs": bool(cost_stress and float(cost_stress[-1].get("net_return", 0.0) or 0.0) > 0.0),
        "threshold_neighborhood_robust": bool(threshold_positive_fraction >= 0.60),
    }
    status = "PASS_LOCKED_HOLDOUT" if all(checks.values()) else "FAIL_LOCKED_HOLDOUT"
    result = {
        "status": status,
        "horizon": horizon,
        "model_name": model_name,
        "discovery_rows": int(len(tr)),
        "holdout_rows": int(len(te)),
        "holdout_from": str(pd.to_datetime(te.ts, utc=True).min()),
        "holdout_through": str(pd.to_datetime(te.ts, utc=True).max()),
        "discovery_fingerprint": dataset_fingerprint(tr),
        "holdout_fingerprint": dataset_fingerprint(te),
        "thresholds": {"probability": pt, "semantic_edge_logodds": et},
        "classification": {
            "auc_baseline": auc_base,
            "auc_full": auc_full,
            "auc_delta": (auc_full - auc_base) if auc_full is not None and auc_base is not None else None,
            "brier_baseline": brier_base,
            "brier_full": brier_full,
            "ece_baseline": ece_base,
            "ece_full": ece_full,
        },
        "baseline_trading": bm,
        "full_trading": fm,
        # Portfolio-native accounting: equity curve on a fixed time grid.
        # Per-trade stats above are diagnostics; these are the portfolio claims.
        "baseline_portfolio": port_base,
        "full_portfolio": port_full,
        "trades_baseline": int(mask_base.sum()),
        "trades_full": int(mask_full.sum()),
        "mean_net_all_rows_baseline": float(net_base.mean()),
        "mean_net_all_rows_full": float(net_full.mean()),
        "mean_net_delta": float((net_full - net_base).mean()),
        "bootstrap": boot,
        "by_symbol": by_symbol,
        "positive_symbol_fraction": positive_symbols,
        "cost_stress": cost_stress,
        "threshold_stress": threshold_stress,
        "threshold_positive_fraction": threshold_positive_fraction,
        "checks": checks,
        "note": "This is a locked historical holdout test. Passing is evidence for forward paper validation, not evidence of future profitability.",
    }
    return result, model


def run_locked_protocol(
    store: Store,
    output_dir: str | Path,
    *,
    horizons: Iterable[str] = ("5m", "15m", "30m", "1h"),
    model_names: Iterable[str] = ("logistic", "histgb"),
    holdout_fraction: float = 0.20,
    notional_usd: float = 1_000.0,
    bootstrap_samples: int = 500,
    sampling_mode: str = "event",
    allow_reopen: bool = False,
    feature_version: str = "v4",
) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    discovery_table, chosen = discover_configuration(
        store, horizons=horizons, model_names=model_names,
        holdout_fraction=holdout_fraction, notional_usd=notional_usd, sampling_mode=sampling_mode,
        feature_version=feature_version,
    )
    if discovery_table.empty or not chosen:
        result = {"status": "INSUFFICIENT_DATA", "reason": "no viable discovery configuration"}
        (out / "LOCKED_REPORT.json").write_text(json.dumps(result, indent=2))
        return result
    discovery_table.to_csv(out / "DISCOVERY.csv", index=False)
    winner = chosen["winner"]
    horizon = str(winner["horizon"])
    model_name = str(winner["model_name"])
    protocol_config = {
        "horizons": list(horizons), "model_names": list(model_names),
        "holdout_fraction": holdout_fraction, "notional_usd": notional_usd,
        "bootstrap_samples": bootstrap_samples, "sampling_mode": sampling_mode,
        "feature_version": feature_version,
    }
    protocol_hash = hashlib.sha256(json.dumps(protocol_config, sort_keys=True).encode()).hexdigest()
    holdout = chosen["holdout"]
    hts = pd.to_datetime(holdout.ts, utc=True)
    # Holdout identity is bound to the RAW interval/universe/protocol, not to the
    # feature matrix — rebuilding features over the same calendar period collides
    # with the existing lock instead of manufacturing a "new" holdout.
    sealed_fp = holdout_identity(
        symbols=sorted(holdout.symbol.astype(str).unique().tolist()),
        holdout_start=str(hts.min()), holdout_end=str(hts.max()),
        horizon=horizon, label_version="v4", sampling_mode=sampling_mode,
        market_sources=["bybit", "bybit_history"],
    )
    previous_open = store.holdout_opening(sealed_fp)
    if previous_open is not None and not allow_reopen:
        report={
            "status":"HOLDOUT_ALREADY_OPENED", "created_at":datetime.now(timezone.utc).isoformat(),
            "protocol_version":"locked-v2", "protocol_hash":protocol_hash, "config":protocol_config,
            "selected_on_discovery_only":winner, "holdout_identity":sealed_fp,
            "holdout_fingerprint":dataset_fingerprint(holdout),
            "previous_opening":previous_open,
            "governance":{"holdout_used_for_selection":False,"holdout_reopened":False,"live_exchange_execution_available":False},
            "note":"This raw holdout interval was already opened. Add new future data for a fresh holdout; do not tune against the old one.",
        }
        (out / "LOCKED_REPORT.json").write_text(json.dumps(report,indent=2,default=str))
        return report
    # Transactional lock: the opening is persisted BEFORE the holdout is
    # evaluated, so a crash mid-evaluation still leaves the lock burned.
    reopened = previous_open is not None
    store.save_holdout_opening(sealed_fp, horizon, protocol_hash, {
        "opened_at": datetime.now(timezone.utc).isoformat(), "status": "OPENED",
        "evaluation": "pending", "selected_model": model_name, "selected_horizon": horizon,
        "holdout_start": str(hts.min()), "holdout_end": str(hts.max()),
        "holdout_symbols": sorted(holdout.symbol.astype(str).unique().tolist()),
        "output_dir": str(out),
    }, reopened=reopened)
    holdout_result, evidence_model = evaluate_locked_holdout(
        chosen["discovery"], chosen["holdout"], horizon=horizon,
        model_name=model_name, notional_usd=notional_usd,
        bootstrap_samples=bootstrap_samples,
    )
    holdout_result["holdout_reopened"] = bool(reopened)
    holdout_result["holdout_identity"] = sealed_fp
    holdout_result["model_fingerprint"] = evidence_model.training_fingerprint
    evidence_model.save(out / "evidence_model.joblib")
    artifact_sha = hashlib.sha256((out / "evidence_model.joblib").read_bytes()).hexdigest()
    holdout_result["model_artifact_sha256"] = artifact_sha

    # A forward candidate can use all historical information *after* the locked result is recorded.
    # Its paper trades must occur strictly after trained_through and never count as holdout evidence.
    forward_candidate_path = None
    if holdout_result["status"] == "PASS_LOCKED_HOLDOUT":
        all_raw = dataset(store, horizon=horizon, feature_version=feature_version)
        all_df = decision_points(all_raw) if sampling_mode == "event" else all_raw
        baseline, full = _feature_sets(all_df)
        forward_model = SemanticResidualModel.fit(
            all_df, baseline, full, model_name=model_name, horizon=horizon,
            calibrate_thresholds=True, notional_usd=notional_usd,
        )
        forward_candidate_path = out / "forward_candidate.joblib"
        forward_model.save(forward_candidate_path)

    report = {
        "status": holdout_result["status"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": "locked-v2",
        "protocol_hash": protocol_hash,
        "config": protocol_config,
        "selected_on_discovery_only": winner,
        "holdout_identity": sealed_fp,
        "model_fingerprint": evidence_model.training_fingerprint,
        "model_artifact_sha256": artifact_sha,
        "locked_holdout": holdout_result,
        "evidence_model": str(out / "evidence_model.joblib"),
        "forward_candidate": str(forward_candidate_path) if forward_candidate_path else None,
        "governance": {
            "holdout_used_for_selection": False,
            "holdout_reopened": bool(reopened),
            "forward_candidate_trained_after_holdout_evaluation": bool(forward_candidate_path),
            "live_exchange_execution_available": False,
        },
    }
    (out / "LOCKED_REPORT.json").write_text(json.dumps(report, indent=2, default=str))
    # Finalize the already-persisted lock with the evaluation outcome.
    store.save_holdout_opening(sealed_fp,horizon,protocol_hash,{
        "opened_at":report["created_at"],"status":report["status"],"selected_model":model_name,
        "selected_horizon":horizon,"model_fingerprint":evidence_model.training_fingerprint,
        "holdout_start":str(hts.min()),"holdout_end":str(hts.max()),"output_dir":str(out),
    },reopened=reopened)
    return report

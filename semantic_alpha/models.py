from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .backtest import dynamic_costs, stateful_trade_vectors
from .labels import HORIZONS


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def _make_model(name: str):
    if name == "histgb":
        return HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=17,
        )
    if name != "logistic":
        raise ValueError(f"unsupported model: {name}")
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200, class_weight="balanced", C=0.5))


def _model_params(name: str) -> dict[str, Any]:
    """The full hyperparameter surface for a named model family."""
    return dict(_make_model(name).get_params())


def _fingerprint(df: pd.DataFrame) -> str:
    """DEPRECATED weak fingerprint kept for loading legacy artifacts only.
    New models carry the immutable manifest hash (see manifest.py)."""
    cols = [c for c in ["symbol", "ts", "target_return"] if c in df.columns]
    payload = df[cols].astype(str).to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass
class SemanticResidualModel:
    """Paired market-only/full models used to isolate incremental semantic information.

    The semantic edge is the difference in log-odds between the full model and a
    baseline that sees the same market/attention state but no semantic features.
    It is deliberately not called an expected return: it is a model-relative
    residual that must earn its interpretation out of sample.
    """

    baseline_model: Any
    full_model: Any
    baseline_features: list[str]
    full_features: list[str]
    model_name: str
    horizon: str
    trained_rows: int
    trained_through: str | None
    trained_from: str | None
    training_fingerprint: str
    trained_at: str
    probability_threshold: float = 0.64
    semantic_edge_threshold: float = 0.15
    calibration: dict[str, float | int | None] = field(default_factory=dict)
    version: str = "semantic-residual-v3"
    # Immutable identity: the manifest hash IS training_fingerprint. Everything
    # downstream (reports, paper trades, holdout locks, promotion) binds to it.
    manifest: dict[str, Any] = field(default_factory=dict)
    feature_version: str = "v4"

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        baseline_features: list[str],
        full_features: list[str],
        *,
        model_name: str = "histgb",
        horizon: str = "15m",
        calibrate_thresholds: bool = True,
        notional_usd: float = 1_000.0,
        feature_version: str = "v4",
        label_version: str = "v4",
        semantic_models: list[str] | None = None,
        market_sources: list[str] | None = None,
    ) -> "SemanticResidualModel":
        cols = list(dict.fromkeys([*baseline_features, *full_features, "target_return"]))
        d = df.dropna(subset=[c for c in cols if c in df.columns]).sort_values("ts").copy()
        if len(d) < 100:
            raise ValueError("need at least 100 complete rows to train a model bundle")
        y = (d.target_return > 0).astype(int)
        if y.nunique() < 2:
            raise ValueError("training target has one class")

        prob_threshold = 0.64
        edge_threshold = 0.15
        cal_stats: dict[str, float | int | None] = {}
        if calibrate_thresholds and len(d) >= 180:
            cut = int(len(d) * 0.80)
            va = d.iloc[cut:].copy()
            test_start = pd.to_datetime(va.ts, utc=True).min() if "ts" in va else None
            purge_minutes = HORIZONS.get(horizon, 15)
            tr = d[pd.to_datetime(d.ts, utc=True) < test_start - pd.Timedelta(minutes=purge_minutes)].copy() if test_start is not None else d.iloc[:cut].copy()
            ytr = (tr.target_return > 0).astype(int)
            if ytr.nunique() >= 2 and len(va) >= 30:
                bm0, fm0 = _make_model(model_name), _make_model(model_name)
                bm0.fit(tr[baseline_features].fillna(0.0), ytr)
                fm0.fit(tr[full_features].fillna(0.0), ytr)
                pb = bm0.predict_proba(va[baseline_features].fillna(0.0))[:, 1]
                pf = fm0.predict_proba(va[full_features].fillna(0.0))[:, 1]
                edge = _logit(pf) - _logit(pb)
                costs = dynamic_costs(va, notional_usd=notional_usd)
                best = None
                for pt in (0.56, 0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70):
                    for et in (0.05, 0.10, 0.15, 0.20, 0.30):
                        sig = np.where((pf >= pt) & (edge >= et), 1, np.where((pf <= 1-pt) & (edge <= -et), -1, 0))
                        active = sig != 0
                        if active.sum() < max(12, int(len(va) * 0.04)):
                            continue
                        net, accepted = stateful_trade_vectors(va, sig, costs, horizon_minutes=HORIZONS.get(horizon,15), max_positions=3)
                        if accepted.sum() < max(8, int(len(va) * 0.02)):
                            continue
                        mean_net = float(net[accepted].mean())
                        hit = float((sig[accepted] * va.target_return.to_numpy()[accepted] > 0).mean())
                        # Prefer robust mean edge, lightly penalize overly sparse threshold winners.
                        objective = mean_net * np.sqrt(accepted.sum())
                        if best is None or objective > best[0]:
                            best = (objective, pt, et, int(accepted.sum()), mean_net, hit)
                if best is not None:
                    _, prob_threshold, edge_threshold, ntr, mean_net, hit = best
                    cal_stats = {"rows": len(va), "trades": ntr, "mean_net_return": mean_net, "hit_rate": hit}

        bm, fm = _make_model(model_name), _make_model(model_name)
        bm.fit(d[baseline_features].fillna(0.0), y)
        fm.fit(d[full_features].fillna(0.0), y)
        ts = pd.to_datetime(d.ts, utc=True) if "ts" in d else pd.Series(dtype="datetime64[ns, UTC]")
        from .manifest import build_training_manifest
        manifest = build_training_manifest(
            d, model_name=model_name, model_params=_model_params(model_name),
            horizon=horizon, baseline_features=baseline_features, full_features=full_features,
            probability_threshold=float(prob_threshold), semantic_edge_threshold=float(edge_threshold),
            feature_version=feature_version, label_version=label_version,
            semantic_models=semantic_models, market_sources=market_sources,
        )
        return cls(
            baseline_model=bm,
            full_model=fm,
            baseline_features=baseline_features,
            full_features=full_features,
            model_name=model_name,
            horizon=horizon,
            trained_rows=len(d),
            trained_through=str(ts.max()) if len(ts) else None,
            trained_from=str(ts.min()) if len(ts) else None,
            training_fingerprint=manifest["manifest_hash"],
            trained_at=datetime.now(timezone.utc).isoformat(),
            probability_threshold=float(prob_threshold),
            semantic_edge_threshold=float(edge_threshold),
            calibration=cal_stats,
            manifest=manifest,
            feature_version=feature_version,
        )

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.full_features if c not in df.columns]
        if missing:
            raise ValueError(f"missing model features: {missing[:8]}")
        p_base = self.baseline_model.predict_proba(df[self.baseline_features].fillna(0.0))[:, 1]
        p_full = self.full_model.predict_proba(df[self.full_features].fillna(0.0))[:, 1]
        out = pd.DataFrame(index=df.index)
        out["p_baseline_up"] = p_base
        out["p_full_up"] = p_full
        out["semantic_edge_logodds"] = _logit(p_full) - _logit(p_base)
        out["semantic_edge_probability"] = p_full - p_base
        out["direction_confidence"] = np.abs(p_full - 0.5) * 2.0
        return out

    def score_row(self, values: dict[str, float]) -> dict[str, float]:
        d = pd.DataFrame([values])
        r = self.score_frame(d).iloc[0]
        return {k: float(v) for k, v in r.items()}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, path: str | Path) -> "SemanticResidualModel":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError("artifact is not a SemanticResidualModel")
        return obj

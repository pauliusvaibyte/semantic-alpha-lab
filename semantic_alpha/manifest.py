"""Immutable run/model manifests.

The v0.4.0 fingerprint hashed only (symbol, ts, target_return), so two materially
different models trained on the same rows could share an identity and inherit each
other's paper-trading evidence. The manifest hash covers everything that defines
the experiment: dataset values, feature set/version, label version, model family
and hyperparameters, thresholds, semantic engine identity, market sources,
training window, and a hash of the package source itself.

Everything downstream — reports, holdout locks, paper trades, promotion metadata —
must bind to this hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Fingerprint of the exact research matrix (all columns, all values)."""
    if df.empty:
        return hashlib.sha256(b"empty").hexdigest()
    d = df.copy()
    if "ts" in d:
        d["ts"] = pd.to_datetime(d["ts"], utc=True).astype(str)
    d = d.sort_values([c for c in ("ts", "symbol") if c in d.columns]).reset_index(drop=True)
    cols = sorted(d.columns)
    payload = d[cols].to_csv(index=False, float_format="%.12g").encode()
    return hashlib.sha256(payload).hexdigest()


def code_fingerprint() -> str:
    """Hash of the package source — a git-hash substitute for unversioned trees."""
    pkg = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(pkg.rglob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def manifest_hash(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, default=str).encode()).hexdigest()


def build_training_manifest(
    df: pd.DataFrame,
    *,
    model_name: str,
    model_params: dict[str, Any],
    horizon: str,
    baseline_features: list[str],
    full_features: list[str],
    probability_threshold: float,
    semantic_edge_threshold: float,
    feature_version: str,
    label_version: str,
    semantic_models: list[str] | None = None,
    market_sources: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ts = pd.to_datetime(df["ts"], utc=True) if len(df) and "ts" in df else pd.Series(dtype="datetime64[ns, UTC]")
    manifest: dict[str, Any] = {
        "schema": "training-manifest-v1",
        "dataset_fingerprint": dataset_fingerprint(df),
        "rows": int(len(df)),
        "symbols": sorted(df["symbol"].dropna().astype(str).unique().tolist()) if "symbol" in df else [],
        "trained_from": str(ts.min()) if len(ts) else None,
        "trained_through": str(ts.max()) if len(ts) else None,
        "horizon": horizon,
        "feature_version": feature_version,
        "label_version": label_version,
        "baseline_features": sorted(baseline_features),
        "full_features": sorted(full_features),
        "model_name": model_name,
        "model_params": model_params,
        "probability_threshold": float(probability_threshold),
        "semantic_edge_threshold": float(semantic_edge_threshold),
        "semantic_models": sorted(set(semantic_models or [])),
        "market_sources": sorted(set(market_sources or [])),
        "code_fingerprint": code_fingerprint(),
    }
    if extra:
        manifest["extra"] = extra
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def holdout_identity(
    *,
    symbols: list[str],
    holdout_start: str,
    holdout_end: str,
    horizon: str,
    label_version: str,
    sampling_mode: str,
    market_sources: list[str] | None = None,
) -> str:
    """Holdout identity bound to the RAW data window, not the feature build.

    Rebuilding features over the same calendar interval must collide with the
    existing lock — the lock is on the data, not on an implementation detail.
    """
    return manifest_hash({
        "schema": "holdout-identity-v1",
        "symbols": sorted(symbols),
        "holdout_start": holdout_start,
        "holdout_end": holdout_end,
        "horizon": horizon,
        "label_version": label_version,
        "sampling_mode": sampling_mode,
        "market_sources": sorted(set(market_sources or [])),
    })

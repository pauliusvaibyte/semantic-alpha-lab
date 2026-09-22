from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from .models import _make_model
from .research import FEATURE_GROUPS


def feature_importance_report(
    df: pd.DataFrame,
    *,
    model_name: str = "histgb",
    repeats: int = 5,
    seed: int = 17,
) -> pd.DataFrame:
    """Held-out permutation importance for the full model.

    Importance is measured on the chronological final 30% only. It is a diagnostic,
    not causal attribution, but it helps identify whether the model is actually using
    semantic variables or merely carrying them alongside market features.
    """
    features=[x for g in ["price_derivatives","attention","semantics","gap"] for x in FEATURE_GROUPS[g] if x in df.columns]
    d=df.dropna(subset=[*features,"target_return"]).sort_values("ts").reset_index(drop=True)
    if len(d)<120:
        return pd.DataFrame()
    split=int(len(d)*.70); tr,te=d.iloc[:split],d.iloc[split:]
    ytr=(tr.target_return>0).astype(int); yte=(te.target_return>0).astype(int)
    if ytr.nunique()<2 or yte.nunique()<2:
        return pd.DataFrame()
    m=_make_model(model_name); m.fit(tr[features].fillna(0.0),ytr)
    base=float(roc_auc_score(yte,m.predict_proba(te[features].fillna(0.0))[:,1]))
    pi=permutation_importance(m,te[features].fillna(0.0),yte,scoring="roc_auc",n_repeats=repeats,random_state=seed)
    out=pd.DataFrame({
        "feature":features,
        "importance_mean":pi.importances_mean,
        "importance_std":pi.importances_std,
        "heldout_auc":base,
    }).sort_values("importance_mean",ascending=False).reset_index(drop=True)
    group_map={f:g for g,fs in FEATURE_GROUPS.items() for f in fs}
    out["group"]=[group_map.get(x,"other") for x in out.feature]
    return out

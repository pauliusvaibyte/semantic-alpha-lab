from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json

from .audit import leakage_audit
from .drift import drift_report
from .evidence import evidence_gate
from .event_study import event_study_grid
from .models import SemanticResidualModel
from .paper import paper_attribution, paper_summary
from .protocol import run_locked_protocol
from .quality import data_quality
from .research import dataset
from .storage import Store
from .validation import semantic_permutation_placebo, semantic_staleness_test
from .economics import operating_economics


def _decision_markdown(report: dict) -> str:
    locked=report.get("locked_holdout",{}).get("status","UNTESTED")
    evidence=report.get("evidence_gate",{}).get("status","UNTESTED")
    quality="PASS" if report.get("data_quality",{}).get("ok") else "FAIL"
    leakage="PASS" if report.get("leakage",{}).get("ok") else "FAIL"
    placebo=report.get("semantic_placebo",{})
    paper=report.get("paper",{})
    lines=[
        "# Semantic Alpha Campaign Decision",
        "",
        f"- Data quality: **{quality}**",
        f"- Leakage audit: **{leakage}**",
        f"- Locked historical holdout: **{locked}**",
        f"- Forward evidence gate: **{evidence}**",
        f"- Semantic permutation placebo p-value: **{placebo.get('p_value','n/a')}**",
        f"- Closed forward paper trades: **{paper.get('closed_trades',0)}**",
        f"- Paper net return: **{paper.get('net_return','n/a')}**",
        "",
    ]
    if locked != "PASS_LOCKED_HOLDOUT":
        verdict="STOP / RESEARCH: locked holdout did not validate the selected historical configuration."
    elif evidence != "PROMISING_FORWARD_OBSERVED":
        verdict="FORWARD DATA REQUIRED: historical evidence passed far enough to paper-test, but live/forward evidence is not sufficient yet."
    elif int(paper.get("closed_trades",0) or 0) < 100:
        verdict="CONTINUE PAPER VALIDATION: forward evidence exists, but the paper record is still too small."
    else:
        verdict="CHECK CHAMPION GATE: enough evidence exists to run the formal promotion decision."
    lines += [f"## Decision\n\n{verdict}","","Passing any stage is not evidence of future profitability and never enables exchange execution."]
    return "\n".join(lines)


def run_campaign(
    store: Store,
    output_dir: str | Path,
    *,
    horizons: list[str] | None = None,
    model_names: list[str] | None = None,
    holdout_fraction: float = .20,
    notional_usd: float = 1_000.0,
    placebo_iterations: int = 30,
    sampling_mode: str = "event",
    allow_reopen: bool = False,
    feature_version: str = "v4",
) -> dict:
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    quality=data_quality(store)
    leakage=leakage_audit(store)
    # Hard gate: never open the one-shot locked holdout on data that already
    # failed quality or leakage checks. The holdout is burned on first eval.
    if not quality.get("ok") or not leakage.get("ok"):
        report={
            "status":"BLOCKED_INVALID_DATA","created_at":datetime.now(timezone.utc).isoformat(),
            "data_quality":quality,"leakage":leakage,"locked_protocol":{"status":"NOT_OPENED_DATA_GATES_FAILED"},
            "note":"The locked holdout was NOT opened. Fix data quality/leakage issues before consuming the holdout.",
        }
        (out/'CAMPAIGN.json').write_text(json.dumps(report,indent=2,default=str))
        (out/'DECISION.md').write_text(_decision_markdown(report))
        return report
    locked=run_locked_protocol(
        store,out/'locked',horizons=horizons or ["5m","15m","30m","1h"],
        model_names=model_names or ["logistic","histgb"],holdout_fraction=holdout_fraction,
        notional_usd=notional_usd,bootstrap_samples=300,sampling_mode=sampling_mode,allow_reopen=allow_reopen,
        feature_version=feature_version,
    )
    if locked.get("status") in {"INSUFFICIENT_DATA","HOLDOUT_ALREADY_OPENED"}:
        report={"status":"INSUFFICIENT_DATA","created_at":datetime.now(timezone.utc).isoformat(),"data_quality":quality,"leakage":leakage,"locked_protocol":locked}
        (out/'CAMPAIGN.json').write_text(json.dumps(report,indent=2,default=str))
        (out/'DECISION.md').write_text(_decision_markdown(report))
        return report

    selected=locked.get("selected_on_discovery_only",{})
    horizon=str(selected.get("horizon","15m")); model_name=str(selected.get("model_name","histgb"))
    d=dataset(store,horizon=horizon,feature_version=feature_version)
    evidence=evidence_gate(d,horizon=horizon,model_name=model_name,notional_usd=notional_usd,placebo_iterations=placebo_iterations)
    drift=drift_report(d)
    sem_placebo=semantic_permutation_placebo(d,horizon=horizon,model_name="logistic",iterations=placebo_iterations)
    staleness=semantic_staleness_test(d,horizon=horizon,model_name=model_name)
    events=event_study_grid(store,min_n=5)
    if not staleness.empty: staleness.to_csv(out/'semantic_staleness.csv',index=False)
    if not events.empty: events.to_csv(out/'event_study_grid.csv',index=False)

    forward_path=locked.get("forward_candidate")
    paper={"closed_trades":0}
    attribution={"closed_trades":0}
    if forward_path and Path(forward_path).exists():
        m=SemanticResidualModel.load(forward_path)
        paper=paper_summary(store,model_fingerprint=m.training_fingerprint)
        attribution=paper_attribution(store,model_fingerprint=m.training_fingerprint)

    report={
        "status":"CAMPAIGN_READY",
        "created_at":datetime.now(timezone.utc).isoformat(),
        "selected_horizon":horizon,"selected_model":model_name,
        "data_quality":quality,"leakage":leakage,"locked_protocol":locked,
        "locked_holdout":locked.get("locked_holdout",{}),"evidence_gate":evidence,"drift":drift,
        "semantic_placebo":sem_placebo,"semantic_staleness":staleness.to_dict(orient='records') if not staleness.empty else [],
        "event_study":{"tests":int(len(events)),"global_fdr_10pct":int(events.fdr_global_10pct.sum()) if not events.empty and 'fdr_global_10pct' in events else 0},
        "paper":paper,"paper_attribution":attribution,"operating_economics_30d":operating_economics(store,days=30),"forward_candidate":forward_path,
        "live_exchange_execution_available":False,
    }
    (out/'CAMPAIGN.json').write_text(json.dumps(report,indent=2,default=str))
    (out/'DECISION.md').write_text(_decision_markdown(report))
    return report

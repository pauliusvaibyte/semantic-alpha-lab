import pandas as pd

from semantic_alpha.drift import drift_report
from semantic_alpha.evidence import evidence_gate


def test_evidence_gate_requires_real_sample_size():
    d=pd.DataFrame({'symbol':['A']*10,'ts':pd.date_range('2026-01-01',periods=10,freq='min',tz='UTC'),'target_return':[0.0]*10})
    assert evidence_gate(d)['status']=='INSUFFICIENT_DATA'


def test_drift_report_detects_shift():
    d=pd.DataFrame({'ts':pd.date_range('2026-01-01',periods=200,freq='min',tz='UTC'),'symbol':['A']*200,'target_return':[0.0]*200,'x':[0.0]*150+[5.0]*50})
    r=drift_report(d,recent_fraction=.25)
    assert r['ok']
    assert r['features'][0]['feature']=='x'
    assert r['features'][0]['severity']=='HIGH'

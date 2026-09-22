import asyncio
from semantic_alpha.demo import seed_demo
from semantic_alpha.storage import Store
from semantic_alpha.research import dataset, walk_forward_ablation

def test_demo_research_has_walkforward(tmp_path):
    s=Store(str(tmp_path/'r.db'))
    asyncio.run(seed_demo(s, minutes=220))
    d=dataset(s,'15m')
    assert len(d)==220
    wf=walk_forward_ablation(d,cost_bps=8,folds=2,min_train=100)
    assert not wf.empty
    assert set(wf.stage.unique()) >= {'M1_price_derivatives','M3_plus_semantics'}

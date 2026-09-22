from datetime import datetime, timedelta, timezone

import pandas as pd

from semantic_alpha.validation import purged_walk_forward_ablation


def test_purged_walk_forward_has_horizon_gap():
    start=datetime(2026,1,1,tzinfo=timezone.utc)
    rows=[]
    for i in range(260):
        rows.append({
            'symbol':'SOLUSDT','ts':start+timedelta(minutes=i),'target_return':0.002 if i%2 else -0.002,
            'spread_bps':1.0,'funding_rate':0.0001,'open_interest_value':1e9+i,'book_imbalance':(-1)**i*.1,
            'taker_buy_ratio':.55,'basis':0.0,'return_30m':.001*(-1)**i,'oi_change_30m':.001,
            'depth_bid_10bps':100000,'depth_ask_10bps':100000,'trade_notional_1m':10000,'signed_trade_notional_1m':100,
            'liquidation_long_usd_1m':0,'liquidation_short_usd_1m':0,'liquidation_imbalance_1m':0,
            'posts_1m':1,'posts_5m':3,'posts_15m':6,'authors_5m':2,'attention_z_5m':0.1,'attention_accel':1,
            'engagement_velocity_mean_15m':0,'engagement_velocity_max_15m':0,'early_engagement_mean_15m':0,
            'intent_mean_5m':0,'intent_sum_5m':0,'explicit_signal_rate_5m':0,'new_info_rate_5m':0,
            'reactive_rate_5m':0,'promo_rate_5m':0,'shill_rate_5m':0,'evidence_rate_5m':0,'impact_mean_5m':0,
            'intent_agreement_5m':0,'events_15m':0,'event_author_breadth_15m':0,'event_source_domains_15m':0,
            'diffusion_velocity_15m':0,'narrative_novelty_15m':0,'primary_source_rate_5m':0,'research_source_rate_5m':0,
            'rumor_rate_5m':0,'corroborated_rate_5m':0,'semantic_shock':0,'author_alpha_signal_5m':0,
            'author_alpha_mean_5m':0,'author_history_samples_mean_5m':0,'known_author_signal_count_5m':0,
            'contrarian_author_rate_5m':0,'information_price_gap_proxy':0,'crowding_proxy':0,'panic_squeeze_proxy':0,
        })
    d=pd.DataFrame(rows)
    out=purged_walk_forward_ablation(d,horizon='15m',folds=2,min_train=120,model_name='logistic')
    assert not out.empty
    assert set(out.purge_minutes)=={15}
    first=out.iloc[0]
    assert first.train_rows < 120

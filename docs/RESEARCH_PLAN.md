# Research plan v0.5

## Primary question
Does point-in-time semantic social information add **incremental tradable information** beyond a strong price/derivatives/microstructure/attention baseline after realistic costs and latency?

## Required comparison
- **M1:** market/derivatives/microstructure + cross-market state.
- **M2:** M1 + abnormal attention and early engagement.
- **M3:** M2 + Jev semantics, source/author quality, narrative diffusion/novelty.
- **M4:** M3 + information-gap/crowding/squeeze features.

A profitable combined backtest without incremental M2→M4 evidence is not sufficient.

## Anti-leakage rules
- three immutable clocks: `created_at`, `first_seen_at`, `classified_at`;
- features read social data only with `first_seen_at ≤ ts` and semantics only with `classified_at ≤ ts` (`as_of` contract, verified by `leakage_audit`);
- historical candles are timestamped at close availability, never candle-open;
- labels anchor to the realized entry snapshot and refill per-horizon until mature;
- early engagement uses only snapshots available at that time;
- author reputation uses only outcomes that had fully matured before the prediction timestamp;
- rolling market beta uses only preceding market history;
- train labels are purged if their outcome horizon overlaps the test period;
- historical semantic reconstruction is tagged and cannot satisfy the forward-validation gate;
- dense snapshots cannot open overlapping same-symbol positions in the backtest;
- market rows are idempotent per `(source, symbol, ts)` and non-canonical venues never feed labels or baselines;
- live features never persist from stale feed state (per-topic freshness gate).

## Event inference
Posts are clustered into narrative events before inferential event studies. Events anchor to actionable observation time (earliest member availability), semantics join on `(post_id, symbol)`, redirector domains never cluster, and exact destination URLs are decisive. Report:
- number of narrative events and underlying posts;
- abnormal return / signed return;
- MFE/MAE-style path response;
- time-block bootstrap CI (contiguous blocks preserve cross-asset regime dependence);
- nominal p-value and two-sample hi-vs-lo tests;
- Benjamini-Hochberg q-value within a horizon;
- global q-value across all tested horizons.

Do not headline post-level significance.

## Validation battery
1. paired M2 vs M4 held-out increment with block bootstrap;
2. purged expanding walk-forward;
3. leave-one-symbol-out;
4. market-regime slices;
5. circular-shift placebo/null targets;
6. 1m→24h horizon scan;
7. drift report;
8. feature importance;
9. event-level/FDR study;
10. point-in-time provenance coverage;
11. CPCV/PBO + Deflated Sharpe against the append-only `experiment_trials` ledger;
12. adversarial semantic benchmark (prompt injection, fake attribution, spam).

## Economics
All historical returns are net of round-trip execution costs (entry + exit, dynamically estimated from depth/spread). Portfolio metrics are computed on a fixed-time-grid equity curve, not per-trade annualization. Paper exits price on fresh snapshots; stale market data is refused, never reused.

## Forward paper promotion
Research evidence can only nominate a frozen candidate. The candidate then needs forward paper results tied to its exact training fingerprint — the manifest hash covering dataset values, feature columns/version, label version, model class+params, thresholds, semantic models, sources, training window and code. Reports, holdouts and paper trades that do not share the manifest identity cannot be combined. Default champion eligibility requires:
- `PROMISING_FORWARD_OBSERVED` or `PROMISING_FORWARD_VALIDATED` research status;
- clean leakage audit;
- acceptable drift;
- ≥100 closed candidate-specific paper trades;
- positive compounded net return;
- profit factor ≥1.20;
- drawdown no worse than -10%;
- top five winning trades ≤50% of gross gains.

These defaults are deliberately conservative and configurable. Promotion never enables live order execution.

## Kill criteria
Stop or radically rethink the social-semantic thesis when, after enough representative data:
- M4 does not improve M2 on paired/purged tests;
- gains disappear under realistic costs or modest latency;
- performance depends on one asset/regime or a few outliers;
- placebo results are comparable to real-target results;
- apparent historical edge does not replicate on forward-observed semantics;
- the semantic model contributes little feature importance or only encodes price reaction already present in M1/M2.

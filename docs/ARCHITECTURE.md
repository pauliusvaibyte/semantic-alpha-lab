# Architecture v0.5

## Principle
Jev never directly chooses a trade. It turns unstructured social information into independently testable semantic variables. Prediction, execution cost, risk and model promotion are separate quantitative layers.

## Time & availability contract
Every record carries the clock on which it became *usable*, not just when it happened:

- social post: `created_at` (authored), `first_seen_at` (we could observe it);
- semantics: `classified_at` (the classification existed);
- historical candle: timestamped at close availability (`open_ts + interval`);
- label: `base_ts` (realized entry snapshot), `entry_delay_seconds`, `realized_minutes`, `matured`.

A feature at `ts` may only read social records with `first_seen_at ≤ ts` and semantics with `classified_at ≤ ts` — `Store.posts_between`/`semantics_between(as_of=...)` enforce it and `leakage_audit` verifies stored features against it. Labels fill per-horizon and immature rows are refilled; a short-horizon label can never block a long horizon. Exit searches tolerate at most 25% of the horizon (bounded by `max_delay_seconds`), so a "5m" return cannot silently measure 7 minutes — overshoot leaves the horizon `None` until a qualifying row exists, while `realized_minutes` records the true duration. Path statistics (`max_up`/`max_down`/`abs_move`/`range_move`/`fade_ratio`) are only final once their horizon realized; partial-window excursions are recomputed on refill rather than frozen. Market writes are idempotent per `(source, symbol, ts)` and venue timelines never blend.

## Data plane
1. **Social history** — bounded X time windows (twitterapi.io), retried on 429.
2. **Social live** — filter-rule WebSocket; immutable `first_seen_at` captured on arrival. REST recorder keeps per-symbol watermarks so consecutive windows overlap only by a small lag buffer (late-indexed posts) instead of re-billing the same posts.
3. **Engagement** — append-only repeated observations; historical future engagement never overwrites the past.
4. **Semantic extraction** — relevance, communication type, trade intent, catalyst, leading/reactive status, evidence, source role, claim status, promotion/shill, horizon and impact.
5. **Narrative event layer** — deterministic TF-IDF clustering suppresses duplicate narratives and derives breadth/diffusion/novelty factors.
6. **Market live** — Bybit ticker, 50-level book, public trades and all-liquidation feed aggregated into point-in-time snapshots. Per-topic freshness is tracked on both exchange and receive clocks; live persistence refuses stale accumulator state and records reconnect/feed-health telemetry (`feed-health`). Optional second venue (Binance USD-M, public) feeds cross-venue dispersion/lead-lag features; canonical reads pin to `bybit`/`bybit_history`/`bybit_ws` — all Bybit data — so secondary venues can never contaminate labels or baselines, while live-stream rows still count as market history.
7. **Market history** — 1m klines timestamped at candle-close availability plus point-in-time OI/funding where available; unreconstructible historical L2 remains missing.
8. **Primary sources** — `collect-news` normalizes RSS/Atom feeds (exchange blogs, project announcements, regulators, wires) into the same post/semantic/label pipeline; unmatched items land under `MACRO`.
9. **Feature layer v3** — market/derivatives, attention, semantics, source/author, narrative, cross-asset and information-gap factors, plus residualized social shocks (`E[social|market]` ridge on strictly-trailing history) and deterministic author-verification/official-account rates.
10. **Label layer v4** — 1m→24h returns, rolling-beta abnormal returns, future path excursions, plus `abs_move` (magnitude), `range_move` (volatility) and `fade_ratio` (crowding/reversal) heads.

## Research plane
```text
M1 market + derivatives + microstructure
             ↓
M2 + attention / engagement
             ↓
M3 + Jev semantics / authors / narratives
             ↓
M4 + information-gap / crowding / squeeze
```

Evaluation uses purged chronological folds, non-overlapping stateful portfolio simulation, round-trip dynamic execution costs (entry + exit), portfolio-native equity/Sharpe on a fixed time grid, time-block bootstrap, two-sample hi-vs-lo tests, placebo targets, asset holdouts, regime/horizon scans and drift checks.

Selection-overfit controls sit on top of the locked holdout: an append-only `experiment_trials` ledger fingerprints every research run, `overfit-report` computes combinatorial purged CV / Probability of Backtest Overfitting across the M1→M4 ladders, and Deflated Sharpe deflates the best OOS Sharpe by the recorded trial count.

Because the feed is adversarial, `adversarial-benchmark` measures whether the semantic engine survives prompt injection, fake source attribution, negation and spam before its `claim_status`/`source_role` heads are trusted; `OFFICIAL_X_ACCOUNTS` provides a deterministic source-verification channel the classifier cannot talk itself into.

## Semantic residual
Two paired models see the same timestamp:
- baseline sees market + attention;
- full model additionally sees semantic/narrative/gap factors.

`semantic_edge_logodds = logit(P_full(up)) - logit(P_baseline(up))`

This is a model-relative residual, not an assumed expected return. It must earn value out of sample.

## Inferential unit
Headline semantic event studies use **narrative events**, not posts. Copies/reposts in one cluster count as one event. Clustering ignores redirector domains (`t.co` etc.); an exact normalized destination URL is a decisive match. Events anchor to actionable observation time (earliest member availability), persist with stable IDs via `event_memberships`, and semantic interpretations join on `(post_id, symbol)` so multi-asset posts keep per-asset readings. Tests report bootstrap confidence intervals and FDR-adjusted q-values across semantic hypotheses/horizons.

## Governance plane
Every model carries an immutable **manifest** (`manifest.py`): dataset value hash, feature columns/version, label version, model class + hyperparameters, thresholds, semantic engine and requested/effective Jev model, market sources, training window and code hash. `training_fingerprint` is the manifest hash — reports, locked holdouts and paper trades all bind to it, so evidence cannot be recombined across materially different configurations. Holdout identity derives from the raw interval/universe/protocol (not the feature matrix), locks are written before evaluation, and campaigns hard-stop on data-quality/leakage failure rather than burning the holdout. A model can enter the champion registry only after the forward-evidence and paper-trading gates pass.

## Safety boundary
There is deliberately no exchange order endpoint, private exchange key handling, or live-capital execution in v0.5.

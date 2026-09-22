# Semantic Alpha Lab v0.5.0

Research-first crypto social-information alpha engine rebuilt from the idea behind `brainstormity/Jev-X-Sentiment-Analysis`.

**Jev never directly chooses BUY/SELL/HOLD.** It converts unstructured social posts into point-in-time semantic variables. A separate quantitative layer asks whether those variables add incremental predictive value over a strong market/derivatives/attention baseline after costs, leakage controls and realistic portfolio constraints.

## Thesis

```text
new semantic information
× abnormal attention acceleration
× independent narrative diffusion
× source / claim quality
× historical source edge
× derivatives positioning
× order flow / liquidity
× price reaction already realized
                    ↓
        incremental semantic residual
                    ↓
          tradable after costs?
```

A profitable-looking combined backtest is not enough. v0.5 makes time, identity and economics trustworthy so results are worth rejecting on merit, and keeps the hypothesis easy to reject without repeatedly reopening the same holdout.

## Architecture

```text
X history / live filter stream (twitterapi.io)
                     │
                     ▼
      immutable post + first_seen timestamp
                     │
          append-only engagement curve
                     │
                     ▼
             Jev semantic vector
 trade intent / catalyst / leading-vs-reactive / source role /
 claim status / evidence / promotion / impact / horizon
                     │
                     ▼
              narrative events
    duplicate suppression / breadth / diffusion / novelty
                     │
                     ├───────────────────────┐
                     ▼                       ▼
              social factors              Bybit
                                      ticker / funding / OI
                                    L2 / trades / liquidations
                     │                       │
                     └───────────┬───────────┘
                                 ▼
                         v3 feature snapshots
                                 │
               ┌─────────────────┴─────────────────┐
               ▼                                   ▼
    baseline model: M2                    full model: M4
 market+derivatives+attention       +semantics+narrative+gap
               │                                   │
               └─────────────────┬─────────────────┘
                                 ▼
              semantic edge = logit(Pfull)-logit(Pbase)
                                 │
                                 ▼
                     purged / stateful research
                                 │
                                 ▼
                 frozen candidate model fingerprint
                                 │
                                 ▼
                        forward paper trading
                                 │
                                 ▼
                    candidate → champion gate
```

There is **no exchange order execution path** in v0.5.

## Multi-asset social semantics

One external X post is stored once, but may map to multiple tradable assets. Asset-specific semantic interpretations are stored independently, so a relative-value statement such as `long ETH vs BTC` can be bullish for ETH and bearish for BTC without one interpretation overwriting the other. Legacy databases migrate automatically to the new mapping tables.

## What v0.5 adds — correctness release

No new alpha features. This release makes time, identity and economics trustworthy:

- **Point-in-time availability, enforced.** Features only see posts with `first_seen_at ≤ ts` and classifications with `classified_at ≤ ts`; `leakage_audit` verifies stored features against those availability clocks rather than trusting the pipeline.
- **No candle lookahead.** Historical bars are timestamped at candle-close availability (`open_ts + interval`); a feature can no longer know a close before it completes.
- **Labels mature instead of freezing.** Per-horizon merge: a 1m/5m label no longer blocks 1h/4h/24h from filling; labels anchor to the realized entry snapshot (`base_ts`, `entry_delay_seconds`, `realized_minutes`) and maintenance refills immature rows.
- **Idempotent, source-separated market data.** Writes are unique per `(source, symbol, ts)`; compaction and quality never blend venue timelines.
- **Dead feeds can't fake freshness.** WebSocket accumulators track per-topic exchange vs receive timestamps; live persistence refuses stale state and records reconnect/feed-health telemetry (`feed-health`).
- **Real economics.** Backtests charge round-trip entry+exit costs; `portfolio_metrics` reports portfolio-native equity/Sharpe; paper exits price on fresh snapshots and risk caps dominate floors.
- **One identity for everything.** `manifest.py` fingerprints dataset values, feature columns/version, label version, model class+params, thresholds, semantic models, sources, training window and code. Models, reports, holdouts and paper trades all bind to that manifest hash — report A cannot endorse candidate B.
- **Jev model honesty.** The configured model is actually sent to the SDK; the returned `effective_model` is stored as provenance (`jev-latest` currently resolves to `jev-1.13.0`).
- **Holdout can't be burned or bypassed.** Identity derives from the raw interval/universe/protocol, locks write before evaluation, and campaigns hard-stop on quality/leakage failures.
- **Correct event semantics.** Joins on `(post_id, symbol)`; `t.co` redirector domains no longer cluster unrelated posts; an exact normalized destination URL is a decisive match; events anchor to actionable observation time and persist with stable IDs.
- **A regression test per finding.** `tests/test_correctness_release.py` (23 tests); total suite 91. CI runs lint + compile + tests on Python 3.11/3.12; `requirements-lock.txt` pins the tested environment; `dist/` ships the wheel + `SHA256SUMS.txt`.

## What v0.4 adds

- **One-shot locked temporal holdout.** Horizon/model discovery happens only on earlier data. Opening a holdout fingerprints it in SQLite; a second opening is refused unless an explicit override marks the evidence compromised.
- **Event-triggered decision sampling.** Dense 1s/15s recorder ticks are reduced to materially changed social states plus sparse quiet controls before model selection.
- **Holdout fragility tests.** Selected strategies must survive 2× execution costs and a neighborhood of nearby probability/semantic-edge thresholds.
- **Semantic timing controls.** Permutation placebo and staleness tests ask whether correctly aligned social semantics matter, rather than merely acting as slow regime proxies.
- **Parallel semantic-engine universes.** Jev, heuristic or future classifiers can be materialized into separate feature versions and compared on exactly the same timestamps without overwriting each other.
- **Cross-sectional factor diagnostics.** Raw factors get Spearman IC, top-minus-bottom spread and quantile monotonicity reports before ML gets credit.
- **Risk-aware forward paper trading.** Sizing is volatility/risk-budget capped, liquidity capped, feature/market freshness gated and one feature may only be traded once.
- **Paper attribution.** Closed simulated P&L is broken down by asset, side, market regime and semantic-edge strength.
- **Provider-cost ledger.** X post reads and Jev semantic calls carry cost provenance; operating economics subtract data/model cost from simulated paper P&L.
- **Storage health + compaction.** The CLI projects recorder growth and can dry-run/downsample old 1-second market snapshots to minute buckets.
- **Campaign orchestrator.** One command runs data/leakage checks, locked discovery/holdout, semantic placebo/staleness, forward evidence and paper diagnostics.
- **Stricter champion gate.** Promotion requires an uncompromised locked holdout, forward/live evidence, clean leakage/drift checks and a diversified paper record.

Shipped in v0.5 (post-0.4.0 hardening):

- **Labels v4.** `abs_move` (magnitude), `range_move` (volatility) and `fade_ratio` (crowding/reversal) heads alongside direction, so volatility/crowding models can be trained even if direction has no edge.
- **Residualized social shocks.** `E[social|market]` ridge fits on point-in-time feature history emit `attention_residual_5m`, `posts_residual_5m`, `intent_residual_5m`, `semantic_shock_residual` — the component of social activity not explained by the price move that caused it (price→posts reverse-causality control).
- **Deterministic source features.** `verified_author_rate_*` / `official_account_rate_*` from account metadata rather than asking the classifier who the author is.
- **Multi-venue market state.** `--venues bybit,binance` records Binance USD-M snapshots (public, no key); `venue_price_dispersion_bps`, `venue_funding_dispersion`, `venue_basis_dispersion`, `bybit_vs_venues_return_gap_5m`. Canonical reads pin to Bybit (`bybit` + `bybit_history`); other venues never contaminate labels/baselines unless explicitly requested.
- **Adversarial semantic benchmark.** `adversarial-benchmark` probes the classifier with prompt injection, fake source attribution, cashtag stuffing, negation, sarcasm, copy spam, unicode tricks and long-context dilution; reports a per-class manipulability score.
- **Overfit controls.** `overfit-report` runs combinatorial purged CV across the M1→M4 feature ladders and reports PBO plus a Deflated Sharpe p-value against the immutable `experiment_trials` ledger (`trials` command); every research-suite/locked-research/overfit run is fingerprinted and append-only.
- **Primary-source ingestion.** `collect-news` polls RSS/Atom feeds (`NEWS_FEEDS=name=url,...`) into the same post/semantic/label pipeline as X; unmatched items land under `MACRO`.
- **Latency + capacity instrumentation.** `latency-report` splits provider indexing lag from classification lag; `capacity-report` builds the execution-cost curve across position sizes from stored book depth. `semantic-reliability` bins each semantic probability head against realized |abnormal return|.
- **Configurable query languages.** `X_QUERY_LANGS` (default `en`) controls the `lang:` filter on search/stream rules.
- **API resilience.** X search retries 429s with exponential backoff; per-symbol social watermarks stop the forward recorder from re-billing the same posts (~5x overlap → ~2x or less).

The strong v0.3 safeguards remain: beta-adjusted abnormal returns, purged walk-forward validation, stateful non-overlapping backtests, event-level studies, bootstrap CIs, global FDR correction, point-in-time author reputation, leave-one-symbol-out/regime tests, experiment fingerprints and no exchange execution.

## Install

```bash
pip install -e .

# Jev support
pip install -e '.[jev]'

# real-time WebSocket feeds
pip install -e '.[stream]'

# tests/build
pip install -e '.[dev]'

cp .env.example .env
semantic-alpha init-db
semantic-alpha doctor
```

## Fast local validation

```bash
pytest -q
semantic-alpha demo --db ./data/demo.db
semantic-alpha leakage-audit --db ./data/demo.db
semantic-alpha data-quality --db ./data/demo.db
semantic-alpha event-study-grid --db ./data/demo.db --min-n 2
```

The synthetic demo intentionally is **not evidence**. The evidence gate should refuse it because it lacks sufficient independent assets/forward observations.

## Start accumulating the real forward dataset

### Simple REST recorder

```bash
semantic-alpha collect-forward \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,SUIUSDT,AVAXUSDT,LINKUSDT \
  --interval-seconds 60 \
  --engine jev
```

Add `--venues bybit,binance` to `collect-once`/`collect-forward` to record Binance
USD-M snapshots alongside Bybit for cross-venue dispersion features.

Primary-source feeds (exchange blogs, wires, regulators) can be polled into the
same pipeline:

```bash
# .env: NEWS_FEEDS=coindesk=https://www.coindesk.com/arc/outboundfeeds/rss/,binance=https://example.com/rss
semantic-alpha collect-news --symbols BTCUSDT,ETHUSDT,SOLUSDT
```

### Lowest-latency live mode

Create X filter rules once (via twitterapi.io):

```bash
semantic-alpha x-add-rules \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,SUIUSDT,AVAXUSDT,LINKUSDT
```

Then:

```bash
semantic-alpha live \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,SUIUSDT,AVAXUSDT,LINKUSDT \
  --engine jev \
  --market-snapshot-seconds 1 \
  --feature-seconds 15 \
  --semantic-workers 4
```

The live loop is self-healing and self-reporting: stream connect/disconnect, stall, delivery-gap and queue-saturation events are written to `feed_health`; a silent or zombie X socket is force-recycled (`--social-stall-seconds`, `--delivery-recycle-seconds`); semantic calls carry `--classify-timeout` so a hung provider cannot kill a worker; and a free Binance public-REST poll (`--binance-poll-seconds`, 0 disables) keeps venue-dispersion features alive alongside the Bybit stream.

In another process, let outcomes and author reputation mature:

```bash
semantic-alpha maintenance-run \
  --interval-minutes 15
```

Each maintenance cycle also re-observes a small trailing slice of posts via X search so engagement-velocity features get a second observation (`--no-engagement-refresh` disables; it bills per returned post).

## Historical bootstrap

```bash
semantic-alpha historical-market BTCUSDT \
  --start 2026-07-01T00:00:00Z --end 2026-08-01T00:00:00Z
semantic-alpha historical-market SOLUSDT \
  --start 2026-07-01T00:00:00Z --end 2026-08-01T00:00:00Z

semantic-alpha historical-social SOLUSDT \
  --start 2026-07-01T00:00:00Z --end 2026-08-01T00:00:00Z \
  --window-minutes 15

semantic-alpha semanticize-range SOLUSDT \
  --start 2026-07-01T00:00:00Z --end 2026-08-01T00:00:00Z \
  --engine jev

semantic-alpha rebuild-historical-features \
  BTCUSDT,SOLUSDT \
  --start 2026-07-01T00:00:00Z --end 2026-08-01T00:00:00Z \
  --interval-minutes 5

semantic-alpha backfill-labels
```

Historical semantic reconstruction is useful for hypothesis screening, but provenance features prevent it from satisfying the forward-validation gate.

## Main research commands

```bash
semantic-alpha leakage-audit
semantic-alpha data-quality
semantic-alpha event-study-grid --output ./runs/event-study.csv
semantic-alpha semantic-increment-test --horizon 15m
semantic-alpha walk-forward --horizon 15m
semantic-alpha horizon-scan
semantic-alpha robustness-report
semantic-alpha drift-report
semantic-alpha feature-importance
semantic-alpha evidence-gate --horizon 15m
```

Or run the full battery:

```bash
semantic-alpha research-suite \
  --output-dir ./runs/research-suite \
  --horizon 15m \
  --model-name histgb \
  --notional-usd 1000
```

The suite writes, among other files:

```text
EXPERIMENT.json             exact dataset/config fingerprint
dataset.csv
purged_walk_forward.csv
leave_one_symbol_out.csv
regimes.csv
horizon_scan.csv
feature_importance.csv
event_study_grid.csv
candidate_model.joblib
REPORT.json
```

## Research ladder

```text
M1  price + derivatives + microstructure + cross-market context
M2  M1 + abnormal attention + early engagement
M3  M2 + Jev semantics + authors + narrative diffusion/novelty
M4  M3 + information-gap / crowding / squeeze features
```

The central test is M2 versus M4 on the **same held-out observations**. The paired report includes AUC and net-return deltas with block-bootstrap confidence intervals.

## Event studies

The default unit is a narrative event, not a post:

```bash
semantic-alpha event-study --horizon 15m --unit event
semantic-alpha event-study-grid --output ./runs/event-grid.csv
```

Outputs include underlying post count, narrative-event count, future abnormal return, path excursions, bootstrap CI, nominal p-value and FDR-adjusted q-values. Global q-values correct across all scanned horizons.

## Strongest research path

Use the campaign once enough synchronized data exists:

```bash
semantic-alpha campaign \
  --output-dir ./runs/campaign-001 \
  --horizons 5m,15m,30m,1h \
  --models logistic,histgb \
  --sampling-mode event
```

The campaign searches only discovery data, opens one sealed holdout once, and writes `CAMPAIGN.json` + `DECISION.md`. If you rerun against the same holdout fingerprint, it refuses by default. Add future data instead of tuning against the old holdout.

Useful diagnostics:

```bash
semantic-alpha factor-ic-report --horizon 15m
semantic-alpha semantic-placebo --horizon 15m
semantic-alpha semantic-staleness --horizon 15m
semantic-alpha semantic-reliability --horizon 15m
semantic-alpha adversarial-benchmark --engine jev --symbol SOL
semantic-alpha overfit-report --horizon 15m
semantic-alpha trials
semantic-alpha latency-report
semantic-alpha feed-health
semantic-alpha stream-probe --seconds 90      # real created_at→delivery lag, needs active x-add-rules rule
semantic-alpha capacity-report --days 30
semantic-alpha usage-report --days 30
semantic-alpha cost-report --days 30
semantic-alpha storage-health
semantic-alpha compact-market --older-than-days 7          # dry-run
semantic-alpha compact-market --older-than-days 7 --execute
```

To compare semantic engines, materialize separate feature versions from the same raw history and then:

```bash
semantic-alpha semantic-engine-compare \
  --versions v4__jev,v4__heuristic \
  --horizon 15m
```

## Candidate model + forward paper validation

`research-suite` freezes a candidate model. Paper trading uses only features newer than the model's training cutoff and never sends exchange orders:

```bash
semantic-alpha paper-run \
  --model-path ./runs/research-suite/candidate_model.joblib \
  --interval-seconds 15 \
  --notional-usd 1000
```

Inspect candidate-specific results:

```bash
semantic-alpha paper-status \
  --model-path ./runs/research-suite/candidate_model.joblib
```

## Candidate → champion gate

A good historical report cannot promote itself. By default, promotion requires:

- evidence status `PROMISING_FORWARD_VALIDATED`;
- clean leakage audit;
- acceptable feature drift;
- at least 100 closed **candidate-fingerprint-specific** paper trades;
- positive compounded paper return;
- profit factor ≥1.20;
- max drawdown no worse than -10%;
- top five winners ≤50% of gross gains.

```bash
semantic-alpha promote-model \
  ./runs/research-suite/candidate_model.joblib \
  ./runs/research-suite/REPORT.json \
  --registry-dir ./models/registry

semantic-alpha champion-status --registry-dir ./models/registry
```

Promotion only copies a model artifact into a local champion registry. It does not enable live-money trading.

## Point-in-time rules

- Three separate immutable clocks: `created_at` (post authored), `first_seen_at` (we could observe it), `classified_at` (semantics existed). A feature at `ts` may only use records available by `ts` — enforced via `as_of` on every social read and verified by `leakage_audit`.
- Historical candles carry close-availability timestamps: the bar timestamped `t` is only usable from `t` onward, so a feature never knows an incomplete bar's close.
- Labels anchor to the first canonical market snapshot at/after the signal (`base_ts`) and record realized durations; horizons fill independently and immature rows are refilled, never silently shortened by sparse data.
- Forward collection resumes from a per-symbol watermark; windows overlap only by `--lag-overlap-seconds` (default 60) for late-indexed posts, so the provider does not repeatedly re-bill the same posts. `0` gives strict contiguous windows.
- Market writes are idempotent per `(source, symbol, ts)`; secondary venues are stored but never read as canonical truth.
- Live snapshots persist only when every subscribed feed topic is fresh — a disconnected stream cannot mint fresh-looking rows from stale state.
- Engagement is append-only; later likes/reposts do not overwrite historical state.
- Semantics classified long after an old post are marked by provenance and cannot be treated as live observation.
- Historical missing L2/order flow is left missing instead of fabricated.
- Author reputation uses only signals whose full target horizon had matured before that timestamp.
- Market beta is estimated from preceding history only.
- Train/test splits are purged by the target horizon.
- Dense feature snapshots cannot create overlapping same-symbol backtest trades.
- Models, reports, holdouts and paper trades bind to one immutable manifest fingerprint — evidence cannot be recombined across configurations.
- The locked holdout identifies the raw interval/universe/protocol, locks before evaluation, and is one-shot.
- Keep a final untouched forward period.

## Kill the thesis when

After a representative amount of forward data, stop rather than add complexity if:

- M4 does not beat M2 on paired/purged held-out tests;
- gains disappear under plausible fees/spread/depth/latency;
- performance comes from one coin, one regime or a handful of trades;
- placebo targets perform similarly;
- historical semantic edge does not reproduce with genuinely live semantics;
- Jev semantic features are largely ignored once market/attention features are present.

See `docs/ARCHITECTURE.md`, `docs/RESEARCH_PLAN.md` and `CHANGELOG.md`.

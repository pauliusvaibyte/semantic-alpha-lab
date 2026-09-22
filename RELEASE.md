# Semantic Alpha Lab v0.5.0 — correctness release

Research-grade semantic/social alpha discovery system. This release intentionally adds **no new alpha features**: it makes time, identity, and economics trustworthy so that historical results mean something. Still a research and forward paper-validation system, not a live-capital trading bot.

## Release gate

- 91 regression tests pass (includes `tests/test_correctness_release.py`, one test per audit finding).
- Python package compiles cleanly; `ruff check` (correctness-class rules) is clean.
- Wheel builds as `semantic_alpha_lab-0.5.0-py3-none-any.whl`; `dist/SHA256SUMS.txt` records the artifact hash.
- `requirements-lock.txt` pins the exact tested environment.
- Secret scan is clean; no API credentials are bundled.
- Live smoke verified: real X ingestion via `twitterapi.io` (vendor identity), Jev classification, narrative events, and full feature vectors with real Bybit + Binance data.
- There is no exchange order-placement path.

## What was broken and is now fixed

### Time
- Historical social lookahead: posts first seen/classified hours later could appear in earlier reconstructed features and still pass the leakage audit. All social reads now take `as_of` availability cutoffs (`first_seen_at`, `classified_at`), and the audit verifies against them.
- Candle lookahead: historical bars were timestamped at candle *start* while carrying the candle *close* price. Availability timestamps now land at candle completion; open time is kept in `raw`.
- Partial labels froze: an existing 1m/5m label row prevented later 1h/4h/24h horizons from ever filling. Labels now merge per-horizon with a `matured` flag, anchor to the realized entry snapshot (`base_ts`, `entry_delay_seconds`, `realized_minutes`), and maintenance refills immature rows.

### Data
- Market rows were not idempotent and sources were blended. Storage is now `(source, symbol, ts)` unique; compaction and quality are source-aware.
- Stale WebSocket state looked fresh: a dead feed kept emitting rows with fresh local timestamps. Per-topic freshness (exchange vs receive timestamps), reconnect telemetry, and a `feed-health` report now gate live persistence.

### Economics
- Backtests charged one-way cost on entry→exit returns. Round-trip (entry + exit) costs are charged; `portfolio_metrics` produces portfolio-native equity/Sharpe on a fixed grid; paper exits price on fresh snapshots and risk caps dominate floors.

### Provenance
- Model fingerprints hashed only symbol/timestamp/target — materially different models collided. `manifest.py` now fingerprints dataset values, feature columns/version, label version, model class+params, thresholds, semantic models, sources, training window, and code. Models, reports, holdouts, and paper trades all bind to that manifest hash.
- The configured Jev model was never sent to the SDK. It is now sent, and the returned `effective_model` is stored as provenance.
- Holdout identity could be bypassed by changing features, and campaigns could burn the holdout on bad data. Identity derives from raw window/universe/protocol; locks precede evaluation; quality/leakage failures hard-stop.

### Multi-asset & statistics
- `(post_id, symbol)` semantic joins fix cross-asset interpretation bleed; `post_assets` joins drive quality metrics.
- `t.co` redirector domains no longer over-cluster narratives; an exact normalized destination-URL match is decisive; events anchor to actionable observation time and persist with stable IDs.
- Event studies infer at event level with BH-FDR and time-block bootstrap; the append-only `experiment_trials` ledger makes failed experiments countable.

## Evidence boundary

A positive historical result is not sufficient. Champion eligibility requires an uncompromised locked holdout, forward-observed evidence, acceptable drift, and a sufficiently large/diversified paper-trading record. Even champion eligibility is not permission to deploy capital.

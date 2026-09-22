"""One-off bounded parallel drain of the semantic backlog.

The maintenance backfill is sequential (~21s/classify); the TypeSafe credit
outage left ~5.7k pending pairs that would take ~33h to drain that way.
This drains with bounded concurrency, records failures through the same
classification_failures bookkeeping, and stops after 5 consecutive provider
errors (out-of-credits / down) so it can't burn calls against a dead provider.
"""
import asyncio
import sys
from datetime import datetime, timezone

from semantic_alpha.storage import Store
from semantic_alpha.config import settings
from semantic_alpha.semantics.jev import JevSemanticEngine, is_provider_error

CONCURRENCY = 6
BATCH = 60
STOP_AFTER_CONSEC_PROVIDER_ERRORS = 5


async def main() -> int:
    store = Store(settings.db_path if hasattr(settings, "db_path") else None)
    engine = JevSemanticEngine(settings.typesafe_api_key, settings.jev_model)
    total_done = total_err = consec_prov = 0
    while True:
        # re-resolve each batch: prefer the provider's effective model, then the
        # most recently *stored* model, then the alias — coverage checks under
        # the bare alias pre-resolution reclassify already-done posts (burned
        # 1,740 calls on the first run of this script, 1,500 on a maintenance
        # restart before this fallback existed).
        model = (getattr(engine, "effective_model", None)
                 or store.latest_semantic_model()
                 or getattr(engine, "query_model", getattr(engine, "model", "jev")))
        pending = store.unsemanticized_posts(None, model, limit=BATCH)
        if not pending:
            break
        gate = asyncio.Semaphore(CONCURRENCY)

        async def one(p, model=model, gate=gate):
            nonlocal consec_prov
            async with gate:
                try:
                    store.save_semantics(await asyncio.wait_for(engine.classify(p), timeout=90))
                    return True
                except Exception as e:
                    prov = isinstance(e, asyncio.TimeoutError) or is_provider_error(str(e))
                    store.record_classification_failure(p.post_id, p.symbol, model, str(e)[:300], counted=not prov)
                    if prov:
                        consec_prov += 1
                    return False

        results = await asyncio.gather(*(one(p) for p in pending))
        total_done += sum(results)
        total_err += len(results) - sum(results)
        print(f"{datetime.now(timezone.utc).isoformat()} batch done={sum(results)} err={len(results)-sum(results)} total_done={total_done} total_err={total_err}", flush=True)
        if consec_prov >= STOP_AFTER_CONSEC_PROVIDER_ERRORS:
            print("provider down — stopping", flush=True)
            return 2
    print(f"drain complete: classified={total_done} errors={total_err}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .schema import EventCluster, SocialPost

STOP = {"the","a","an","to","of","and","is","in","for","on","this","that","with","it","at","as"}

# Redirector/shortener domains carry zero narrative information: on X every link
# is t.co-wrapped, so domain overlap alone must never cluster posts. Only an
# exact normalized URL match is decisive evidence of a shared story.
IGNORED_LINK_DOMAINS = {
    "t.co", "x.com", "twitter.com", "mobile.twitter.com", "pic.twitter.com",
    "bit.ly", "buff.ly", "ow.ly", "tinyurl.com", "goo.gl", "is.gd", "rb.gy", "shorturl.at",
}


def normalized_text(text: str) -> str:
    words = [x for x in re.findall(r"[a-z0-9$]{2,}", text.lower()) if x not in STOP]
    return " ".join(words)


def _norm_url(u: str) -> str:
    """Canonical URL key: scheme/www/casing/trailing-slash/tracking removed."""
    try:
        p = urlparse(u.strip())
    except ValueError:
        return u.strip().lower()
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = re.sub(r"/+$", "", p.path or "")
    query = "&".join(
        sorted(q for q in (p.query or "").split("&") if q and not q.lower().startswith(("utm_", "ref=", "ref_src=")))
    )
    return f"{host}{path}" + (f"?{query}" if query else "")


def effective_urls(p: SocialPost) -> list[str]:
    """Best-known destination URLs for a post.

    Vendor payloads may carry expanded URLs in entities; prefer those over the
    raw t.co wrappers in the text so unrelated posts do not share a redirector.
    """
    out: list[str] = []
    raw = p.raw if isinstance(p.raw, dict) else {}
    for ent in (raw.get("entities") or {}).get("urls") or []:
        u = ent.get("expanded_url") or ent.get("expandedUrl") or ent.get("url")
        if u:
            out.append(str(u))
    if not out:
        out = list(p.urls or [])
    return out


def _url_keys(urls: list[str]) -> tuple[set[str], set[str]]:
    """Return (exact normalized URL keys, non-redirector domains)."""
    keys: set[str] = set()
    domains: set[str] = set()
    for u in urls:
        keys.add(_norm_url(u))
        host = (urlparse(u).netloc or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host and host not in IGNORED_LINK_DOMAINS:
            domains.add(host)
    return keys, domains


def text_similarity(a: str, b: str) -> float:
    if not a.strip() or not b.strip():
        return 0.0
    try:
        x = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform([normalized_text(a), normalized_text(b)])
        return float(cosine_similarity(x[0:1], x[1:2])[0, 0])
    except ValueError:
        return 0.0


def batch_similarity(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0))
    try:
        x = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform([normalized_text(t) for t in texts])
        return cosine_similarity(x)
    except ValueError:
        return np.eye(len(texts))


@dataclass
class _MutableCluster:
    event_id: str
    symbol: str
    started_at: object
    last_seen_at: object
    representative_text: str
    post_ids: list[str] = field(default_factory=list)
    authors: set[str] = field(default_factory=set)
    domains: set[str] = field(default_factory=set)
    url_keys: set[str] = field(default_factory=set)
    similarities: list[float] = field(default_factory=list)
    actionable_at: object = None


def _observed_at(p: SocialPost) -> datetime:
    """When the post became visible to us (the tradable timeline)."""
    return p.first_seen_at or p.created_at


class EventClusterer:
    """Fast TF-IDF narrative clustering for a bounded recent window.

    This is deliberately deterministic and local. It is not meant to be the final
    semantic embedding model; it gives the research engine a reproducible event
    abstraction and prevents duplicate reposts from masquerading as new events.
    Clusters order and gap on *observation* time (first_seen_at), so a post
    indexed hours late by the vendor extends a narrative from when we could have
    acted on it, not from when it was authored.
    """

    def __init__(self, max_gap_minutes: int = 30, similarity: float = 0.36):
        self.max_gap = timedelta(minutes=max_gap_minutes)
        self.similarity = similarity

    def _pair_score(self, text: str, url_keys: set[str], domains: set[str], c: _MutableCluster, rep_sim: float) -> float:
        score = rep_sim
        if url_keys and c.url_keys and url_keys & c.url_keys:
            # Exact destination URL match is decisive — the same link IS the
            # same story, above whatever text-similarity gate is configured.
            score = 1.0
        elif domains and c.domains and domains & c.domains:
            # Same real (non-redirector) domain: a nudge, never decisive alone.
            score = min(1.0, score + 0.15)
        return score

    def cluster(self, posts: list[SocialPost]) -> list[EventCluster]:
        ordered = sorted(posts, key=lambda x: (_observed_at(x), x.created_at))
        if not ordered:
            return []
        sims = batch_similarity([p.text for p in ordered])
        clusters: list[_MutableCluster] = []
        rep_idx: list[int] = []

        for i, p in enumerate(ordered):
            seen = _observed_at(p)
            url_keys, domains = _url_keys(effective_urls(p))
            best: _MutableCluster | None = None
            best_score = 0.0
            for ci, c in enumerate(clusters):
                if c.symbol != p.symbol or seen - c.last_seen_at > self.max_gap:
                    continue
                score = self._pair_score(p.text, url_keys, domains, c, float(sims[i, rep_idx[ci]]))
                if score > best_score:
                    best_score, best = score, c

            if best is None or best_score < self.similarity:
                eid = hashlib.sha1(f"{p.symbol}|{p.post_id}".encode()).hexdigest()[:16]
                best = _MutableCluster(eid, p.symbol, p.created_at, seen, p.text, actionable_at=seen)
                clusters.append(best)
                rep_idx.append(i)
                best_score = 1.0

            best.post_ids.append(p.post_id)
            best.last_seen_at = max(best.last_seen_at, seen)
            best.actionable_at = min(best.actionable_at, seen) if best.actionable_at else seen
            best.similarities.append(best_score)
            if p.author_username:
                best.authors.add(p.author_username.lower())
            best.domains |= domains
            best.url_keys |= url_keys

        return [EventCluster(
            event_id=c.event_id,
            symbol=c.symbol,
            started_at=c.started_at,
            last_seen_at=c.last_seen_at,
            representative_text=c.representative_text,
            post_ids=c.post_ids,
            unique_authors=len(c.authors),
            source_domains=len(c.domains),
            similarity_mean=float(np.mean(c.similarities)) if c.similarities else 0.0,
            actionable_at=c.actionable_at,
        ) for c in clusters]

    def attach_or_create(
        self,
        posts: list[SocialPost],
        active_events: list[EventCluster],
    ) -> tuple[list[EventCluster], list[EventCluster]]:
        """Persistent lifecycle: attach posts to still-open events; cluster the rest.

        Returns (updated_existing_events, newly_created_events). Rolling windows
        re-running `cluster()` mint fresh event_ids for one long-lived narrative;
        attachment keeps the original identity so an event is never double-counted.
        """
        open_events = []
        for e in active_events:
            rep_urls = re.findall(r"https?://\S+", e.representative_text or "")
            rk, rd = _url_keys(rep_urls)
            open_events.append(_MutableCluster(
                e.event_id, e.symbol, e.started_at, e.last_seen_at, e.representative_text,
                post_ids=list(e.post_ids), actionable_at=e.actionable_at,
                domains=rd, url_keys=rk,
            ))
        remaining: list[SocialPost] = []
        for p in sorted(posts, key=lambda x: (_observed_at(x), x.created_at)):
            seen = _observed_at(p)
            url_keys, domains = _url_keys(effective_urls(p))
            best = None
            best_score = 0.0
            for c in open_events:
                if c.symbol != p.symbol or seen - c.last_seen_at > self.max_gap:
                    continue
                score = self._pair_score(p.text, url_keys, domains, c, text_similarity(p.text, c.representative_text))
                if score > best_score:
                    best_score, best = score, c
            if best is not None and best_score >= self.similarity:
                if p.post_id not in best.post_ids:
                    best.post_ids.append(p.post_id)
                    best.similarities.append(best_score)
                    if p.author_username:
                        best.authors.add(p.author_username.lower())
                best.last_seen_at = max(best.last_seen_at, seen)
                best.actionable_at = min(best.actionable_at, seen) if best.actionable_at else seen
                best.domains |= domains
                best.url_keys |= url_keys
            else:
                remaining.append(p)

        new_clusters = self.cluster(remaining)
        out_updated = [
            e.model_copy(update={
                "post_ids": c.post_ids, "last_seen_at": c.last_seen_at,
                "actionable_at": c.actionable_at, "unique_authors": len(c.authors),
                "source_domains": len(c.domains),
                "similarity_mean": float(np.mean(c.similarities)) if c.similarities else 0.0,
            }) for e, c in zip(active_events, open_events, strict=True)
            if e.post_ids != c.post_ids or e.last_seen_at != c.last_seen_at
        ]
        return out_updated, new_clusters


def assign_posts_to_events(store, symbol: str, posts: list[SocialPost], *, max_gap_minutes: int = 30, similarity: float = 0.36) -> dict:
    """Incremental event maintenance for live/historical rebuilds.

    Loads still-open events, attaches matching posts (stable event_id), clusters
    the remainder into new events. Posts already claimed by an event are skipped.
    """
    if not posts:
        return {"attached": 0, "new_events": 0}
    # Events live in per-symbol narrative universes; the function's symbol governs
    # (a caller passing a mixed-symbol list would otherwise create events whose
    # stored symbol disagrees with the queried universe).
    posts = [p if p.symbol == symbol else p.model_copy(update={"symbol": symbol}) for p in posts]
    clusterer = EventClusterer(max_gap_minutes=max_gap_minutes, similarity=similarity)
    now = max(_observed_at(p) for p in posts)
    active_payloads = store.events_between(symbol, now - timedelta(minutes=max_gap_minutes), now + timedelta(minutes=max_gap_minutes))
    active = [EventCluster.model_validate(x) for x in active_payloads]
    claimed = store.event_memberships([p.post_id for p in posts], symbol=symbol)
    unclaimed = [p for p in posts if p.post_id not in claimed]
    updated, new_events = clusterer.attach_or_create(unclaimed, active)
    for e in updated:
        store.save_event(e)
    for e in new_events:
        store.save_event(e)
    clustered_new = sum(len(e.post_ids) for e in new_events)
    return {"attached": len(unclaimed) - clustered_new, "new_events": len(new_events)}

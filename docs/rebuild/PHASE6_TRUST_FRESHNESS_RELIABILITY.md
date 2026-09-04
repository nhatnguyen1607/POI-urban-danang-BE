# Phase 6 - Trust, Data Freshness, and Reliability

Status: IMPLEMENTED_ON_REVIEW_BRANCH

## Trust model

- Canonical identity data remains separate from live-status evidence.
- Evidence records carry source type, observation/verification timestamps,
  TTL, verification method, confidence reason, status, and conflict metadata.
- Current status is asserted only from fresh verified evidence. Imported
  opening hours are reference-only.
- Conflicting comparable evidence is exposed as CONFLICT; it is never hidden
  by a synthetic winner.
- Accommodation availability is UNVERIFIED without a configured official
  partner and requires a provider handoff.
- GPS-verified post-visit feedback may become user evidence, but never mutates
  canonical POIs.

## Reliability controls

- Planner, agent, and route endpoint classes have configurable per-client
  limits with HTTP 429 and Retry-After.
- Photon, Google Places, OSRM, and weather boundaries use bounded concurrency,
  timeout, bounded transient retry with backoff/jitter, circuit breaking,
  request coalescing, and short-lived caches.
- Provider failures return controlled fallback/error behavior; logs contain
  operation metadata and hashed request keys, not raw queries or credentials.

## Product semantics

- Traveler surfaces show compact trust labels with source and verification
  details, while stale hours and unknown hotel availability remain explicit.
- Business analysis presents observable evidence, proxy signals, limitations,
  and a verification checklist. Ordering is internal navigation, not an
  investment score or financial recommendation.

## Validation limits

The load profiles are a local single-process baseline using only CSV-backed
search. They do not call public providers and do not establish production
capacity. Official live availability remains unavailable until credentials,
license approval, provider SLAs, and separate acceptance tests exist.

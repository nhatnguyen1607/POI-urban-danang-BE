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
- POI search, planner, agent, and route endpoint classes also have configurable
  active-work admission limits. They do not maintain a pending queue; excess
  capacity fails fast with HTTP 503 and Retry-After.
- Photon, Google Places, OSRM, and weather boundaries use bounded concurrency,
  timeout, bounded transient retry with backoff/jitter, circuit breaking,
  request coalescing, and short-lived caches.
- Provider failures return controlled fallback/error behavior; logs contain
  operation metadata and hashed request keys, not raw queries or credentials.
- Circuit recovery admits one HALF_OPEN probe only. Identical safe provider
  requests are coalesced and successful results use bounded TTL caches.

## Optional Google boundary

- Google Places and Google Maps remain optional legacy integrations.
- Both frontend and backend require an explicit enable flag in addition to a
  credential. Without that opt-in, normal search uses Photon and no Google
  endpoint is called.
- Core startup, external search, itinerary, route, trust/freshness, tests, and
  build do not require a Google credential or Google Billing.

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

## Pre-merge reliability gate

- Controlled local spike: 240 requests at concurrency 30; 16 successful, 224
  clean HTTP 503 responses, 0 HTTP 429, 0 timeout; p50 9.4 ms, p95 115.4 ms,
  p99 123.7 ms. Peak active work was 8, pending queue remained 0, and the
  health endpoint remained responsive after the spike.
- Coalescing/cache proof: 20 simultaneous identical clients produced one
  underlying execution and 19 coalesced followers; one repeated request inside
  TTL produced one cache hit. Operational events contained only a key hash.
- No-Google controlled flow: cities, Photon-backed external search,
  recommendations, trip preview, OSRM route, and trust overlay passed with no
  Google credential and zero Google network requests.
- Focused trust checks passed for stale hours, unverified hotel availability,
  equal-strength conflict, GPS evidence isolation, and neutral B2B output.

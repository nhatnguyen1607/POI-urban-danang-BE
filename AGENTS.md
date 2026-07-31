# UrbanAgent Backend Agent Rules

## Required reading

Before analyzing or changing this repository, read:

1. `URBANAGENT_CODEX_CONTEXT.md`
2. `PLANNING.md`
3. `README.md`
4. `package.json`
5. The sibling frontend repository:
   `../POI-urban-danang-FE`

The canonical project context is:

`POI-urban-danang-BE/URBANAGENT_CODEX_CONTEXT.md`

The copy in the frontend repository is a mirror. Never modify either
context file unless the user explicitly approves a context update.

## Project workflow

The project must be changed phase by phase.

1. Audit and planning
2. Phase 0 safety fixes
3. Data platform foundation
4. Traveler API v2
5. Traveler frontend rebuild
6. City Pack automation
7. Pilot and monetization
8. Partner product

Never begin a later phase until the user explicitly writes:

`APPROVED PHASE <number>`

## Change safety

- Preserve existing working functionality.
- Do not rewrite the full project in one task.
- Do not delete legacy endpoints until equivalent v2 endpoints pass tests.
- Do not fabricate POI fields.
- Do not replace missing coordinates with Da Nang center coordinates.
- Preserve missing rating, review-count, address, and admin-boundary values as unknown/null unless a verified source provides them.
- Treat `Global_ID` as the canonical legacy POI key for Phase 0.
- Treat `Alias_Global_IDs` as preserved merged-row IDs, not independent product POIs.
- Treat `RestaurantID` as a source identifier; do not call it Google `place_id` unless a source contract proves that.
- Exclude urban-void rows from traveler POIs, recommendations, itineraries, maps, and product POI counts.
- Do not log or expose secrets.
- Do not read or print Firebase Admin private keys.
- Do not modify production Firebase data.
- Do not run destructive database or Git commands without approval.
- Do not scrape Google Maps.
- Do not add external paid services without approval.

## Required project records

Maintain these files:

- `docs/rebuild/MASTER_PLAN.md`
- `docs/rebuild/CURRENT_STATE.md`
- `docs/rebuild/WORKLOG.md`
- `docs/rebuild/DECISIONS.md`
- `docs/rebuild/TEST_REPORT.md`
- `docs/rebuild/DATA_AUDIT.md`

`WORKLOG.md` must be append-only. Do not rewrite previous entries.

At the end of every implementation batch:

1. Run relevant tests and builds.
2. Update `CURRENT_STATE.md`.
3. Append to `WORKLOG.md`.
4. Update `TEST_REPORT.md`.
5. List changed files.
6. List unresolved risks.
7. Stop and wait for approval.

## Data locations

Phase 0 canonical runtime data:

- `data/canonical/urbanagent_poi_master_v1.csv`

Approved dataset decision:

- `docs/rebuild/URBANAGENT_DATASET_DECISION.md`

Expected canonical dataset contract:

- rows: `4166`
- unique `Global_ID`: `4166`
- SHA-256: `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`

Legacy/raw input data are source inputs/backups only and must not be overwritten:

- `data/raw/legacy/poi_data_ggmap.csv`
- `data/raw/legacy/poi_data_foody.csv`

The canonical CSV and manifest may not be modified unless the user explicitly approves a new dataset decision.

## Completion rule

Do not state that a phase is complete unless:

- the project builds,
- relevant tests pass,
- documentation is updated,
- existing compatibility is checked,
- remaining failures are reported honestly.

## External POI Source Guardrails

No agent may add, ingest, scrape, persist, cache, expose, train on, or
redistribute a new external POI source without an explicit approved source
onboarding decision.

Before changing code for a new source, the agent must read:

- `URBANAGENT_CODEX_CONTEXT.md`
- `docs/rebuild/MULTI_SOURCE_POI_STRATEGY.md`
- `docs/rebuild/DATA_SOURCE_LICENSE_POLICY.md`
- `docs/rebuild/DECISIONS.md`
- `docs/rebuild/CURRENT_STATE.md`

A source onboarding proposal must include:

- source owner,
- source documentation,
- license or contract version,
- allowed storage,
- allowed caching,
- allowed display,
- allowed API redistribution,
- attribution requirements,
- ML and derived-use rules,
- refresh and deletion rules,
- estimated cost,
- rate limits,
- data-quality sample,
- security classification,
- dry-run plan,
- rollback plan.

Never scrape Wanderlog, Google Maps, Tripadvisor, Yelp, Mindtrip, Layla, or
another commercial or competitor product as a substitute for an approved API,
open dataset, direct agreement, or licensed source.

No source-expansion work may change:

- the canonical CSV,
- the canonical SHA-256,
- the approved application POI count,
- the default runtime repository,

during an active Phase 2 batch.

Every multi-source spike must be isolated and reversible.

Every spike must produce:

- exact-match report,
- fuzzy-match report,
- ambiguity report,
- probable-new-entity report,
- duplicate report,
- source-conflict report,
- license report,
- attribution report,
- field-completeness report,
- cost report.

Do not automatically merge ambiguous POIs.

Do not invent missing ratings, reviews, opening hours, addresses, freshness,
coordinates, or provider identifiers.

Restricted provider payloads must not be committed to Git, included in Git
bundles, placed in public fixtures, or exposed through public APIs.

Required approval phrase before any source spike:

`APPROVED MULTI-SOURCE POI SPIKE`

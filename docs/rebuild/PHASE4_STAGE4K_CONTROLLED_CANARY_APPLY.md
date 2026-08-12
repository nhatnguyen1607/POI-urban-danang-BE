# Phase 4 Stage 4K: Controlled Canary Canonical Apply

## Status

`PASSED - REVIEW BRANCH ONLY - NOT MERGED`

Stage 4K is the first authorized Phase 4 canonical-data change. Its scope is
limited to safe additions for ten explicitly human-approved existing POIs. It
does not create, delete, merge, rename, recategorize, move, or change the
source identity of any POI.

## Approval Scope

The read-only human decision input contains exactly:

- `APPROVE`: 10;
- `REJECT`: 5;
- `DEFER`: 10;
- total unique cases: 25.

All ten approved cases have `STRONG` evidence and the proposed operation
`ENRICH_EXISTING_SAFE_FIELDS_ONLY`. They target ten distinct canonical POIs:

| Canonical ID | Canonical name | Approved safe fields |
| --- | --- | --- |
| `google_maps_241` | All Seasons Buffet Da Nang | address, website, phone |
| `google_maps_1557` | 40Plus Cafe & Vinyl Music | address, phone |
| `google_maps_2368` | Mỳ Quảng Cô Sáu | address, website, phone |
| `google_maps_2088` | BonPas Bakery & Cafe | address, phone |
| `google_maps_2716` | Roost Coffee Roasters \| Da Nang | address, phone |
| `google_maps_1156` | Hải Sản Quán Có | address, website, phone |
| `google_maps_1603` | Ly's Bakery | address, website, phone |
| `google_maps_1602` | Bakery Hoàng Phát | address, phone |
| `google_maps_5247` | Thìa Gỗ Đà Nẵng | address, website, phone |
| `google_maps_1738` | Gear Cafe’ And Bistro | address, website, phone |

The exact approval set and whitelist are enforced in code. Every `REJECT` and
`DEFER` case produces zero mutations.

## Apply Result

The canonical schema has `Address_Current` but no phone or website column.
Changing that schema would require runtime/header-contract changes outside
Stage 4K. The canary therefore applied only values that have a valid canonical
destination:

| Field | Planned | Applied | Skipped |
| --- | ---: | ---: | ---: |
| address | 10 | 10 | 0 |
| phone | 0 | 0 | 10 |
| website | 0 | 0 | 6 |
| **Total** | **10** | **10** | **16** |

All 16 skipped fields are classified
`SKIPPED_UNSUPPORTED_CANONICAL_FIELD`. No value was forced into an unrelated
column and no non-empty canonical value was overwritten.

Explicitly locked fields were unchanged: name, coordinates, latitude,
longitude, category, external IDs, canonical ID, source identity, and row
identity. In particular, `google_maps_5247` remains named
`Thìa Gỗ Đà Nẵng`.

## Canonical Integrity

- Rows before: `4166`.
- Rows after: `4166`.
- Distinct `Global_ID` after: `4166`.
- Canonical targets modified: `10`.
- Non-target rows unchanged: `4156/4156`.
- New rows: `0`.
- Deleted rows: `0`.
- Merged rows: `0`.
- SHA-256 before:
  `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
- SHA-256 after:
  `e1f7fd635087eecb56dac8a2f3ed810f481ff129a12d19332e2d68f08ed56f96`.

The updater preserves the UTF-8 BOM, CRLF line endings, row order, column
order, and untouched CSV tokens. Exact-diff validation permits only the ten
approved `Address_Current` additions.

## Provenance And License

Every applied field is traceable through:

`human_canary_decisions.csv -> APPROVE -> Stage 4J evidence case -> Overture
source record -> field provenance -> canonical target`.

All applied addresses retain Overture source ID, fixed snapshot reference,
`CDLA-Permissive-2.0`, `OPEN_PERMISSIVE_CANDIDATE`, attribution, and license
URL in the non-Git operational apply artifacts. All ten source IDs were also
found in the existing Stage 4G provenance/license manifest. Google is absent;
no OSM, Wikidata, or Commons field was added in this canary.

## Rollback

The pre-apply canonical file and metadata are stored outside Git on drive D:.
A controlled rollback was executed against the applied candidate in memory.
It restored a byte-identical canonical file with SHA-256
`5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`.
The feature branch intentionally retains the applied state.

## Runtime Compatibility

- Runtime code changed: `NO`.
- Runtime data changed: `YES`, only ten approved address additions.
- CSV repository remains the default runtime.
- PostgreSQL/PostGIS remains explicit opt-in.
- Canonical loader and quality semantics: `PASS`, 4166 application POIs.
- Recommendation and itinerary compatibility: `PASS`.
- Traveler API v2 recommendation smoke: `PASS`.
- Traveler API v2 trip-preview fixture and endpoint smoke: `PASS`.
- Public API schema changed: `NO`.
- External runtime provider introduced: `NO`.
- Production database or Firebase touched: `NO`.

The new canonical SHA is the Stage 4K baseline. Future incremental processing
must invalidate only state that depends on canonical identity/resolution hash;
Stage 4K did not rerun the full-city external candidate pipeline.

## Validation

- Stage 4K focused tests: `15 passed`, `0 failed`.
- Relevant Stage 4J approval/apply-plan tests: `6 passed`, `0 failed`.
- Phase 0 canonical/runtime compatibility tests: `10 passed`, `0 failed`.
- Traveler API v2 recommendation focused tests: `2 passed`, `0 failed`.
- Traveler API v2 trip-preview focused tests: `2 passed`, `0 failed`.
- Stage 4F provenance/license checks: `8 passed`; its historical
  canonical-baseline assertion is intentionally superseded by this authorized
  Stage 4K SHA.

## Limitations And Proposed Stage 4L

This ten-record canary is not evidence of city-wide enrichment quality. Phone
and website values remain unapplied because the current canonical CSV has no
compatible columns. No remaining review case is approved by this result.

Proposed Stage 4L only: merge the verified Stage 4K canary after user review,
then monitor canonical-loader, recommendation, trip-preview, attribution, and
rollback behavior on main before considering a separately reviewed second
enrichment batch. Do not auto-approve bulk candidates or `CREATE_NEW` cases.

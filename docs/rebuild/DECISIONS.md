# Decisions

Updated: 2026-07-26 01:49:31 +07:00.

## Accepted

- BE `URBANAGENT_CODEX_CONTEXT.md` remains the canonical context.
- FE `URBANAGENT_CODEX_CONTEXT.md` is a mirror and is byte-identical after this sync.
- The approved Phase 0 runtime dataset is `data/canonical/urbanagent_poi_master_v1.csv`.
- The canonical CSV and manifest were not modified.
- The dataset decision file was not modified.
- `Global_ID` is the canonical legacy key exposed by compatibility APIs.
- `RestaurantID` is a source identifier, not a verified Google `place_id`.
- `Alias_Global_IDs` preserves merged row IDs and must not create extra product POIs.
- Urban-void rows are excluded from traveler POIs, recommendations, itineraries, maps, and POI counts.
- Missing coordinates, address, admin-boundary, rating, review count, and freshness values must stay unknown/null unless verified.
- Phase 0 may add tests and safety foundation only.
- Phase 1 remains blocked until explicit approval.

## New Decisions In This Fix Batch

- Fix BOM at the CSV parser header layer instead of editing the canonical CSV.
- Use Node built-in `node:test` for Phase 0 backend tests, avoiding new dependencies.
- Keep Phase 0 tests under `tests/phase0/`.
- Expose `imageUrls` as the canonical normalized image collection and keep `imageUrl` as a compatibility first-image field.
- Keep frontend app code unchanged in this blocker fix; only sync frontend rules/context.
- Run frontend production build to `C:\tmp\urbanagent-fe-build-phase0-fix` to avoid modifying repo `dist`.
- Leave broad frontend lint debt unresolved and reported, per user instruction.

## Still Open

- Decide whether restored root CSV Git LFS pointer files should remain tracked for legacy compatibility in the eventual Phase 0 commit.
- Decide whether `.gitignore` changes from the earlier Phase 0 patch should be kept.
- Decide when to add a real frontend test runner.
- Decide whether Phase 1 should restore road-name density through verified address/admin data or leave route density proximity-only until PostGIS.

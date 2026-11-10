# NaviGo maintenance integration plan

This is a bounded maintenance audit, not a new product rebuild phase.
Architecture: React/Vite -> Express src/server.js -> canonical CSV repository,
explicit retrieval/planning services, Firebase persistence and route expert facade.
PostGIS is an explicit optional repository; Python multimodal inference requires
a separately provisioned artifact bundle. Detailed evidence: docs/maintenance/.

Problems: OSRM cached single-use Response bodies; matrix walk mismatch and zero
longitude loss; Phase4 current-baseline assertions stale; FE lint debt; model
artifact location/package not proven; old master-context status text is stale.

Target: preserve modular monolith and compatibility endpoints. Restore reviewed
runtime/test changes only. Keep current traveler UI, auth, canonical CSV,
provenance, repository abstraction and all history. No files split or legacy
endpoints deleted. No database/schema/data migration and no training changes.

Sequence: inventory every branch/PR; compare ancestry, patches and code; run
baseline; repair route/provider contracts and test expectations; run full suites,
Functions build and FE checks; publish separate integration PRs; preserve main
until deployment constraints and remaining checks permit merge.

Risks: external deployment on main merge; production Firebase/provider behavior
not covered by test doubles; actual model/index assets absent; historical City
Pack fixtures require preserved versions. Acceptance requires passing scoped
tests, unchanged canonical SHA and honest NOT VERIFIED flags; full system
verification cannot be claimed without real configured browser/API flows.

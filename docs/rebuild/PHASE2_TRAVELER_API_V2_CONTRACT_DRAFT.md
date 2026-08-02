# Phase 2 Traveler API v2 Contract Draft

Updated: 2026-07-26 22:04:35 +07:00.

Status: `IMPLEMENTED ON REVIEW BRANCH - NOT MERGED`.

This contract draft now has a Phase 2 Batch 1 implementation for city metadata,
city status, POI search, POI detail, common envelopes, request IDs, pagination,
and the OpenAPI artifact. Phase 2 Batch 2 implements standalone recommendation
v2. Itinerary preview v2 now has an approved documentation-only Batch 3 design
package but is not implemented. Conditional persistence routes are retained
for design continuity but are marked `CONDITIONAL - NOT APPROVED FOR
IMPLEMENTATION`.

## 1. Approved Contract Decisions

- API base path: `/api/v2`.
- Trip resource name: `/trips`.
- Standalone recommendation endpoint is accepted:
  `POST /api/v2/recommendations`.
- Guest preview is stateless and non-persistent by default.
- Raw internal scoring signals are not a public contract.
- Public recommendations expose `score`, `reason`, `reasonCodes`, `warnings`,
  and POI provenance.
- `cityId` is required for city-scoped endpoints.
- Only `GET /api/v2/cities` is not city-scoped.
- Da Nang is the only supported city in Phase 2.
- Unknown explicit `cityId` returns `CITY_NOT_SUPPORTED`.
- There is no silent Da Nang fallback for an explicit invalid `cityId`.
- OpenAPI 3.1 is `APPROVED AND REQUIRED` in Phase 2 Batch 1.
- Batch 1 must create a draft OpenAPI artifact covering only approved Batch 1
  endpoints and common schemas.
- Only after the OpenAPI artifact exists may validation record an actual
  `openApiSha256`, calculated from the generated artifact.
- Batch 1 OpenAPI artifact:
  `docs/rebuild/PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json`.
- Batch 2 OpenAPI SHA-256:
  `371e5de7db74b3fdeaf52999e2f417db0078309edb9ff5fe399dfec210c60da9`.

## 2. Public Metadata Boundary

Public v2 responses must not expose local filesystem paths, database table
names, SQL details, `DATABASE_URL`, internal repository class names, or
implementation-specific storage mode. Traveler clients must not depend on
whether POIs are backed by CSV or PostgreSQL.

Public metadata uses:

- `datasetVersion`
- `contractVersion`
- `applicationPoiCount`
- `qualitySummary`
- `capabilityStatus`

Repository mode may appear only in non-production diagnostics, test evidence,
or internal logs.

`contractVersion` is a version identifier, not a cryptographic hash. The Phase
2 draft value is `phase2-traveler-api-v2-draft-1`.

## 3. Null Semantics

Normative rule:

- Unknown is not an empty string.
- Unknown is not zero.
- Unknown is not `false` unless the field is truly Boolean.
- Unknown is represented by `null` and/or an explicit status field.

Examples:

- Unknown district: `"district": null`.
- Missing normalized rating: `"normalized": null`.
- Missing review count: `"count": null`, `"countStatus": "unknown"`.
- Missing freshness: `"observedAt": null`, `"lastVerifiedAt": null`.
- Unknown route totals: `"totalDistanceKm": null`, not zero.
- Missing origin: route leg distance/time is null and status says why.

## 4. Request ID Contract

- The server generates `requestId` when the request does not include one.
- Clients may send `X-Request-Id`.
- Accepted `X-Request-Id` length is 1 to 128 characters.
- Accepted characters are ASCII letters, digits, dot, underscore, colon, and
  hyphen.
- If `X-Request-Id` is missing or invalid, the server generates a new
  requestId and continues the request.
- The server must not echo an invalid supplied tracing value.
- Every v2 response returns the effective `requestId`.
- Logs use the same requestId.
- Logs must not include tokens, credentials, or sensitive payloads.
- An otherwise valid API request must not be rejected solely because a tracing
  header is invalid.

## 5. Authentication Contract

Core public read/preview endpoints are unauthenticated unless a later batch
adds a specific limit.

Conditional authenticated endpoints use:

```http
Authorization: Bearer <Firebase ID token>
```

Missing token:

- HTTP 401
- `AUTH_REQUIRED`

Invalid token:

- HTTP 401
- `INVALID_TOKEN`

Local development fallback:

- May be used only outside production for tests/local development.
- Must never work when production mode is active.
- Automated tests must not contact Firebase production.

This contract does not redesign authentication.

## 6. Common Metadata

Every v2 response requires only:

- `apiVersion`
- `requestId`

City-scoped responses also require:

- `cityId`

Do not require these fields on every endpoint:

- `datasetVersion`
- `applicationPoiCount`
- `qualitySummary`
- `capabilityStatus`

Use `datasetVersion` only where lineage context is relevant.
`applicationPoiCount`, `qualitySummary`, and `capabilityStatus` primarily
belong in `GET /api/v2/cities/:cityId/status`. They may appear in POI search
metadata only when explicitly justified. Recommendation and trip-preview
responses must not repeat the full city quality report by default.

## 7. Common Success Envelope

```json
{
  "ok": true,
  "data": {},
  "meta": {
    "apiVersion": "v2",
    "requestId": "req_20260726_abc123",
    "cityId": "da-nang"
  }
}
```

## 8. Common Error Envelope

```json
{
  "ok": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "query is required",
    "details": [
      {
        "field": "query",
        "rule": "non_empty_string"
      }
    ]
  },
  "meta": {
    "apiVersion": "v2",
    "requestId": "req_20260726_abc123"
  }
}
```

Recommended error codes:

| HTTP | Code | Meaning |
| --- | --- | --- |
| 400 | `VALIDATION_ERROR` | Missing or invalid request fields. |
| 401 | `AUTH_REQUIRED` | Firebase ID token is required. |
| 401 | `INVALID_TOKEN` | Firebase ID token is invalid. |
| 403 | `FORBIDDEN` | Authenticated user cannot perform action. |
| 404 | `NOT_FOUND` | Resource not found. |
| 409 | `STATE_CONFLICT` | Edit/replan conflicts with trip state. |
| 422 | `CITY_NOT_SUPPORTED` | City is not available in Phase 2. |
| 422 | `NO_FEASIBLE_ITINERARY` | A valid trip-preview request cannot produce a feasible preview. |
| 500 | `INTERNAL_ERROR` | Unexpected backend failure. |
| 502 | `UPSTREAM_UNAVAILABLE` | Optional upstream helper failed. |

## 9. Capability Status Model

A capability may be `available` only after implementation and validation.

Allowed states:

- `unavailable`: not implemented or not enabled.
- `planned`: designed but not approved for implementation yet.
- `experimental`: implemented only in non-production/test mode.
- `available`: implemented and validated for the current environment.

Phase 2 draft city status must not claim production readiness. Use
`PHASE2_DRAFT` or `EXPERIMENTAL`, not `READY_FOR_BETA`.

Capability transition rule:

- Planning examples may show `planned`.
- After a Batch 1 endpoint is implemented and validated, its capability may
  become `experimental` for local/test status.
- It becomes `available` only when the approved environment criteria pass.
- An implemented endpoint must not remain permanently marked `planned`.
- No status may claim production readiness unless separately approved.

Example:

```json
{
  "cityId": "da-nang",
  "status": "PHASE2_DRAFT",
  "capabilityStatus": {
    "poiSearch": "planned",
    "recommendations": "experimental",
    "tripPreview": "planned",
    "tripSave": "unavailable",
    "tripEdit": "unavailable",
    "tripReplan": "unavailable",
    "feedbackPersistence": "unavailable",
    "liveOpeningHours": "unavailable",
    "liveBooking": "unavailable",
    "roadNetworkRouting": "unavailable"
  }
}
```

## 10. Shared Models

### SourceIdentifier

`sourceIdentifiers` is the public typed collection for source identifiers.

```json
{
  "namespace": "restaurant_id",
  "value": "123456",
  "source": "foody"
}
```

Rules:

- `RestaurantID` is a source identifier.
- `RestaurantID` is never assumed to be a Google Place ID.
- Google, Foody, canonical, and future adapter IDs must be namespaced.

### RatingSummary

Normalized rating and source ratings are separate.

```json
{
  "normalized": {
    "value": 4.25,
    "scale": 5,
    "status": "known"
  },
  "google": {
    "value": 4.3,
    "scale": 5,
    "ratingCount": 120,
    "ratingCountStatus": "known"
  },
  "foody": {
    "value": 8.5,
    "scale": 10,
    "sampleReviewCount": 2,
    "sampleReviewCountStatus": "known"
  },
  "reviewCount": {
    "value": null,
    "status": "unknown"
  }
}
```

Rules:

- Normalized rating is scale 5.
- Foody source rating remains scale 10 when available.
- Review count is nullable.
- Sample review count is not total review count.

### TravelerPoi

```json
{
  "id": "DN000001",
  "globalId": "DN000001",
  "cityId": "da-nang",
  "name": "Example Cafe",
  "category": "Cafe",
  "categoryNormalized": "cafe",
  "location": {
    "lat": 16.06,
    "lon": 108.22,
    "hasCoordinates": true,
    "coordinateStatus": "valid"
  },
  "address": {
    "current": null,
    "raw": null,
    "district": null,
    "adminNormalizationStatus": "pending_spatial_join"
  },
  "rating": {
    "normalized": {
      "value": null,
      "scale": 5,
      "status": "unknown"
    },
    "google": {
      "value": null,
      "scale": 5,
      "ratingCount": null,
      "ratingCountStatus": "unknown"
    },
    "foody": {
      "value": null,
      "scale": 10,
      "sampleReviewCount": null,
      "sampleReviewCountStatus": "unknown"
    },
    "reviewCount": {
      "value": null,
      "status": "unknown"
    }
  },
  "images": {
    "imageUrls": [],
    "imageUrl": null
  },
  "provenance": {
    "source": "google_maps+foody",
    "sourceIdentifiers": [],
    "aliasGlobalIds": [],
    "mergeStatus": "merged",
    "dataQualityFlags": []
  },
  "freshness": {
    "observedAt": null,
    "lastVerifiedAt": null,
    "status": "unknown"
  },
  "warnings": [
    "OPENING_HOURS_UNKNOWN",
    "address_unknown"
  ]
}
```

### RecommendationItem

```json
{
  "poi": {},
  "score": 86,
  "reason": "Matches the requested cafe intent and traveler context.",
  "reasonCodes": [
    "intent_match",
    "category_match"
  ],
  "warnings": [],
  "provenance": {
    "source": "google_maps+foody",
    "sourceIdentifiers": []
  }
}
```

Rules:

- `score` range is 0 to 100.
- Raw internal signal weights are not public.
- Debug/raw signals may exist only in explicit non-production diagnostics.
- Tie-breaking is deterministic: score descending, then normalized name
  ascending, then canonical `Global_ID` ascending.

### TripStop

```json
{
  "stopId": "stop_1",
  "order": 1,
  "poi": {},
  "travelFromPrevious": {
    "distanceMeters": null,
    "travelDurationMinutes": null,
    "transport": "motorbike",
    "travelMode": "motorbike",
    "estimationMethod": null,
    "estimationPolicyVersion": null,
    "calculationSource": "missing-origin",
    "distanceKnown": false,
    "travelTimeKnown": false
  },
  "durationMinutes": 60,
  "durationSource": "category_default",
  "durationPolicyVersion": "phase2-batch3-duration-v1",
  "reason": "Structured stop reason.",
  "reasonCodes": [
    "intent_match"
  ],
  "warnings": []
}
```

### RouteSummary

```json
{
  "totalDistanceKm": null,
  "totalTravelMinutes": null,
  "totalStayMinutes": 165,
  "totalPlanMinutes": null,
  "distanceFullyKnown": false,
  "travelTimeFullyKnown": false,
  "knownLegCount": 2,
  "unknownLegCount": 1,
  "calculationSource": "partial-local-haversine-estimate",
  "status": "partial",
  "warnings": [
    "ORIGIN_NOT_PROVIDED"
  ]
}
```

Rules:

- Unknown route totals must not be represented as zero.
- If any required leg is unknown, total distance and total travel time are
  either `null` or explicitly marked partial.
- `knownLegCount` and `unknownLegCount` must explain partial totals.
- `calculationSource` must not imply road-network routing unless that exists.

### TripPlan

```json
{
  "tripId": null,
  "preview": true,
  "authenticated": false,
  "saveEligible": false,
  "cityId": "da-nang",
  "query": "quan cafe yen tinh",
  "durationMinutes": 180,
  "transport": "motorbike",
  "stops": [],
  "routeSummary": {},
  "alternatives": [],
  "explanation": {
    "summary": null,
    "reasonCodes": []
  },
  "dataFreshness": {
    "source": "canonical_dataset",
    "observedAt": null,
    "lastVerifiedAt": null,
    "status": "unknown"
  },
  "warnings": []
}
```

## 11. Pagination Contract

Query parameters:

| Name | Required | Rule |
| --- | --- | --- |
| `cityId` | yes | Supported city ID. |
| `q` | no | Text query. |
| `category` | no | Category filter. |
| `source` | no | `google_maps`, `foody`, `all`, or `canonical`. |
| `limit` | no | Default `20`, maximum `100`. |
| `cursor` | no | Opaque cursor returned by the previous page. |

Page response:

```json
{
  "items": [],
  "page": {
    "total": 4166,
    "limit": 20,
    "nextCursor": null
  }
}
```

Sorting:

- If `q` is present: relevance descending, then normalized name ascending, then
  canonical `Global_ID` ascending.
- If `q` is absent: normalized name ascending, then canonical `Global_ID`
  ascending.
- The logical pagination contract must behave consistently for CSV and
  PostgreSQL implementations.
- Batch 1 owns limit validation, default limit 20, maximum limit 100, opaque
  cursor behavior, cursor validation, deterministic search/list sorting,
  canonical `Global_ID` final tie-break, `total`, `nextCursor`, CSV-default
  endpoint behavior, and OpenAPI contract coverage for POI search.

## 12. Core Approved Scope

The following endpoints are approved for Phase 2 specification and may be
implemented only after the user approves the implementation batch.

### GET `/api/v2/cities`

Purpose: list supported traveler cities.

Auth: none.

Success:

```json
{
  "ok": true,
  "data": {
    "cities": [
      {
        "cityId": "da-nang",
        "displayName": "Da Nang",
        "countryCode": "VN",
        "timezone": "Asia/Ho_Chi_Minh",
        "currency": "VND",
        "status": "PHASE2_DRAFT",
        "capabilityStatus": {
          "poiSearch": "planned",
          "recommendations": "experimental",
          "tripPreview": "planned",
          "tripSave": "unavailable",
          "tripEdit": "unavailable",
          "tripReplan": "unavailable"
        }
      }
    ]
  },
  "meta": {
    "apiVersion": "v2",
    "requestId": "req_20260726_abc123"
  }
}
```

### GET `/api/v2/cities/:cityId/status`

Purpose: return one city status, dataset contract metadata, quality summary,
and capability status.

Auth: none.

Success:

```json
{
  "ok": true,
  "data": {
    "city": {
      "cityId": "da-nang",
      "displayName": "Da Nang",
      "status": "PHASE2_DRAFT"
    },
    "dataset": {
      "datasetVersion": "urbanagent-poi-master-v1",
      "contractVersion": "phase2-traveler-api-v2-draft-1",
      "applicationPoiCount": 4166
    },
    "qualitySummary": {
      "status": "known_gaps",
      "missingAddressStatus": "known_gap",
      "missingRatingStatus": "known_gap",
      "missingReviewCountStatus": "known_gap",
      "openingHoursStatus": "unknown",
      "freshnessStatus": "unknown"
    },
    "capabilityStatus": {
      "poiSearch": "planned",
      "recommendations": "experimental",
      "tripPreview": "planned",
      "tripSave": "unavailable",
      "tripEdit": "unavailable",
      "tripReplan": "unavailable"
    }
  },
  "meta": {
    "apiVersion": "v2",
    "requestId": "req_20260726_abc123",
    "cityId": "da-nang"
  }
}
```

Unsupported city:

```json
{
  "ok": false,
  "error": {
    "code": "CITY_NOT_SUPPORTED",
    "message": "City is not supported in Phase 2.",
    "details": [
      {
        "field": "cityId",
        "value": "hue"
      }
    ]
  },
  "meta": {
    "apiVersion": "v2",
    "requestId": "req_20260726_abc123"
  }
}
```

### GET `/api/v2/pois/search`

Purpose: search/list traveler-safe canonical POIs.

Auth: none.

Rules:

- `cityId` is required.
- Unknown city returns `CITY_NOT_SUPPORTED`.
- Default limit is 20; maximum limit is 100.
- Cursor is opaque and must be validated.
- Response includes `total`, `limit`, and `nextCursor`.
- When `q` is present, sort by relevance descending, normalized name
  ascending, and canonical `Global_ID` ascending.
- When `q` is absent, sort by normalized name ascending and canonical
  `Global_ID` ascending.
- Logical pagination must behave consistently for CSV and PostgreSQL.

Acceptance:

- `source=google_maps` uses merged-source compatibility and totals 3946.
- `source=foody` uses merged-source compatibility and totals 225.
- `source=all` or `source=canonical` totals 4166.
- No urban-void rows are returned.

### GET `/api/v2/pois/:poiId`

Purpose: return one traveler-safe POI by canonical `Global_ID`.

Auth: none.

Rules:

- `cityId` is required.
- Lookup uses canonical `Global_ID`.
- `RestaurantID` is not accepted as `placeId`.
- Merged aliases may be returned in provenance but do not create extra POIs.

### POST `/api/v2/recommendations`

Purpose: return traveler POI recommendations for a query.

Auth: optional.

Request:

```json
{
  "cityId": "da-nang",
  "query": "quan cafe yen tinh",
  "context": {
    "location": {
      "lat": 16.06,
      "lon": 108.22
    },
    "durationMinutes": 180,
    "maxDistanceKm": 14
  },
  "limit": 5
}
```

Acceptance:

- Nonempty result smoke for `quan cafe yen tinh`.
- Public output uses `score`, `reason`, `reasonCodes`, `warnings`, and POI
  provenance.
- Raw scoring signals are not public contract.
- Deterministic tie-breaking: score descending, normalized name ascending,
  canonical `Global_ID` ascending.
- Missing `cityId` returns `VALIDATION_ERROR`.
- Unsupported explicit `cityId` returns `CITY_NOT_SUPPORTED`.
- Default limit is `5`; maximum limit is `20`.
- The smoke/evaluation fixture foundation is not a scored relevance benchmark
  and must not be used for recommendation-quality superiority claims.

### POST `/api/v2/trips/preview`

Purpose: stateless, guest-safe itinerary preview.

Auth: optional.

Status: `APPROVED DESIGN FOR PHASE 2 BATCH 3 - DOCUMENTATION PENDING MERGE - NOT IMPLEMENTED`.

Authoritative Batch 3 design package:

- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_SCOPE.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_API_CONTRACT.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_EVALUATION_PLAN.md`
- `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_IMPLEMENTATION_BOUNDARIES.md`

Draft request decisions:

- `cityId` is required and only `da-nang` is supported in Batch 3.
- `query` is required and must be nonempty.
- `trip.durationMinutes` is optional; when supplied it must be an integer from
  15 to 480.
- `startLocation` is optional.
- If `startLocation` is present, latitude and longitude are both required and
  must be valid numeric coordinates.
- `constraints.maxStopsPerDay` is optional, minimum 1 and maximum 6.
- `recommendationOptions.limit` is optional, minimum 1 and maximum 30.

Acceptance:

- Does not require login.
- Does not persist guest trip by default.
- Missing origin remains unknown/null.
- Unknown route totals are null or partial, never zero.
- No Da Nang center coordinate fallback.
- Response uses the existing v2 `ok/data/meta` envelope.
- Trip stops appear under `data.trip.stops`.
- Missing-origin first leg uses `calculationSource: "missing-origin"`.
- Known later legs may use `calculationSource: "local-haversine-estimate"`.
- Road-network routing is not part of Batch 3.
- Opening hours are used only if approved runtime data exists; otherwise the
  response emits `OPENING_HOURS_UNKNOWN`.

## 13. Conditional Scope - Not Approved For Implementation

The following routes are contract drafts only. They require a later explicit
persistence decision before implementation.

### POST `/api/v2/trips`

Status: `CONDITIONAL - NOT APPROVED FOR IMPLEMENTATION`.

Purpose: create and persist an authenticated traveler trip.

Auth: required.

### GET `/api/v2/trips/:tripId`

Status: `CONDITIONAL - NOT APPROVED FOR IMPLEMENTATION`.

Purpose: return an authenticated saved trip.

Auth: required.

### PATCH `/api/v2/trips/:tripId`

Status: `CONDITIONAL - NOT APPROVED FOR IMPLEMENTATION`.

Purpose: update trip metadata or editable draft fields.

Auth: required.

### POST `/api/v2/trips/:tripId/replan`

Status: `CONDITIONAL - NOT APPROVED FOR IMPLEMENTATION`.

Purpose: replan remaining stops from current state.

Auth: required.

### POST `/api/v2/trips/:tripId/stops`

Status: `CONDITIONAL - NOT APPROVED FOR IMPLEMENTATION`.

Purpose: add a POI stop to a trip.

Auth: required.

### DELETE `/api/v2/trips/:tripId/stops/:stopId`

Status: `CONDITIONAL - NOT APPROVED FOR IMPLEMENTATION`.

Purpose: remove a stop from a trip.

Auth: required.

### POST `/api/v2/trips/:tripId/feedback`

Status: `CONDITIONAL - NOT APPROVED FOR IMPLEMENTATION`.

Purpose: persist traveler feedback on a trip or stop.

Auth: required for saved trips.

## 14. Optional Capability Endpoint

`GET /api/v2/meta/capabilities` remains optional. It is not required for Batch
1 unless the user approves it.

Any capability response must use capability states, not premature booleans.

## 15. Compatibility Test Matrix

| Behavior | Legacy route | v2 route |
| --- | --- | --- |
| Google source count | `/api/eda?source=google_maps` -> 3946 | `/api/v2/pois/search?cityId=da-nang&source=google_maps` -> equivalent total |
| Foody source count | `/api/eda?source=foody` -> 225 | `/api/v2/pois/search?cityId=da-nang&source=foody` -> equivalent total |
| Quality count | `/api/pois/data-quality` -> 4166 | `/api/v2/cities/da-nang/status` -> 4166 |
| Recommendation | `/api/agent/recommend-poi` nonempty | `/api/v2/recommendations` nonempty |
| Itinerary preview | legacy preview route where enabled | `/api/v2/trips/preview` nonempty |
| Missing origin | first leg null distance/time | first leg null distance/time |
| Runtime | CSV default | CSV default |
| Optional Postgres | explicit env only | explicit env only |

## 16. Open Contract Questions

- Does conditional trip persistence belong in Phase 2?
- If persistence is approved, does Firestore remain the persistence mechanism?
- Is a Firebase emulator required for conditional persistence tests?
- What process and reviewers will create the curated query fixture?

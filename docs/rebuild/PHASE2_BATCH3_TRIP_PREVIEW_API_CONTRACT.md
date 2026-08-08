# UrbanAgent Phase 2 Batch 3 - Trip Preview API Contract

Status: IMPLEMENTED ON REVIEW BRANCH - NOT MERGED

Updated: 2026-08-01

Primary endpoint:

`POST /api/v2/trips/preview`

This file specifies the approved design contract for later implementation. It must not be treated as active runtime behavior until documentation is merged, post-merge validation passes, and Batch 3 runtime code is implemented in a later task.

## 1. Endpoint Summary

`POST /api/v2/trips/preview` returns a stateless itinerary preview for Da Nang using the approved canonical POI runtime.

The endpoint:

- accepts a traveler intent and trip constraints,
- obtains candidates from the Batch 2 recommendation flow,
- chooses and orders stops deterministically,
- estimates only known travel legs,
- preserves missing-origin and missing-data semantics,
- returns warnings instead of fabricating data,
- does not persist the trip.

## 2. Headers

Supported request headers:

- `Content-Type: application/json`
- `X-Request-Id`: optional caller-provided request identifier.

If `X-Request-Id` is absent, the server may generate a public-safe request ID using the existing v2 request metadata pattern.

No authentication header is required for Batch 3. If a caller sends authentication metadata, the Batch 3 preview remains stateless and must not persist a trip.

## 3. Request Body

Proposed request shape:

```json
{
  "cityId": "da-nang",
  "query": "quiet cafes and a riverside walk",
  "trip": {
    "dayCount": 1,
    "date": "2026-08-15",
    "startTime": "14:00",
    "durationMinutes": 180,
    "transport": "motorbike",
    "pace": "balanced",
    "dailyWindow": {
      "start": "14:00",
      "end": "17:00"
    },
    "party": {
      "adults": 2,
      "children": 0
    },
    "budget": "moderate"
  },
  "startLocation": {
    "lat": 16.0678,
    "lon": 108.2208,
    "label": "hotel"
  },
  "preferences": {
    "interests": ["cafe", "riverside", "view"],
    "food": ["local"],
    "avoid": ["crowded"],
    "indoorOutdoor": "mixed"
  },
  "constraints": {
    "mustIncludePoiIds": [],
    "excludePoiIds": [],
    "maxStopsPerDay": 4,
    "maxDistanceKm": 14
  },
  "recommendationOptions": {
    "limit": 12
  }
}
```

## 4. Request Fields

`cityId`

- Required.
- Only `da-nang` is supported in Batch 3.
- Other values return `CITY_NOT_SUPPORTED`.

`query`

- Required.
- String.
- Trimmed length: 1 to 500 characters.
- Used as the primary traveler intent for Batch 2 recommendations.

`trip.date`

- Optional.
- ISO calendar date string: `YYYY-MM-DD`.
- If absent, calendar-specific opening-hours validation is unknown.

`trip.startTime`

- Optional.
- Local 24-hour time string: `HH:mm`.
- If absent, the preview may return relative timing only.

`trip.durationMinutes`

- Optional.
- Integer.
- Minimum: 15.
- Maximum: 480.
- When present, used as a hard scheduling budget for dwell time plus known travel time.
- When absent, daily-window and per-stop duration policy drive schedule construction.

`trip.transport`

- Optional.
- Enum: `walk`, `motorbike`, `car`, `taxi`.
- Default: `motorbike`.
- Used only for approximate known-leg travel-time estimates.

`trip.pace`

- Optional.
- Enum: `relaxed`, `balanced`, `packed`.
- Default: `balanced`.
- Controls target dwell-time pressure and stop count.

`trip.dayCount`

- Optional integer.
- Minimum: 1.
- Maximum: 7.
- Default: 1.
- Batch 3 uses a one-week maximum to keep the preview bounded and deterministic.

`trip.dailyWindow`

- Optional object.
- If present, both start and end values are required local `HH:mm` values.
- Existing `start` / `end` request fields remain supported.
- Demo-compatible `startTime` / `endTime` aliases are also supported for
  frontend clients that already model daily windows with those names.
- `end` must be strictly after `start` on the same day.
- Window span must be at least 15 minutes and at most 960 minutes. This window
  bound is separate from the existing `trip.durationMinutes` scheduling-budget
  bound.

`trip.dayWindows`

- Optional array.
- Maximum: 7 items.
- Each item overrides `trip.dailyWindow` for one `dayNumber`.
- `dayNumber` must be unique, 1 to 7, and within `trip.dayCount`.
- Each item accepts either `start` / `end` or `startTime` / `endTime` local
  `HH:mm` fields.
- This field is a backward-compatible demo extension for video-ready
  multi-day previews. It does not create persisted trips, saved schedules, or
  editable trip state.

`trip.party`

- Optional.
- Public traveler party hints.
- Batch 3 must not persist this object.

`trip.budget`

- Optional.
- Enum: `budget`, `moderate`, `premium`, `unknown`.
- Default: `unknown`.
- Used only as a recommendation hint if the existing recommendation layer supports it.

`startLocation`

- Optional.
- If present, both `lat` and `lon` are required and must be valid numeric coordinates.
- If absent, the first travel leg must remain unknown.
- No Da Nang center fallback is allowed.

`preferences`

- Optional.
- Structured hints for candidate selection.
- Unknown keys should be ignored or reported as validation warnings, not persisted.

`constraints.maxStopsPerDay`

- Optional integer.
- Minimum: 1.
- Maximum: 6.
- Default is derived from pace:
  - `relaxed`: 3
  - `balanced`: 4
  - `packed`: 6

`constraints.mustIncludePoiIds`

- Optional unique array.
- Maximum: 20 canonical POI IDs.
- Invalid or unsupported IDs are handled as deterministic warnings or hard-constraint failures depending on request semantics.

`constraints.excludePoiIds`

- Optional unique array.
- Maximum: 100 canonical POI IDs.
- Excluded IDs must not appear in scheduled stops.

`constraints.maxDistanceKm`

- Optional positive number.
- Applies only to known approximate route distance.
- Must not be enforced against unknown-origin distance.

`recommendationOptions.limit`

- Optional integer.
- Minimum: 1.
- Maximum: 30.
- Default: 12.
- This is the candidate request size, not the final stop count.

## 5. Validation Rules

The implementation must reject:

- unsupported `cityId`,
- missing or empty `query`,
- non-object request body,
- invalid `trip.durationMinutes`,
- invalid `trip.dayCount`,
- invalid `trip.dailyWindow`,
- invalid coordinate type,
- coordinates outside valid geographic ranges,
- malformed date or time values,
- unsupported enum values,
- `maxStopsPerDay` outside bounds,
- too many `mustIncludePoiIds`,
- too many `excludePoiIds`,
- `recommendationOptions.limit` outside bounds.

Coordinate validation:

- `lat` must be between `-90` and `90`.
- `lon` must be between `-180` and `180`.
- If `startLocation` is present, missing either coordinate is a validation error.
- Coordinates must not be replaced by Da Nang center coordinates.

## 6. Success Response

Proposed success envelope using the existing Traveler API v2 `ok/data/meta` contract:

```json
{
  "ok": true,
  "data": {
    "trip": {
      "tripId": null,
      "preview": true,
      "persisted": false,
      "authenticated": false,
      "saveEligible": false,
      "cityId": "da-nang",
      "query": "quiet cafes and a riverside walk",
      "durationMinutes": 180,
      "transport": "motorbike",
      "pace": "balanced",
      "date": "2026-08-15",
      "startTime": "14:00",
      "dayCount": 1,
      "dailyWindow": {
        "start": "14:00",
        "end": "17:00"
      },
      "timeKnown": true,
      "feasibilityStatus": "FEASIBLE_WITH_WARNINGS",
      "stops": [
        {
          "stopId": "stop_1",
          "order": 1,
          "poi": {
            "id": "DN000001",
            "globalId": "DN000001",
            "cityId": "da-nang",
            "name": "Example Cafe",
            "category": "Cafe",
            "categoryNormalized": "cafe",
            "location": {
              "lat": 16.067,
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
              "reviewCount": {
                "value": null,
                "status": "unknown"
              }
            },
            "images": {
              "imageUrl": null,
              "imageUrls": []
            }
          },
          "arrivalTime": "14:00",
          "departureTime": "14:55",
          "durationMinutes": 60,
          "durationSource": "category_default",
          "durationPolicyVersion": "phase2-batch3-duration-v1",
          "travelFromPrevious": {
            "distanceMeters": null,
            "travelDurationMinutes": null,
            "transport": "motorbike",
            "travelMode": "motorbike",
            "estimationMethod": null,
            "estimationPolicyVersion": null,
            "calculationSource": "missing-origin",
            "distanceKnown": false,
            "travelTimeKnown": false,
            "warnings": [
              {
                "code": "ORIGIN_NOT_PROVIDED",
                "severity": "info",
                "message": "Start location was not provided; first-leg distance and travel time are unknown.",
                "scope": "leg"
              }
            ]
          },
          "reason": "Matches the requested cafe intent and traveler context.",
          "reasonCodes": ["intent_match", "category_match", "ranked_candidate"],
          "warnings": ["DURATION_ESTIMATED", "ORIGIN_NOT_PROVIDED", "OPENING_HOURS_UNKNOWN"]
        },
      ],
      "routeSummary": {
        "totalDistanceKm": null,
        "totalTravelMinutes": null,
        "totalStayMinutes": 60,
        "totalPlanMinutes": null,
        "distanceFullyKnown": false,
        "travelTimeFullyKnown": false,
        "knownLegCount": 0,
        "unknownLegCount": 1,
        "calculationSource": "partial-local-haversine-estimate",
        "status": "partial",
        "warnings": ["DURATION_ESTIMATED", "ORIGIN_NOT_PROVIDED", "OPENING_HOURS_UNKNOWN"]
      },
      "alternatives": [],
      "explanation": {
        "summary": "Preview generated from deterministic recommendation candidates and approximate local travel estimates.",
        "reasonCodes": ["intent_match", "category_match", "route_preview"]
      },
      "dataFreshness": {
        "source": "canonical_dataset",
        "observedAt": null,
        "lastVerifiedAt": null,
        "status": "unknown"
      },
      "warnings": [
        {
          "code": "ORIGIN_NOT_PROVIDED",
          "message": "Start location was not provided; first-leg distance and travel time are unknown."
        },
        {
          "code": "OPENING_HOURS_UNKNOWN",
          "message": "Verified opening-hours data is not available for one or more stops."
        }
      ],
      "provenance": {
        "source": "canonical",
        "datasetVersion": "urbanagent-poi-master-v1",
        "externalLiveDataUsed": false
      }
    },
  "meta": {
    "apiVersion": "v2",
    "requestId": "req_20260801_001",
    "cityId": "da-nang"
  }
}
```

## 7. Stop Object Semantics

Each itinerary stop must include:

- `order`: one-based integer order.
- `poi`: public-safe canonical POI summary.
- `arrivalTime`: local `HH:mm` when time is known, otherwise `null`.
- `departureTime`: local `HH:mm` when time is known, otherwise `null`.
- `durationMinutes`: deterministic positive integer.
- `durationSource`: `requested`, `category_default`, or `fallback`.
- `durationPolicyVersion`: versioned duration policy identifier.
- `travelFromPrevious`: leg summary from the previous known point.
- `reasonCodes`: deterministic machine-readable reasons.
- `warnings`: deterministic stop-level warning codes.

The `poi` summary must preserve Phase 0 semantics:

- no fabricated coordinates,
- no fabricated address,
- missing rating remains `null`,
- missing review count remains `null`,
- `imageUrls` is an array,
- `imageUrl` is `imageUrls[0]` or `null`.

## 8. Travel Leg Semantics

When the origin for a leg is unknown:

- `distanceMeters`: `null`
- `travelDurationMinutes`: `null`
- `estimationMethod`: `null`
- `estimationPolicyVersion`: `null`
- `distanceKnown`: `false`
- `calculationSource`: `missing-origin` may be included as an explanatory field.
- `warnings`: public-safe leg-level warning objects.

`calculationSource` must not replace `estimationMethod`.

Travel from the unknown origin to the first stop is not included in schedule feasibility.

No Da Nang center fallback, hotel fallback, or synthetic coordinate may be used.

When both endpoints have valid coordinates:

- `distanceMeters`: numeric integer rounded to the nearest 10 meters,
- `travelDurationMinutes`: numeric integer rounded up to whole minutes,
- `travelMode`: request travel mode,
- `estimationMethod`: `local-haversine-estimate`,
- `estimationPolicyVersion`: `phase2-batch3-travel-time-v1`,
- `distanceKnown`: `true`,
- `calculationSource`: optional duplicate explanatory label.
- warnings include `TRAVEL_TIME_ESTIMATED`.

Travel estimates are not road-network routes. They must be documented as approximate.

## 9. Route Summary Semantics

If any leg is unknown:

- `routeSummary.distanceFullyKnown` must be `false`.
- `routeSummary.travelTimeFullyKnown` must be `false`.
- aggregate distance and travel minutes must be `null` unless the response also exposes explicitly partial known-only totals.
- `knownLegCount` and `unknownLegCount` must explain the partial route.
- warnings must explain partial route knowledge.

If all legs are known:

- `routeSummary.distanceFullyKnown` is `true`,
- `routeSummary.travelTimeFullyKnown` is `true`,
- `totalDistanceKm` is numeric,
- `totalTravelMinutes` is numeric,
- `calculationSource` is `local-haversine-estimate`.

## 10. Deterministic Itinerary Construction

The implementation should use this deterministic sequence:

1. Validate request.
2. Resolve `cityId`; reject anything outside Da Nang.
3. Normalize traveler hints without persisting them.
4. Request ranked candidates from the Batch 2 recommendation logic.
5. Remove excluded IDs.
6. Validate must-include IDs against the same city and traveler-eligible POI set.
7. Choose target stop count from `maxStopsPerDay`, pace, duration budget, and daily window.
8. Resolve stop duration by requested, category default, or fallback policy.
9. Build the ordered route:
   - with a known start location, prefer nearest feasible candidate after recommendation score,
   - without a start location, keep the first stop recommendation-led and mark first leg unknown,
   - for later legs, use approximate geographic ordering between valid coordinates.
10. Break ties by recommendation rank, canonical POI ID, then normalized name.
11. Trim or reduce stops until duration constraints are satisfied.
12. Attach warnings for missing or uncertain data.
13. Return a non-persistent preview response.

## 11. Schedule And Duration Feasibility

Duration feasibility includes:

- dwell time at each stop,
- known travel time between stops,
- optional time-window constraints.

Requested `durationMinutes` is optional and must be 15 to 480 minutes when supplied.

Stop durations use policy version `phase2-batch3-duration-v1` and expose `durationMinutes`, `durationSource`, and `durationPolicyVersion`.

If `startTime` is known:

- arrival and departure times are absolute local times,
- no timezone conversion is required for Batch 3,
- times are formatted as `HH:mm`.

If `startTime` is absent:

- arrival and departure times are `null`,
- relative offsets may be returned only if explicitly included in the final implementation contract,
- warnings should include `schedule_time_unknown`.

If no feasible stop can be selected:

- return a controlled error such as `NO_FEASIBLE_ITINERARY`,
- include public-safe guidance,
- do not return fabricated stops.

## 12. Travel-Time Policy

Travel policy version: `phase2-batch3-travel-time-v1`.

Every calculated leg uses `estimationMethod: "local-haversine-estimate"` and must include warning `TRAVEL_TIME_ESTIMATED`.

| Travel mode | Speed km/h | Transfer overhead minutes | Minimum rounded minutes |
| --- | ---: | ---: | ---: |
| `walk` | 4.5 | 0 | 2 |
| `motorbike` | 22 | 3 | 4 |
| `car` | 18 | 5 | 5 |
| `taxi` | 18 | 7 | 7 |

Distance is rounded to the nearest 10 meters. Travel time is rounded up to the next whole minute after fixed overhead. These values are approximate and must not be described as road-network routing or live travel time.

## 13. Opening-Hours Handling

Batch 3 may use opening-hours data only when it already exists in the approved runtime POI object.

If verified opening-hours data is absent:

- do not infer hours from category, rating, review text, or external services,
- do not query live providers,
- attach `OPENING_HOURS_UNKNOWN`,
- do not exclude a POI solely because opening hours are unknown.

If future approved runtime data contains opening hours:

- the implementation may mark stops as `open`, `closed`, `unknown`, or `not_applicable`,
- closed stops may be deprioritized or excluded only under explicit tested rules,
- warnings must explain excluded must-include stops.

If approved runtime opening-hours data exists but cannot be parsed, attach `OPENING_HOURS_UNPARSEABLE`.

If approved runtime opening-hours data proves a hard schedule conflict, attach `OPENING_HOURS_CONFLICT` and apply feasibility rules.

## 14. Feasibility Status

Preview and day objects may expose feasibility status.

Allowed values:

- `FEASIBLE`
- `FEASIBLE_WITH_WARNINGS`
- `PARTIAL`
- `INFEASIBLE`

Status criteria are defined in `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md`.

Feasibility status does not replace structural request validation.

## 15. Integration With Batch 2 Recommendations

Batch 3 should call the same candidate-scoring path used by `POST /api/v2/recommendations` rather than creating an unrelated ranking system.

The preview may add itinerary-specific logic after candidate retrieval:

- geographic ordering,
- dwell-time feasibility,
- duration trimming,
- stop uniqueness,
- warning aggregation.

The preview must preserve recommendation transparency:

- carry forward relevant `reasonCodes`,
- add itinerary-specific `reasonCodes`,
- expose public-safe provenance,
- keep deterministic ordering.

## 16. Error Contract

Errors must use the existing v2 error envelope pattern.

Validation error example:

```json
{
  "ok": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed.",
    "details": [
      {
        "field": "trip.durationMinutes",
        "message": "durationMinutes must be an integer between 15 and 480."
      }
    ]
  },
  "meta": {
    "apiVersion": "v2",
    "requestId": "req_20260801_001"
  }
}
```

Unsupported city example:

```json
{
  "ok": false,
  "error": {
    "code": "CITY_NOT_SUPPORTED",
    "message": "Only da-nang is supported in this release.",
    "details": {
      "cityId": "hanoi"
    }
  },
  "meta": {
    "apiVersion": "v2",
    "requestId": "req_20260801_001"
  }
}
```

Narrow hard-constraint failure example:

```json
{
  "ok": false,
  "error": {
    "code": "NO_FEASIBLE_ITINERARY",
    "message": "The requested hard trip constraints cannot produce a feasible itinerary.",
    "details": {
      "cityId": "da-nang",
      "hardConstraintsSatisfied": false
    }
  },
  "meta": {
    "apiVersion": "v2",
    "requestId": "req_20260801_001",
    "cityId": "da-nang"
  }
}
```

`NO_FEASIBLE_ITINERARY` is allowed only when the request is structurally valid, all explicit hard constraints must be satisfied, no compliant schedule can be constructed, and silently relaxing hard constraints is forbidden.

Do not return this error merely because recommendation candidates are limited, opening hours are unknown, optional stops cannot be scheduled, the preview is incomplete, or optional POI coordinates are missing. Those cases normally return HTTP 200 with `PARTIAL` or `INFEASIBLE`, warnings, and unscheduled explanations.

## 17. Warning Taxonomy

The warning taxonomy, stable ordering, trigger rules, scopes, severities, continuation behavior, traveler-safe meanings, and required fixture cases are defined in `docs/rebuild/PHASE2_BATCH3_TRIP_PREVIEW_PLAN.md`.

Public warning severity enum is exactly:

- `info`
- `warning`
- `error`

Stable warning order:

1. `OPENING_HOURS_CONFLICT`
2. `UNSCHEDULED_MUST_INCLUDE`
3. `COORDINATES_MISSING`
4. `OPENING_HOURS_UNPARSEABLE`
5. `DAILY_WINDOW_TIGHT`
6. `INSUFFICIENT_CANDIDATES`
7. `PARTIAL_PREVIEW`
8. `OPENING_HOURS_UNKNOWN`
9. `DURATION_ESTIMATED`
10. `TRAVEL_TIME_ESTIMATED`
11. `ORIGIN_NOT_PROVIDED`
12. `MAX_STOPS_APPLIED`
13. `BUDGET_DATA_UNKNOWN`

Warnings of the same type and scope are ordered by stable scope position and canonical `Global_ID` where applicable.

## 18. Public-Safety Requirements

Responses must not expose:

- environment variables,
- database URLs,
- Firebase credentials,
- source file paths,
- stack traces,
- private ranking weights,
- raw provider payloads,
- unapproved source identifiers.

The endpoint may expose:

- public canonical POI fields,
- public-safe source label,
- reason codes,
- warnings,
- request ID.

## 19. Approval State

Implementation exists on PR #9 review branch and has not merged. Production/main runtime remains at the merged Phase 2 Batch 2 baseline until PR #9 is merged and post-merge validation passes.

Design approval has been granted through:

`APPROVED PHASE 2 BATCH 3`

Design documentation must be merged and post-merge validated before runtime implementation begins.

Approval is limited to the documented Batch 3 scope and boundaries.

## 20. Non-Approval Notice

This contract is not an implementation approval. Batch 3 implementation requires explicit user approval after this design package is reviewed.

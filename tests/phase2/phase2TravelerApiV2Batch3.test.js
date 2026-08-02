const assert = require('node:assert/strict');
const fs = require('node:fs');
const { spawn } = require('node:child_process');
const http = require('node:http');
const path = require('node:path');
const test = require('node:test');

const fixture = require('../fixtures/phase2/tripPreviewQueries.json');
const {
  buildTripPreview,
  normalizeOpeningHours,
  scheduleCandidates,
} = require('../../src/modules/travelerApiV2/tripPreview');
const {
  CATEGORY_DEFAULTS,
  DURATION_POLICY_VERSION,
  policyCategoryForPoi,
  resolveStopDuration,
} = require('../../src/modules/travelerApiV2/tripPreviewDurationPolicy');
const {
  MODE_ASSUMPTIONS,
  TRAVEL_ESTIMATION_METHOD,
  TRAVEL_POLICY_VERSION,
  estimateTravelLeg,
  unknownLeg,
} = require('../../src/modules/travelerApiV2/tripPreviewTravelPolicy');
const {
  WARNING_ORDER,
  uniqueWarnings,
} = require('../../src/modules/travelerApiV2/tripPreviewWarnings');
const { validateTripPreviewRequest } = require('../../src/modules/travelerApiV2/tripPreviewValidation');

const DEFAULT_CITY_ID = 'da-nang';

function requestJson({ port, method = 'GET', path: requestPath, headers = {}, body }) {
  return new Promise((resolve, reject) => {
    const payload = body === undefined ? null : JSON.stringify(body);
    const req = http.request({
      hostname: '127.0.0.1',
      port,
      method,
      path: requestPath,
      headers: {
        ...headers,
        ...(payload ? { 'content-type': 'application/json', 'content-length': Buffer.byteLength(payload) } : {}),
      },
    }, (res) => {
      const chunks = [];
      res.on('data', (chunk) => chunks.push(chunk));
      res.on('end', () => {
        const text = Buffer.concat(chunks).toString('utf8');
        let parsed = null;
        try {
          parsed = text ? JSON.parse(text) : null;
        } catch (_) {
          parsed = null;
        }
        resolve({
          statusCode: res.statusCode,
          body: parsed,
          text,
        });
      });
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

async function waitForTravelerApi(port, child) {
  const startedAt = Date.now();
  let lastError = null;
  while (Date.now() - startedAt < 45000) {
    if (child.exitCode !== null) {
      throw new Error(`Server exited before readiness check completed with code ${child.exitCode}`);
    }
    try {
      const response = await requestJson({ port, path: '/api/v2/cities' });
      if (response.statusCode === 200) return;
      lastError = new Error(`Readiness returned HTTP ${response.statusCode}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 750));
  }
  throw lastError || new Error('Timed out waiting for traveler API readiness');
}

async function stopServer(child) {
  if (!child || child.exitCode !== null) return;
  child.kill();
  const exited = await new Promise((resolve) => {
    const timeout = setTimeout(() => resolve(false), 5000);
    child.once('exit', () => {
      clearTimeout(timeout);
      resolve(true);
    });
  });
  if (exited) return;
  child.kill('SIGKILL');
  await new Promise((resolve) => child.once('exit', resolve));
}

function warningCodeList(trip) {
  return (trip.warnings || []).map((warning) => warning.code);
}

function warningCodeListFromScheduled(scheduled) {
  return (scheduled.warnings || []).map((warning) => warning.code);
}

function stripVariableMeta(response) {
  return {
    ...response.body,
    meta: {
      ...response.body.meta,
      requestId: '<request-id>',
    },
  };
}

function assertNoPublicLeak(value) {
  const text = JSON.stringify(value);
  for (const forbidden of [
    'DATABASE_URL',
    'PostgresPoiRepository',
    'CanonicalCsvPoiRepository',
    'data/canonical',
    'urbanagent_poi_master_v1.csv',
    'poi_entities',
    'sourceIds',
    'placeId',
    'signals',
    'scoreRaw',
    'stack',
    'Bearer ',
    'token',
  ]) {
    assert.equal(text.includes(forbidden), false, `${forbidden} leaked in public response`);
  }
}

function fixtureCasesByMode(mode) {
  return fixture.cases.filter((item) => item.executionMode === mode);
}

test('Phase 2 Batch 3 fixture records the approved deterministic coverage set', () => {
  assert.equal(fixture.fixtureVersion, 'phase2-trip-preview-smoke-v1');
  assert.equal(fixture.contractVersion, 'phase2-batch3-trip-preview-v1');
  assert.equal(fixture.cityId, DEFAULT_CITY_ID);
  assert.equal(
    fixture.datasetSha256,
    '5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae',
  );
  assert.equal(fixture.cases.length, 18);
  assert.equal(fixtureCasesByMode('endpoint').length, 16);
  assert.equal(fixtureCasesByMode('engine_unit').length, 2);
  assert.equal(new Set(fixture.cases.map((item) => item.id)).size, fixture.cases.length);
  assert.ok(fixture.cases.every((item) => ['endpoint', 'engine_unit'].includes(item.executionMode)));
  assert.ok(fixture.notes.some((note) => note.includes('not a travel-quality benchmark')));
  assert.ok(fixture.notes.some((note) => note.includes('executionMode')));
});

test('Phase 2 Batch 3 request validation is deterministic and bounded', () => {
  const invalid = validateTripPreviewRequest({
    cityId: '',
    query: '',
    trip: {
      date: '2026-99-99',
      startTime: '25:00',
      durationMinutes: 14,
      dayCount: 8,
      transport: 'rocket',
      pace: 'rush',
      dailyWindow: { start: '10:00', end: '10:05' },
    },
    startLocation: { lat: 999, lon: 108.2 },
    constraints: {
      maxStopsPerDay: 7,
      mustIncludePoiIds: Array.from({ length: 21 }, (_, index) => `poi_${index}`),
      excludePoiIds: Array.from({ length: 101 }, (_, index) => `poi_${index}`),
    },
    recommendationOptions: { limit: 31 },
  });

  assert.deepEqual(invalid.errors.map((error) => error.field), [
    'cityId',
    'query',
    'trip.date',
    'trip.startTime',
    'trip.durationMinutes',
    'trip.dayCount',
    'trip.transport',
    'trip.pace',
    'trip.dailyWindow',
    'startLocation.lat',
    'constraints.maxStopsPerDay',
    'constraints.mustIncludePoiIds',
    'constraints.excludePoiIds',
    'recommendationOptions.limit',
  ]);

  const valid = validateTripPreviewRequest({
    cityId: DEFAULT_CITY_ID,
    query: ' quan cafe yen tinh ',
    trip: {
      dayCount: '2',
      startTime: '09:00',
      durationMinutes: '180',
      transport: 'walk',
      pace: 'relaxed',
      budget: 'moderate',
    },
    startLocation: { lat: '16.06', lon: '108.22', label: ' hotel ' },
    constraints: {
      mustIncludePoiIds: ['google_maps_34', 'google_maps_34'],
      excludePoiIds: ['google_maps_0', 'google_maps_0'],
      maxStopsPerDay: '3',
      maxDistanceKm: '10',
    },
    recommendationOptions: { limit: '12' },
  });

  assert.equal(valid.value.query, 'quan cafe yen tinh');
  assert.equal(valid.value.trip.dayCount, 2);
  assert.equal(valid.value.trip.transport, 'walk');
  assert.equal(valid.value.constraints.mustIncludePoiIds.length, 1);
  assert.equal(valid.value.constraints.excludePoiIds.length, 1);
  assert.equal(valid.value.recommendationOptions.limit, 12);
});

test('Phase 2 Batch 3 duration and local Haversine policies expose approved versions', () => {
  assert.equal(policyCategoryForPoi({ categoryNormalized: 'cafe' }), 'cafe');
  assert.equal(policyCategoryForPoi({ categoryNormalized: 'restaurant' }), 'restaurant');
  assert.equal(policyCategoryForPoi({ categoryNormalized: 'bakery' }), 'bakery');
  assert.equal(policyCategoryForPoi({ categoryNormalized: 'shopping' }), 'shopping');
  assert.equal(policyCategoryForPoi({ categoryNormalized: 'nightlife' }), 'nightlife');
  assert.equal(policyCategoryForPoi({ categoryNormalized: 'unknown' }), 'fallback');

  const cafeDuration = resolveStopDuration({ poi: { categoryNormalized: 'cafe' } });
  assert.equal(cafeDuration.durationMinutes, CATEGORY_DEFAULTS.cafe);
  assert.equal(cafeDuration.durationSource, 'category_default');
  assert.equal(cafeDuration.durationPolicyVersion, DURATION_POLICY_VERSION);
  assert.deepEqual(cafeDuration.warningCodes, ['DURATION_ESTIMATED']);

  const fallbackDuration = resolveStopDuration({ poi: { categoryNormalized: 'other' } });
  assert.equal(fallbackDuration.durationMinutes, 60);
  assert.equal(fallbackDuration.durationSource, 'fallback');

  for (const [mode, assumption] of Object.entries(MODE_ASSUMPTIONS)) {
    const leg = estimateTravelLeg({
      from: { lat: 16.0678, lon: 108.2208 },
      to: { lat: 16.08, lon: 108.23 },
      transport: mode,
    });
    assert.equal(leg.estimationMethod, TRAVEL_ESTIMATION_METHOD);
    assert.equal(leg.estimationPolicyVersion, TRAVEL_POLICY_VERSION);
    assert.equal(leg.travelMode, mode);
    assert.ok(leg.distanceMeters % 10 === 0);
    assert.ok(leg.travelDurationMinutes >= assumption.minimumMinutes);
  }

  const unknown = unknownLeg({ transport: 'motorbike' });
  assert.equal(unknown.distanceMeters, null);
  assert.equal(unknown.travelDurationMinutes, null);
  assert.equal(unknown.estimationMethod, null);
  assert.equal(unknown.distanceKnown, false);

  const missingCoordinate = estimateTravelLeg({
    from: { lat: null, lon: 108.22 },
    to: { lat: 16.08, lon: 108.23 },
    transport: 'motorbike',
  });
  assert.equal(missingCoordinate.calculationSource, 'missing-coordinates');
  assert.equal(missingCoordinate.distanceKnown, false);
});

test('Phase 2 Batch 3 opening hours and warning taxonomy are conservative', () => {
  assert.deepEqual(normalizeOpeningHours(''), { status: 'unknown', ranges: [] });
  assert.equal(normalizeOpeningHours('06:00 - 21:00').status, 'known');
  assert.equal(normalizeOpeningHours('09:30 - 14:00 | 16:00 - 21:00').ranges.length, 2);
  assert.equal(normalizeOpeningHours('always open maybe').status, 'unparseable');

  const sorted = uniqueWarnings([
    { code: 'BUDGET_DATA_UNKNOWN', scope: 'preview' },
    { code: 'OPENING_HOURS_CONFLICT', scope: 'stop', poiId: 'b' },
    { code: 'OPENING_HOURS_CONFLICT', scope: 'stop', poiId: 'a' },
    { code: 'ORIGIN_NOT_PROVIDED', scope: 'leg' },
  ]);
  assert.deepEqual(sorted.map((warning) => warning.code), [
    'OPENING_HOURS_CONFLICT',
    'OPENING_HOURS_CONFLICT',
    'ORIGIN_NOT_PROVIDED',
    'BUDGET_DATA_UNKNOWN',
  ]);
  assert.ok(sorted.every((warning) => ['info', 'warning', 'error'].includes(warning.severity)));
  assert.equal(WARNING_ORDER.includes('hard'), false);
  assert.equal(WARNING_ORDER.includes('hard/warning'), false);
});

test('Phase 2 Batch 3 trip preview engine satisfies curated fixture invariants', {
  timeout: 120000,
}, async () => {
  const metrics = {
    deterministicReplayPass: 0,
    deterministicReplayTotal: 0,
    exclusionViolations: 0,
    exclusionChecks: 0,
    duplicateStops: 0,
    duplicateChecks: 0,
    scheduledStops: 0,
    satisfiableMustIncludeTotal: 0,
    satisfiableMustIncludeScheduled: 0,
    dailyWindowOverflow: 0,
    dailyWindowChecks: 0,
    knownOpeningHoursConflicts: 0,
    knownOpeningHoursConflictChecks: 0,
    unscheduledItems: 0,
    unscheduledItemsExplained: 0,
    warningAssertions: 0,
    warningAssertionsPassed: 0,
    geographicCompactnessPass: 0,
    geographicCompactnessTotal: 0,
  };

  for (const testCase of fixtureCasesByMode('endpoint')) {
    const validation = validateTripPreviewRequest(testCase.input);
    assert.equal(validation.errors, undefined, `${testCase.id} should be structurally valid`);
    const first = await buildTripPreview(validation.value);
    const second = await buildTripPreview(validation.value);

    if (testCase.expectedError) {
      assert.equal(first.error?.code, testCase.expectedError, testCase.id);
      continue;
    }

    assert.equal(first.error, undefined, testCase.id);
    assert.equal(first.trip.feasibilityStatus, testCase.expectedFeasibility, testCase.id);
    assert.deepEqual(
      first.trip.stops.map((stop) => stop.poi.globalId),
      second.trip.stops.map((stop) => stop.poi.globalId),
      `${testCase.id} stop order must be deterministic`,
    );
    assert.deepEqual(warningCodeList(first.trip), warningCodeList(second.trip), `${testCase.id} warnings must be deterministic`);
    metrics.deterministicReplayPass += 1;
    metrics.deterministicReplayTotal += 1;

    const ids = first.trip.stops.map((stop) => stop.poi.globalId);
    metrics.scheduledStops += ids.length;
    metrics.duplicateChecks += 1;
    metrics.duplicateStops += ids.length - new Set(ids).size;
    for (const excluded of new Set(validation.value.constraints.excludePoiIds)) {
      metrics.exclusionChecks += 1;
      if (ids.includes(excluded)) metrics.exclusionViolations += 1;
    }
    for (const required of validation.value.constraints.mustIncludePoiIds) {
      if (required === 'not-a-real-id') continue;
      metrics.satisfiableMustIncludeTotal += 1;
      if (ids.includes(required)) metrics.satisfiableMustIncludeScheduled += 1;
    }
    metrics.unscheduledItems += first.trip.unscheduled.length;
    metrics.unscheduledItemsExplained += first.trip.unscheduled.filter((item) => item.reasonCode && item.message).length;
    metrics.geographicCompactnessTotal += 1;
    if (
      first.trip.routeSummary.totalDistanceKm === null ||
      (first.trip.routeSummary.totalDistanceKm >= 0 && first.trip.routeSummary.totalDistanceKm <= 80)
    ) {
      metrics.geographicCompactnessPass += 1;
    }

    for (const warningCode of testCase.expectedWarnings || []) {
      metrics.warningAssertions += 1;
      if (warningCodeList(first.trip).includes(warningCode)) metrics.warningAssertionsPassed += 1;
    }
    if (warningCodeList(first.trip).includes('OPENING_HOURS_CONFLICT')) {
      metrics.knownOpeningHoursConflictChecks += 1;
      metrics.knownOpeningHoursConflicts += 1;
    }

    for (const stop of first.trip.stops) {
      assert.equal(stop.durationPolicyVersion, DURATION_POLICY_VERSION);
      assert.equal(['requested', 'category_default', 'fallback'].includes(stop.durationSource), true);
      assert.equal(Object.prototype.hasOwnProperty.call(stop, 'durationPolicyCategory'), false);
      assert.equal(Object.prototype.hasOwnProperty.call(stop, 'recommendation'), false);
      assert.equal(Object.prototype.hasOwnProperty.call(stop.travelFromPrevious, 'legOrder'), false);
      assert.equal(Object.prototype.hasOwnProperty.call(stop.poi, 'placeId'), false);
      assert.equal(Object.prototype.hasOwnProperty.call(stop.poi, 'sourceIds'), false);
      if (stop.arrivalTime && stop.departureTime && first.trip.dailyWindow) {
        metrics.dailyWindowChecks += 1;
        if (stop.arrivalTime < first.trip.dailyWindow.start || stop.departureTime > first.trip.dailyWindow.end) {
          metrics.dailyWindowOverflow += 1;
        }
        assert.ok(stop.arrivalTime >= first.trip.dailyWindow.start, `${testCase.id} arrival overflow`);
        assert.ok(stop.departureTime <= first.trip.dailyWindow.end, `${testCase.id} departure overflow`);
      }
    }
    assertNoPublicLeak(first.trip);
  }

  assert.equal(metrics.deterministicReplayPass, metrics.deterministicReplayTotal);
  assert.equal(metrics.exclusionViolations, 0);
  assert.ok(metrics.exclusionChecks > 0);
  assert.equal(metrics.duplicateStops, 0);
  assert.ok(metrics.duplicateChecks > 0);
  assert.equal(metrics.satisfiableMustIncludeScheduled, metrics.satisfiableMustIncludeTotal);
  assert.ok(metrics.satisfiableMustIncludeTotal > 0);
  assert.equal(metrics.dailyWindowOverflow, 0);
  assert.ok(metrics.dailyWindowChecks > 0);
  assert.equal(metrics.knownOpeningHoursConflicts, metrics.knownOpeningHoursConflictChecks);
  assert.ok(metrics.knownOpeningHoursConflictChecks > 0);
  assert.equal(metrics.unscheduledItemsExplained, metrics.unscheduledItems);
  assert.equal(metrics.warningAssertionsPassed, metrics.warningAssertions);
  assert.equal(metrics.geographicCompactnessPass, metrics.geographicCompactnessTotal);
  assert.ok(metrics.geographicCompactnessTotal > 0);
});

test('Phase 2 Batch 3 engine-unit fixtures execute production scheduling policies', () => {
  const engineCases = fixtureCasesByMode('engine_unit');
  assert.equal(engineCases.length, 2);

  for (const testCase of engineCases) {
    const validation = validateTripPreviewRequest(testCase.input);
    assert.equal(validation.errors, undefined, `${testCase.id} should be structurally valid`);

    const first = scheduleCandidates({
      candidates: testCase.engineUnit.candidates,
      request: validation.value,
      mustIncludeIds: testCase.engineUnit.mustIncludeIds || [],
    });
    const second = scheduleCandidates({
      candidates: [...testCase.engineUnit.candidates].reverse(),
      request: validation.value,
      mustIncludeIds: testCase.engineUnit.mustIncludeIds || [],
    });

    assert.equal(first.feasibilityStatus, testCase.expectedFeasibility, testCase.id);
    assert.deepEqual(
      first.stops.map((stop) => stop.poi.globalId),
      second.stops.map((stop) => stop.poi.globalId),
      `${testCase.id} must be independent of input object iteration order`,
    );
    assert.deepEqual(warningCodeListFromScheduled(first), warningCodeListFromScheduled(second), `${testCase.id} warnings must be deterministic`);

    for (const warningCode of testCase.expectedWarnings || []) {
      assert.ok(warningCodeListFromScheduled(first).includes(warningCode), `${testCase.id} should emit ${warningCode}`);
    }

    const ids = first.stops.map((stop) => stop.poi.globalId);
    assert.equal(new Set(ids).size, ids.length, `${testCase.id} should not duplicate stops`);

    if (testCase.id === 'missing-coordinates-degraded-geography') {
      assert.equal(first.stops.length, 1);
      const stop = first.stops[0];
      assert.equal(stop.poi.location.lat, null);
      assert.equal(stop.poi.location.lon, null);
      assert.equal(stop.poi.location.hasCoordinates, false);
      assert.equal(stop.travelFromPrevious.distanceMeters, null);
      assert.equal(stop.travelFromPrevious.travelDurationMinutes, null);
      assert.equal(stop.travelFromPrevious.calculationSource, 'missing-coordinates');
      assert.equal(stop.travelFromPrevious.distanceKnown, false);
      assert.ok(stop.warnings.includes('COORDINATES_MISSING'));
    }

    if (testCase.id === 'deterministic-tie-case') {
      assert.deepEqual(ids, testCase.engineUnit.expectedStopOrder);
      assert.ok(first.stops.every((stop) => stop.travelFromPrevious.estimationMethod === TRAVEL_ESTIMATION_METHOD));
      assert.equal(JSON.stringify(first), JSON.stringify(second));
    }
  }
});

test('Phase 2 Batch 3 OpenAPI schema matches public runtime trip preview fields', async () => {
  const openApiPath = path.join(__dirname, '..', '..', 'docs', 'rebuild', 'PHASE2_TRAVELER_API_V2_OPENAPI_DRAFT.json');
  const openApi = JSON.parse(fs.readFileSync(openApiPath, 'utf8'));
  const previewCase = fixture.cases.find((item) => item.id === 'one-day-balanced-trip');
  const validation = validateTripPreviewRequest(previewCase.input);
  const result = await buildTripPreview(validation.value);
  const trip = result.trip;
  const stop = trip.stops[0];
  const leg = stop.travelFromPrevious;

  assert.equal(openApi.openapi, '3.1.0');
  assert.ok(openApi.paths['/api/v2/trips/preview']?.post);

  const tripSchemaFields = Object.keys(openApi.components.schemas.TripPreviewTrip.properties).sort();
  assert.deepEqual(Object.keys(trip).sort(), tripSchemaFields);

  const stopSchemaFields = Object.keys(openApi.components.schemas.TripPreviewStop.properties).sort();
  assert.deepEqual(Object.keys(stop).sort(), stopSchemaFields);

  const legSchemaFields = Object.keys(openApi.components.schemas.TripPreviewTravelLeg.properties).sort();
  assert.deepEqual(Object.keys(leg).sort(), legSchemaFields);

  assert.deepEqual(
    openApi.components.schemas.TripPreviewStop.properties.durationSource.enum,
    ['requested', 'category_default', 'fallback'],
  );
  assert.equal(Object.prototype.hasOwnProperty.call(trip, 'recommendationSummary'), false);
  assert.equal(Object.prototype.hasOwnProperty.call(stop, 'durationPolicyCategory'), false);
  assert.equal(Object.prototype.hasOwnProperty.call(stop, 'recommendation'), false);
  assert.equal(Object.prototype.hasOwnProperty.call(leg, 'legOrder'), false);
});

test('Phase 2 Batch 3 engine preserves missing-origin null semantics and budget warnings', {
  timeout: 90000,
}, async () => {
  const missingOriginCase = fixture.cases.find((item) => item.id === 'missing-origin');
  const validation = validateTripPreviewRequest(missingOriginCase.input);
  const result = await buildTripPreview(validation.value);

  assert.equal(result.trip.stops.length > 0, true);
  const firstLeg = result.trip.stops[0].travelFromPrevious;
  assert.equal(firstLeg.distanceMeters, null);
  assert.equal(firstLeg.travelDurationMinutes, null);
  assert.equal(firstLeg.estimationMethod, null);
  assert.equal(firstLeg.distanceKnown, false);
  assert.equal(firstLeg.calculationSource, 'missing-origin');
  assert.ok(warningCodeList(result.trip).includes('ORIGIN_NOT_PROVIDED'));

  const budgetValidation = validateTripPreviewRequest({
    ...missingOriginCase.input,
    trip: {
      ...missingOriginCase.input.trip,
      budget: 'moderate',
    },
  });
  const budgetResult = await buildTripPreview(budgetValidation.value);
  assert.ok(warningCodeList(budgetResult.trip).includes('BUDGET_DATA_UNKNOWN'));
});

test('Phase 2 Batch 3 endpoint returns v2 envelope, request IDs, and deterministic preview data', {
  timeout: 120000,
}, async () => {
  const port = 23000 + Math.floor(Math.random() * 1000);
  const child = spawn(process.execPath, ['src/server.js'], {
    cwd: path.join(__dirname, '..', '..'),
    env: {
      ...process.env,
      PORT: String(port),
      FEATURE_GUEST_ITINERARY_PREVIEW: 'false',
      URBANAGENT_POI_REPOSITORY: '',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const logs = [];
  child.stdout.on('data', (chunk) => logs.push(chunk.toString('utf8')));
  child.stderr.on('data', (chunk) => logs.push(chunk.toString('utf8')));

  try {
    await waitForTravelerApi(port, child);
    const body = fixture.cases.find((item) => item.id === 'one-day-balanced-trip').input;

    const preview = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/preview',
      headers: { 'x-request-id': 'phase2_batch3:preview-1' },
      body,
    });
    assert.equal(preview.statusCode, 200, logs.join('').slice(-1000));
    assert.equal(preview.body.ok, true);
    assert.equal(preview.body.meta.apiVersion, 'v2');
    assert.equal(preview.body.meta.cityId, DEFAULT_CITY_ID);
    assert.equal(preview.body.meta.requestId, 'phase2_batch3:preview-1');
    assert.equal(preview.body.data.trip.tripId, null);
    assert.equal(preview.body.data.trip.preview, true);
    assert.equal(preview.body.data.trip.persisted, false);
    assert.equal(preview.body.data.trip.stops.length > 0, true);
    assert.equal(new Set(preview.body.data.trip.stops.map((stop) => stop.poi.globalId)).size, preview.body.data.trip.stops.length);
    assert.ok(preview.body.data.trip.stops.every((stop) => Array.isArray(stop.reasonCodes) && stop.reasonCodes.length > 0));
    assert.ok(preview.body.data.trip.warnings.every((warning) => ['info', 'warning', 'error'].includes(warning.severity)));
    assertNoPublicLeak(preview.body);

    const repeated = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/preview',
      headers: { 'x-request-id': 'phase2_batch3:preview-2' },
      body,
    });
    assert.equal(repeated.statusCode, 200);
    assert.deepEqual(stripVariableMeta(repeated).data, stripVariableMeta(preview).data);

    const invalidRequestId = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/preview',
      headers: { 'x-request-id': 'invalid/request/id' },
      body,
    });
    assert.equal(invalidRequestId.statusCode, 200);
    assert.notEqual(invalidRequestId.body.meta.requestId, 'invalid/request/id');
    assert.match(invalidRequestId.body.meta.requestId, /^req_[0-9a-f-]{36}$/);
  } finally {
    await stopServer(child);
  }
});

test('Phase 2 Batch 3 endpoint rejects invalid preview requests and no persistence routes exist', {
  timeout: 120000,
}, async () => {
  const port = 24000 + Math.floor(Math.random() * 1000);
  const child = spawn(process.execPath, ['src/server.js'], {
    cwd: path.join(__dirname, '..', '..'),
    env: {
      ...process.env,
      PORT: String(port),
      FEATURE_GUEST_ITINERARY_PREVIEW: 'false',
      URBANAGENT_POI_REPOSITORY: '',
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  const logs = [];
  child.stdout.on('data', (chunk) => logs.push(chunk.toString('utf8')));
  child.stderr.on('data', (chunk) => logs.push(chunk.toString('utf8')));

  try {
    await waitForTravelerApi(port, child);

    const unsupportedCity = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/preview',
      body: { cityId: 'hue', query: 'cafe' },
    });
    assert.equal(unsupportedCity.statusCode, 422);
    assert.equal(unsupportedCity.body.error.code, 'CITY_NOT_SUPPORTED');

    const invalidWindow = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/preview',
      body: {
        cityId: DEFAULT_CITY_ID,
        query: 'cafe',
        trip: {
          dailyWindow: { start: '10:00', end: '10:05' },
        },
      },
    });
    assert.equal(invalidWindow.statusCode, 400);
    assert.equal(invalidWindow.body.error.code, 'VALIDATION_ERROR');
    assert.equal(invalidWindow.body.error.details[0].field, 'trip.dailyWindow');

    const invalidOverlap = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/trips/preview',
      body: fixture.cases.find((item) => item.id === 'impossible-hard-constraints').input,
    });
    assert.equal(invalidOverlap.statusCode, 422);
    assert.equal(invalidOverlap.body.error.code, 'NO_FEASIBLE_ITINERARY');

    const recommendation = await requestJson({
      port,
      method: 'POST',
      path: '/api/v2/recommendations',
      body: {
        cityId: DEFAULT_CITY_ID,
        query: 'quan cafe yen tinh',
        limit: 5,
      },
    });
    assert.equal(recommendation.statusCode, 200, logs.join('').slice(-1000));
    assert.equal(recommendation.body.data.recommendations.length > 0, true);

    for (const forbiddenRoute of [
      { method: 'POST', path: '/api/v2/trips' },
      { method: 'GET', path: '/api/v2/trips/not-a-trip' },
      { method: 'PATCH', path: '/api/v2/trips/not-a-trip' },
      { method: 'DELETE', path: '/api/v2/trips/not-a-trip' },
      { method: 'POST', path: '/api/v2/trips/not-a-trip/replan' },
      { method: 'POST', path: '/api/v2/trips/not-a-trip/stops' },
      { method: 'DELETE', path: '/api/v2/trips/not-a-trip/stops/stop_1' },
      { method: 'POST', path: '/api/v2/feedback' },
    ]) {
      const response = await requestJson({
        port,
        method: forbiddenRoute.method,
        path: forbiddenRoute.path,
        body: {},
      });
      assert.equal(response.statusCode, 404, `${forbiddenRoute.method} ${forbiddenRoute.path} must not exist`);
    }
  } finally {
    await stopServer(child);
  }
});

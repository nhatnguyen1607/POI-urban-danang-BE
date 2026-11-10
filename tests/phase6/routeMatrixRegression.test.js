const assert = require('node:assert/strict');
const test = require('node:test');
const { estimateMatrix } = require('../../src/services/routeMatrixService');

test('matrix accepts the planner walk enum and retains legacy walking alias', () => {
  const input = { origin: { lat: 16, lon: 108 }, destinations: [{ lat: 16.05, lon: 108 }] };
  const walk = estimateMatrix({ ...input, transport: 'walk' })[0];
  const legacy = estimateMatrix({ ...input, transport: 'walking' })[0];
  const motorbike = estimateMatrix({ ...input, transport: 'motorbike' })[0];
  assert.equal(walk.estimatedMinutes, legacy.estimatedMinutes);
  assert.ok(walk.estimatedMinutes > motorbike.estimatedMinutes * 3);
  assert.equal(walk.source, 'local-haversine-estimate');
});

test('matrix preserves zero longitude and accepts lng alias', () => {
  const result = estimateMatrix({
    origin: { lat: 0, lon: 0, lng: 100 },
    destinations: [{ lat: 0.05, lon: 0, lng: 100 }], transport: 'walk',
  })[0];
  const alias = estimateMatrix({
    origin: { lat: 0, lng: 0 }, destinations: [{ lat: 0.05, lng: 0 }], transport: 'walk',
  })[0];
  assert.ok(Number.isFinite(result.distanceKm));
  assert.ok(Number.isFinite(result.estimatedMinutes));
  assert.equal(result.distanceKm, alias.distanceKm);
  assert.equal(result.estimatedMinutes, alias.estimatedMinutes);
});

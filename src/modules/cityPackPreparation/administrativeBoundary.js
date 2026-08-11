const fs = require('node:fs');

const BOUNDARY_CLASSIFICATIONS = Object.freeze({
  INSIDE_DANANG: 'INSIDE_DANANG',
  BOUNDARY_EDGE: 'BOUNDARY_EDGE',
  OUTSIDE_DANANG: 'OUTSIDE_DANANG',
  INVALID_COORDINATE: 'INVALID_COORDINATE',
});

const DEFAULT_EDGE_EPSILON = 1e-10;

function validCoordinate(latitude, longitude) {
  return Number.isFinite(latitude)
    && Number.isFinite(longitude)
    && latitude >= -90
    && latitude <= 90
    && longitude >= -180
    && longitude <= 180;
}

function pointOnSegment(point, start, end, epsilon = DEFAULT_EDGE_EPSILON) {
  const [x, y] = point;
  const [x1, y1] = start;
  const [x2, y2] = end;
  const cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1);
  if (Math.abs(cross) > epsilon) return false;
  return x >= Math.min(x1, x2) - epsilon
    && x <= Math.max(x1, x2) + epsilon
    && y >= Math.min(y1, y2) - epsilon
    && y <= Math.max(y1, y2) + epsilon;
}

function classifyAgainstRing(point, ring, epsilon = DEFAULT_EDGE_EPSILON) {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index, index += 1) {
    const currentPoint = ring[index];
    const previousPoint = ring[previous];
    if (pointOnSegment(point, previousPoint, currentPoint, epsilon)) return 'EDGE';
    const crosses = ((currentPoint[1] > point[1]) !== (previousPoint[1] > point[1]))
      && point[0] < (
        ((previousPoint[0] - currentPoint[0]) * (point[1] - currentPoint[1]))
        / (previousPoint[1] - currentPoint[1])
      ) + currentPoint[0];
    if (crosses) inside = !inside;
  }
  return inside ? 'INSIDE' : 'OUTSIDE';
}

function classifyAgainstPolygon(point, polygon, epsilon) {
  const outer = classifyAgainstRing(point, polygon[0], epsilon);
  if (outer === 'EDGE') return 'EDGE';
  if (outer === 'OUTSIDE') return 'OUTSIDE';
  for (const hole of polygon.slice(1)) {
    const holeResult = classifyAgainstRing(point, hole, epsilon);
    if (holeResult === 'EDGE') return 'EDGE';
    if (holeResult === 'INSIDE') return 'OUTSIDE';
  }
  return 'INSIDE';
}

function geometryPolygons(geometry) {
  if (!geometry || !Array.isArray(geometry.coordinates)) {
    throw new Error('Boundary GeoJSON has no coordinates.');
  }
  if (geometry.type === 'Polygon') return [geometry.coordinates];
  if (geometry.type === 'MultiPolygon') return geometry.coordinates;
  throw new Error(`Unsupported boundary geometry: ${geometry.type || 'unknown'}.`);
}

function classifyAdministrativeBoundary(record, boundary, options = {}) {
  if (
    record?.latitude === null
    || record?.latitude === undefined
    || record?.latitude === ''
    || record?.longitude === null
    || record?.longitude === undefined
    || record?.longitude === ''
  ) {
    return BOUNDARY_CLASSIFICATIONS.INVALID_COORDINATE;
  }
  const latitude = Number(record?.latitude);
  const longitude = Number(record?.longitude);
  if (!validCoordinate(latitude, longitude)) {
    return BOUNDARY_CLASSIFICATIONS.INVALID_COORDINATE;
  }
  const geometry = boundary?.type === 'Feature' ? boundary.geometry : boundary;
  const point = [longitude, latitude];
  let inside = false;
  for (const polygon of geometryPolygons(geometry)) {
    const result = classifyAgainstPolygon(point, polygon, options.edgeEpsilon);
    if (result === 'EDGE') return BOUNDARY_CLASSIFICATIONS.BOUNDARY_EDGE;
    if (result === 'INSIDE') inside = true;
  }
  return inside
    ? BOUNDARY_CLASSIFICATIONS.INSIDE_DANANG
    : BOUNDARY_CLASSIFICATIONS.OUTSIDE_DANANG;
}

function loadBoundaryArtifact(filePath) {
  const artifact = JSON.parse(fs.readFileSync(filePath, 'utf8'));
  geometryPolygons(artifact.type === 'Feature' ? artifact.geometry : artifact);
  return artifact;
}

function summarizeBoundaryClassifications(records, boundary, options = {}) {
  const counts = Object.fromEntries(Object.values(BOUNDARY_CLASSIFICATIONS).map((value) => [value, 0]));
  const classified = records.map((record) => {
    const boundaryClassification = classifyAdministrativeBoundary(record, boundary, options);
    counts[boundaryClassification] += 1;
    return { ...record, boundaryClassification };
  });
  return { classified, counts };
}

module.exports = {
  BOUNDARY_CLASSIFICATIONS,
  DEFAULT_EDGE_EPSILON,
  classifyAdministrativeBoundary,
  classifyAgainstRing,
  loadBoundaryArtifact,
  pointOnSegment,
  summarizeBoundaryClassifications,
  validCoordinate,
};

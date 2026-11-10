const ReteNetwork = require('./rete_algorithm');
const InferenceEngine = require('./backward_chaining');
const FuzzyLogic = require('./fuzzy_logic');
const POIDensityEngine = require('./poi_density_engine');
const { fetchWeatherData } = require('./weather_service');
const { osrmGuard } = require('../infrastructure/external/providerGuards');

// ============================================================================
//  HỆ CHUYÊN GIA TỔNG HỢP (Expert System Facade)
//  Tích hợp:
//  - Backward Chaining (Suy diễn lùi)
//  - Rete Algorithm (Mạng lưới lưu trữ nhanh)
//  - Fuzzy Logic đa biến (Lập luận mờ: Giờ + POI + Thời tiết)
//  - POI Density Engine (Suy luận mật độ từ dữ liệu)
// ============================================================================

// Singleton
const reteNetwork = new ReteNetwork();
const poiEngine = new POIDensityEngine();
let engine = null;

/**
 * Initialize the Expert System (call once at server startup)
 */
async function initExpertSystem() {
  await Promise.all([
    reteNetwork.load(),
    poiEngine.load(),
  ]);
  engine = new InferenceEngine(reteNetwork, poiEngine);
  return { reteNetwork, engine, FuzzyLogic, poiEngine };
}

function routeError(code, message, cause) {
  const error = new Error(message, cause ? { cause } : undefined);
  error.code = code;
  // Keep legacy /api/route HTTP/error shape; category is carried in details.
  return error;
}

function validateRouteCoordinates(originLat, originLng, destLat, destLng) {
  if (![originLat, originLng, destLat, destLng].every(value => typeof value === 'number' && Number.isFinite(value))
    || Math.abs(originLat) > 90 || Math.abs(destLat) > 90
    || Math.abs(originLng) > 180 || Math.abs(destLng) > 180) {
    throw routeError('INVALID_ROUTE_COORDINATES', 'Invalid origin or destination coordinates');
  }
}

/**
 * Fetch real driving route from OSRM public API
 */
async function fetchOSRMRoute(originLat, originLng, destLat, destLng) {
  // Use alternatives=3 to get up to 3 alternative routes
  const url = `https://router.project-osrm.org/route/v1/driving/${originLng},${originLat};${destLng},${destLat}?overview=full&geometries=geojson&steps=true&annotations=true&alternatives=3`;

  validateRouteCoordinates(originLat, originLng, destLat, destLng);
  // Cache/coalesce parsed values, never a single-use Fetch Response.
  return osrmGuard.execute(
    `${originLat},${originLng}:${destLat},${destLng}`,
    async ({ signal }) => {
      let response;
      let raw;
      try {
        response = await fetch(url, { signal });
        raw = await response.text();
      } catch (cause) {
        throw routeError('OSRM_NETWORK', `OSRM network failure: ${cause.message}`, cause);
      }
      let data;
      try {
        data = JSON.parse(raw);
      } catch (cause) {
        if (response.ok) throw routeError('OSRM_PARSE', 'OSRM returned invalid JSON', cause);
      }
      if (!response.ok) {
        const error = routeError('OSRM_HTTP', `OSRM HTTP ${response.status}${data?.code ? ` (${data.code})` : ''}${data?.message ? `: ${data.message}` : ''}`);
        error.upstreamStatus = response.status;
        throw error;
      }
      if (data?.code !== 'Ok') {
        throw routeError(data?.code === 'NoRoute' ? 'OSRM_NO_ROUTE' : 'OSRM_SERVICE',
          `OSRM ${data?.code || 'Unknown'}${data?.message ? `: ${data.message}` : ''}`);
      }
      if (!Array.isArray(data.routes) || data.routes.length === 0) {
        throw routeError('OSRM_NO_ROUTE', 'OSRM NoRoute: no routes returned');
      }
      if (data.routes.some(route => !route || !Number.isFinite(route.distance) || route.distance < 0
        || !Number.isFinite(route.duration) || route.duration < 0
        || route.geometry?.type !== 'LineString' || !Array.isArray(route.geometry.coordinates)
        || route.geometry.coordinates.length < 2
        || route.geometry.coordinates.some(point => !Array.isArray(point) || point.length < 2
          || !Number.isFinite(point[0]) || Math.abs(point[0]) > 180
          || !Number.isFinite(point[1]) || Math.abs(point[1]) > 90)
        || !Array.isArray(route.legs?.[0]?.steps))) {
        throw routeError('OSRM_PARSE', 'OSRM returned an invalid route payload');
      }
      return data.routes.map(route => ({
        geometry: route.geometry,
        distance: route.distance,
        duration: route.duration,
        steps: route.legs[0].steps,
      }));
    },
  );
}

/**
 * Find and validate route from origin to destination
 * @param {number} originLat
 * @param {number} originLng
 * @param {number} destLat
 * @param {number} destLng
 * @returns {Object} Route + ES validation results
 */
async function findOptimalRoute(originLat, originLng, destLat, destLng) {
  validateRouteCoordinates(originLat, originLng, destLat, destLng);
  if (!reteNetwork.loaded) {
    await reteNetwork.load();
  }

  const weather = await fetchWeatherData(originLat, originLng); // Lấy thời tiết tại điểm đi

  // Step 1: Fetch multiple real routes from OSRM
  const routesData = await fetchOSRMRoute(originLat, originLng, destLat, destLng);

  // Step 2 & 3: Run backward chaining inference engine on all alternatives and build response
  const processedRoutes = routesData.map(routeData => {
    const esResult = engine.validateRoute(routeData, weather);
    const steps = routeData.steps
      .filter(s => s.name && s.distance > 0)
      .map(s => ({
        instruction: _buildVietnameseInstruction(s),
        road: s.name,
        distance: s.distance,
        duration: s.duration,
        maneuver: s.maneuver?.type || '',
        modifier: s.maneuver?.modifier || '',
        location: s.maneuver?.location || [],
      }));

    return {
      route: routeData.geometry,
      distance: routeData.distance,
      duration: routeData.duration,
      steps,
      esValidation: esResult,
    };
  });

  // Có thể sắp xếp để tuyến đường nào valid nằm lên đầu tiên
  processedRoutes.sort((a, b) => {
    if (a.esValidation.valid && !b.esValidation.valid) return -1;
    if (!a.esValidation.valid && b.esValidation.valid) return 1;
    return a.esValidation.warnings.length - b.esValidation.warnings.length;
  });

  return { routes: processedRoutes };
}

/**
 * Build Vietnamese instruction from OSRM step
 */
function _buildVietnameseInstruction(step) {
  const type = step.maneuver?.type || '';
  const modifier = step.maneuver?.modifier || '';
  const road = step.name || 'đường không tên';
  const distKm = (step.distance / 1000).toFixed(1);

  const modMap = {
    'left': 'rẽ trái',
    'right': 'rẽ phải',
    'slight left': 'rẽ nhẹ sang trái',
    'slight right': 'rẽ nhẹ sang phải',
    'sharp left': 'rẽ gấp sang trái',
    'sharp right': 'rẽ gấp sang phải',
    'straight': 'đi thẳng',
    'uturn': 'quay đầu',
  };

  const typeMap = {
    'depart': `Xuất phát trên ${road}`,
    'arrive': `Đến nơi - ${road}`,
    'turn': `${modMap[modifier] || modifier} vào ${road}`,
    'new name': `Tiếp tục trên ${road}`,
    'merge': `Nhập vào ${road}`,
    'fork': `${modMap[modifier] || 'rẽ'} vào ${road}`,
    'roundabout': `Đi qua vòng xuyến, ra ${road}`,
    'end of road': `Cuối đường, ${modMap[modifier] || ''} vào ${road}`,
    'continue': `Tiếp tục trên ${road}`,
  };

  const instruction = typeMap[type] || `${modMap[modifier] || type} trên ${road}`;
  return `${instruction} (${distKm} km)`;
}

module.exports = {
  initExpertSystem,
  findOptimalRoute,
  ReteNetwork,
  InferenceEngine,
  FuzzyLogic,
  POIDensityEngine,
  fetchOSRMRoute,
  validateRouteCoordinates,
};

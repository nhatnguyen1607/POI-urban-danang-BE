const express = require('express');
const { getPoiDataQualityReport, loadPOIsForEdaSource } = require('../../services/poiDataService');
const { getCityConfig, listCityConfigs } = require('../cities/cityConfig');
const { sendError, sendSuccess, travelerApiV2Context } = require('./requestContext');
const { searchTravelerPois } = require('./poiSearch');
const { getTravelerRecommendations, validateRecommendationRequest } = require('./recommendations');
const { serializeCity, serializeCityStatus, serializePoi } = require('./serializers');
const { buildTripPreview } = require('./tripPreview');
const { validateTripPreviewRequest } = require('./tripPreviewValidation');

const router = express.Router();

function requireSupportedCity(req, res) {
  const cityId = req.params.cityId || req.query.cityId || req.body?.cityId;
  if (!cityId) {
    sendError(req, res, 400, 'VALIDATION_ERROR', 'cityId is required', {
      details: [{ field: 'cityId', rule: 'required' }],
    });
    return null;
  }
  const city = getCityConfig(cityId);
  if (!city) {
    sendError(req, res, 422, 'CITY_NOT_SUPPORTED', 'City is not supported in Phase 2.', {
      details: [{ field: 'cityId', value: cityId }],
    });
    return null;
  }
  return city;
}

router.use(travelerApiV2Context);

router.get('/cities', (req, res) => {
  sendSuccess(req, res, {
    cities: listCityConfigs().map(serializeCity),
  });
});

router.get('/cities/:cityId/status', async (req, res) => {
  try {
    const city = requireSupportedCity(req, res);
    if (!city) return;
    const quality = await getPoiDataQualityReport();
    sendSuccess(req, res, serializeCityStatus(city, quality), { cityId: city.cityId });
  } catch (error) {
    sendError(req, res, 500, 'INTERNAL_ERROR', 'Failed to read city status');
  }
});

router.get('/pois/search', async (req, res) => {
  try {
    const city = requireSupportedCity(req, res);
    if (!city) return;
    const { pois } = await loadPOIsForEdaSource({
      cityId: city.cityId,
      source: req.query.source,
    });
    const result = searchTravelerPois(pois, req.query);
    if (result.error) {
      sendError(req, res, 400, 'VALIDATION_ERROR', 'Invalid pagination parameter', {
        cityId: city.cityId,
        details: [result.error],
      });
      return;
    }
    sendSuccess(req, res, result, { cityId: city.cityId });
  } catch (error) {
    sendError(req, res, 500, 'INTERNAL_ERROR', 'Failed to search POIs');
  }
});

router.get('/pois/:poiId', async (req, res) => {
  try {
    const city = requireSupportedCity(req, res);
    if (!city) return;
    const { pois } = await loadPOIsForEdaSource({
      cityId: city.cityId,
      source: 'all',
    });
    const poi = pois.find((item) => item.globalId === req.params.poiId || item.id === req.params.poiId);
    if (!poi) {
      sendError(req, res, 404, 'NOT_FOUND', 'POI not found.', { cityId: city.cityId });
      return;
    }
    sendSuccess(req, res, { poi: serializePoi(poi) }, { cityId: city.cityId });
  } catch (error) {
    sendError(req, res, 500, 'INTERNAL_ERROR', 'Failed to read POI detail');
  }
});

router.post('/recommendations', async (req, res) => {
  try {
    const validation = validateRecommendationRequest(req.body);
    if (validation.error) {
      sendError(req, res, 400, 'VALIDATION_ERROR', 'Invalid recommendation request', {
        details: [validation.error],
      });
      return;
    }

    const city = requireSupportedCity(req, res);
    if (!city) return;

    const result = await getTravelerRecommendations({
      query: validation.query,
      context: validation.context,
      limit: validation.limit,
      cityId: city.cityId,
    });
    sendSuccess(req, res, result, { cityId: city.cityId });
  } catch (error) {
    sendError(req, res, 500, 'INTERNAL_ERROR', 'Failed to recommend POIs');
  }
});

router.post('/trips/preview', async (req, res) => {
  try {
    const validation = validateTripPreviewRequest(req.body);
    if (validation.errors) {
      sendError(req, res, 400, 'VALIDATION_ERROR', 'Invalid trip preview request', {
        details: validation.errors,
      });
      return;
    }

    const city = getCityConfig(validation.value.cityId);
    if (!city) {
      sendError(req, res, 422, 'CITY_NOT_SUPPORTED', 'Only da-nang is supported in this release.', {
        details: { cityId: validation.value.cityId },
      });
      return;
    }

    const result = await buildTripPreview(validation.value);
    if (result.error) {
      sendError(req, res, result.error.status, result.error.code, result.error.message, {
        cityId: validation.value.cityId,
        details: result.error.details,
      });
      return;
    }

    sendSuccess(req, res, { trip: result.trip }, { cityId: city.cityId });
  } catch (error) {
    sendError(req, res, 500, 'INTERNAL_ERROR', 'Failed to create trip preview');
  }
});

module.exports = {
  travelerApiV2Router: router,
  requireSupportedCity,
};

const { recommendPOIs } = require('../../services/poiRetrievalService');
const { normalizeText } = require('../../services/poiDataService');
const {
  DEFAULT_RECOMMENDATION_LIMIT,
  MAX_RECOMMENDATION_LIMIT,
} = require('./constants');
const { serializePoi } = require('./serializers');

function parseRecommendationLimit(value) {
  if (value === undefined || value === null || value === '') {
    return { limit: DEFAULT_RECOMMENDATION_LIMIT };
  }
  const limit = Number(value);
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_RECOMMENDATION_LIMIT) {
    return {
      error: {
        field: 'limit',
        rule: `integer_between_1_and_${MAX_RECOMMENDATION_LIMIT}`,
      },
    };
  }
  return { limit };
}

function validateLocation(location) {
  if (location === undefined || location === null) return null;
  if (typeof location !== 'object' || Array.isArray(location)) {
    return {
      field: 'context.location',
      rule: 'object_with_valid_lat_lon',
    };
  }
  const lat = Number(location.lat);
  const lon = Number(location.lon ?? location.lng);
  if (!Number.isFinite(lat) || lat < -90 || lat > 90) {
    return {
      field: 'context.location.lat',
      rule: 'number_between_-90_and_90',
    };
  }
  if (!Number.isFinite(lon) || lon < -180 || lon > 180) {
    return {
      field: 'context.location.lon',
      rule: 'number_between_-180_and_180',
    };
  }
  return null;
}

function validateRecommendationRequest(body = {}) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) {
    return {
      error: {
        field: 'body',
        rule: 'object',
      },
    };
  }

  const cityId = String(body.cityId || '').trim();
  if (!cityId) {
    return {
      error: {
        field: 'cityId',
        rule: 'required',
      },
    };
  }

  const query = String(body.query || '').trim();
  if (!query) {
    return {
      error: {
        field: 'query',
        rule: 'non_empty_string',
      },
    };
  }

  if (body.context !== undefined && (typeof body.context !== 'object' || body.context === null || Array.isArray(body.context))) {
    return {
      error: {
        field: 'context',
        rule: 'object',
      },
    };
  }

  const locationError = validateLocation(body.context?.location);
  if (locationError) return { error: locationError };

  const limitResult = parseRecommendationLimit(body.limit);
  if (limitResult.error) return { error: limitResult.error };

  return {
    cityId,
    query,
    context: body.context || {},
    limit: limitResult.limit,
  };
}

function recommendationName(item) {
  return normalizeText(item?.poi?.name || item?.title || item?.name || '');
}

function recommendationId(item) {
  return String(item?.poi?.globalId || item?.poi?.id || item?.globalId || item?.id || '');
}

function recommendationScore(item) {
  const score = Number(item?.score);
  return Number.isFinite(score) ? score : 0;
}

function rankRecommendationItems(items = []) {
  return [...items].sort((a, b) => (
    recommendationScore(b) - recommendationScore(a) ||
    recommendationName(a).localeCompare(recommendationName(b), 'en') ||
    recommendationId(a).localeCompare(recommendationId(b), 'en')
  ));
}

function buildReasonCodes(recommendation) {
  const signals = recommendation.signals || {};
  const codes = [];
  if (recommendation.intent || signals.category >= 0.3) codes.push('intent_match');
  if (signals.category >= 0.3) codes.push('category_match');
  if (signals.semantic > 0 || signals.modelSemantic > 0) codes.push('query_text_match');
  if (signals.preference >= 0.45) codes.push('preference_match');
  if (signals.rating >= 0.75) codes.push('rating_signal');
  if (signals.review > 0) codes.push('review_signal');
  if (signals.distance >= 0.7) codes.push('distance_fit');
  if (signals.distanceKm === null || signals.distanceKm === undefined) codes.push('distance_unknown');
  if (!codes.length) codes.push('canonical_dataset_match');
  return [...new Set(codes)];
}

function publicPoiFromRecommendation(recommendation) {
  return {
    id: recommendation.globalId || recommendation.id,
    globalId: recommendation.globalId || recommendation.id,
    cityId: recommendation.cityId,
    name: recommendation.name || recommendation.title,
    category: recommendation.category || null,
    categoryNormalized: normalizeText(recommendation.category || '') || null,
    lat: recommendation.lat,
    lon: recommendation.lon,
    hasCoordinates: Boolean(recommendation.hasCoordinates),
    coordinateStatus: recommendation.coordinateStatus,
    addressRaw: recommendation.address || null,
    district: recommendation.district || null,
    rating: recommendation.rating ?? null,
    reviewCount: recommendation.reviewCount ?? null,
    imageUrls: recommendation.imageUrls || [],
    imageUrl: recommendation.imageUrl || null,
    source: recommendation.source || null,
    sourceId: recommendation.sourceId || null,
    sourceIds: recommendation.sourceIds || [],
    aliasGlobalIds: recommendation.aliasGlobalIds || [],
    mergeStatus: recommendation.mergeStatus || null,
    dataQualityFlags: recommendation.dataQualityFlags || [],
  };
}

function serializeRecommendationItem(recommendation) {
  const poi = serializePoi(publicPoiFromRecommendation(recommendation));
  return {
    poi,
    score: recommendationScore(recommendation),
    reason: recommendation.reason || 'Recommended from canonical traveler POI signals.',
    reasonCodes: buildReasonCodes(recommendation),
    warnings: Array.isArray(recommendation.warnings) ? recommendation.warnings : [],
    provenance: poi.provenance,
  };
}

async function getTravelerRecommendations({ query, context, limit, cityId }) {
  const recommendation = await recommendPOIs({
    query,
    context: {
      ...context,
      cityId,
    },
    limit: MAX_RECOMMENDATION_LIMIT,
  });
  const ranked = rankRecommendationItems(recommendation.results || []).slice(0, limit);
  return {
    recommendations: ranked.map(serializeRecommendationItem),
    query: recommendation.query || query,
    cityId,
    limit,
    warnings: Array.isArray(recommendation.warnings) ? recommendation.warnings : [],
  };
}

module.exports = {
  buildReasonCodes,
  getTravelerRecommendations,
  parseRecommendationLimit,
  rankRecommendationItems,
  serializeRecommendationItem,
  validateRecommendationRequest,
};

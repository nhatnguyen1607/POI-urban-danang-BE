const {
  CAPABILITY_STATES,
  CONTRACT_VERSION,
  DATASET_VERSION,
  PHASE2_CITY_STATUS,
} = require('./constants');

function knownStatus(value) {
  return value === null || value === undefined || value === '' ? 'unknown' : 'known';
}

function nullIfUnknown(value) {
  return value === '' || value === undefined ? null : value;
}

function cityDisplayName(city) {
  return city.displayNameEn || city.displayNameVi || city.cityId;
}

function buildCapabilityStatus() {
  return {
    cities: CAPABILITY_STATES.EXPERIMENTAL,
    cityStatus: CAPABILITY_STATES.EXPERIMENTAL,
    poiSearch: CAPABILITY_STATES.EXPERIMENTAL,
    poiDetail: CAPABILITY_STATES.EXPERIMENTAL,
    recommendations: CAPABILITY_STATES.EXPERIMENTAL,
    tripPreview: CAPABILITY_STATES.EXPERIMENTAL,
    tripSave: CAPABILITY_STATES.UNAVAILABLE,
    tripEdit: CAPABILITY_STATES.UNAVAILABLE,
    tripReplan: CAPABILITY_STATES.UNAVAILABLE,
    feedbackPersistence: CAPABILITY_STATES.UNAVAILABLE,
    liveOpeningHours: CAPABILITY_STATES.UNAVAILABLE,
    liveBooking: CAPABILITY_STATES.UNAVAILABLE,
    roadNetworkRouting: CAPABILITY_STATES.UNAVAILABLE,
  };
}

function serializeCity(city) {
  return {
    cityId: city.cityId,
    displayName: cityDisplayName(city),
    countryCode: city.countryCode,
    timezone: city.timezone,
    currency: city.currency,
    status: PHASE2_CITY_STATUS,
    capabilityStatus: buildCapabilityStatus(),
  };
}

function qualityStatus(count) {
  if (!Number.isFinite(Number(count))) return 'unknown';
  return count > 0 ? 'known_gap' : 'known';
}

function buildQualitySummary(quality) {
  const totals = quality?.totals || {};
  return {
    status: 'known_gaps',
    missingAddress: {
      count: totals.missingAddress ?? null,
      status: qualityStatus(totals.missingAddress),
    },
    missingRating: {
      count: totals.missingRating ?? null,
      status: qualityStatus(totals.missingRating),
    },
    missingReviewCount: {
      count: totals.missingReviewCount ?? null,
      status: qualityStatus(totals.missingReviewCount),
    },
    openingHours: {
      status: 'unknown',
    },
    freshness: {
      status: 'unknown',
    },
    adminBoundary: {
      status: 'pending_spatial_join',
    },
  };
}

function serializeCityStatus(city, quality) {
  return {
    city: serializeCity(city),
    dataset: {
      datasetVersion: DATASET_VERSION,
      contractVersion: CONTRACT_VERSION,
      applicationPoiCount: quality?.totals?.applicationPois ?? null,
    },
    qualitySummary: buildQualitySummary(quality),
    capabilityStatus: buildCapabilityStatus(),
  };
}

function sourceIdentifierFromValue(value, fallbackSource) {
  const text = String(value || '').trim();
  if (!text) return null;
  const separatorIndex = text.indexOf(':');
  if (separatorIndex > 0) {
    return {
      namespace: text.slice(0, separatorIndex),
      value: text.slice(separatorIndex + 1),
      source: fallbackSource || text.slice(0, separatorIndex),
    };
  }
  return {
    namespace: 'restaurant_id',
    value: text,
    source: fallbackSource || null,
  };
}

function buildSourceIdentifiers(poi) {
  const identifiers = [];
  const seen = new Set();
  for (const sourceId of poi.sourceIds || []) {
    const identifier = sourceIdentifierFromValue(sourceId, poi.source);
    if (!identifier) continue;
    const key = `${identifier.namespace}:${identifier.value}:${identifier.source || ''}`;
    if (!seen.has(key)) {
      identifiers.push(identifier);
      seen.add(key);
    }
  }
  if (poi.sourceId) {
    const identifier = sourceIdentifierFromValue(poi.sourceId, poi.source);
    const key = `${identifier.namespace}:${identifier.value}:${identifier.source || ''}`;
    if (!seen.has(key)) identifiers.push(identifier);
  }
  return identifiers;
}

function buildWarnings(poi) {
  const warnings = [];
  if (!poi.addressCurrent && !poi.addressRaw) warnings.push('address_unknown');
  if (!poi.openingHoursRaw) warnings.push('opening_hours_unknown');
  warnings.push('freshness_unknown');
  if (poi.rating === null || poi.rating === undefined) warnings.push('rating_unknown');
  if (poi.reviewCount === null || poi.reviewCount === undefined) warnings.push('review_count_unknown');
  return warnings;
}

function serializeRating(poi) {
  return {
    normalized: {
      value: poi.rating ?? null,
      scale: 5,
      status: knownStatus(poi.rating),
    },
    google: {
      value: poi.googleRating ?? null,
      scale: 5,
      ratingCount: poi.googleRatingCount ?? null,
      ratingCountStatus: knownStatus(poi.googleRatingCount),
    },
    foody: {
      value: poi.foodyRating10 ?? null,
      scale: 10,
      sampleReviewCount: poi.foodyReviewSampleCount ?? null,
      sampleReviewCountStatus: knownStatus(poi.foodyReviewSampleCount),
    },
    reviewCount: {
      value: poi.reviewCount ?? null,
      status: knownStatus(poi.reviewCount),
    },
  };
}

function serializePoi(poi) {
  return {
    id: poi.globalId || poi.id,
    globalId: poi.globalId || poi.id,
    cityId: poi.cityId,
    name: poi.name,
    category: poi.category,
    categoryNormalized: poi.categoryNormalized,
    location: {
      lat: poi.lat,
      lon: poi.lon,
      hasCoordinates: Boolean(poi.hasCoordinates),
      coordinateStatus: poi.coordinateStatus,
    },
    address: {
      current: nullIfUnknown(poi.addressCurrent),
      raw: nullIfUnknown(poi.addressRaw),
      district: nullIfUnknown(poi.district),
      adminNormalizationStatus: nullIfUnknown(poi.adminNormalizationStatus),
    },
    rating: serializeRating(poi),
    images: {
      imageUrls: poi.imageUrls || [],
      imageUrl: poi.imageUrl || null,
    },
    provenance: {
      source: poi.source,
      sourceIdentifiers: buildSourceIdentifiers(poi),
      aliasGlobalIds: poi.aliasGlobalIds || [],
      mergeStatus: poi.mergeStatus || null,
      dataQualityFlags: poi.dataQualityFlags || [],
    },
    freshness: {
      observedAt: null,
      lastVerifiedAt: null,
      status: 'unknown',
    },
    warnings: buildWarnings(poi),
  };
}

module.exports = {
  buildCapabilityStatus,
  buildQualitySummary,
  buildSourceIdentifiers,
  serializeCity,
  serializeCityStatus,
  serializePoi,
  serializeRating,
};

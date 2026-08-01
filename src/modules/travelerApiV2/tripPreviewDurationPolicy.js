const { normalizeText } = require('../../services/poiDataService');

const DURATION_POLICY_VERSION = 'phase2-batch3-duration-v1';

const CATEGORY_DEFAULTS = Object.freeze({
  cafe: 60,
  coffee: 60,
  tea: 60,
  restaurant: 75,
  food: 75,
  seafood: 75,
  local_food: 75,
  dessert: 45,
  bakery: 45,
  snack: 45,
  attraction: 75,
  landmark: 75,
  viewpoint: 75,
  museum: 90,
  cultural: 90,
  historic: 90,
  gallery: 90,
  beach: 90,
  park: 90,
  nature: 90,
  riverside: 90,
  shopping: 75,
  market: 75,
  mall: 75,
  nightlife: 90,
  bar: 90,
  pub: 90,
  spa: 90,
  wellness: 90,
  activity: 90,
  entertainment: 90,
  hotel: 30,
  lodging: 30,
});

const GLOBAL_FALLBACK_MINUTES = 60;

function policyCategoryForPoi(poi = {}) {
  const haystack = normalizeText([
    poi.categoryNormalized,
    poi.category,
    poi.name,
  ].filter(Boolean).join(' '));

  for (const category of Object.keys(CATEGORY_DEFAULTS)) {
    if (haystack.includes(normalizeText(category))) return category;
  }
  return 'fallback';
}

function resolveStopDuration({ poi, requestedDurationMinutes } = {}) {
  if (Number.isInteger(requestedDurationMinutes) && requestedDurationMinutes >= 15 && requestedDurationMinutes <= 480) {
    return {
      durationMinutes: requestedDurationMinutes,
      durationSource: 'requested',
      durationPolicyVersion: DURATION_POLICY_VERSION,
      policyCategory: 'requested',
      warningCodes: [],
    };
  }

  const policyCategory = policyCategoryForPoi(poi);
  if (policyCategory !== 'fallback') {
    return {
      durationMinutes: CATEGORY_DEFAULTS[policyCategory],
      durationSource: 'category_default',
      durationPolicyVersion: DURATION_POLICY_VERSION,
      policyCategory,
      warningCodes: ['DURATION_ESTIMATED'],
    };
  }

  return {
    durationMinutes: GLOBAL_FALLBACK_MINUTES,
    durationSource: 'fallback',
    durationPolicyVersion: DURATION_POLICY_VERSION,
    policyCategory,
    warningCodes: ['DURATION_ESTIMATED'],
  };
}

module.exports = {
  CATEGORY_DEFAULTS,
  DURATION_POLICY_VERSION,
  GLOBAL_FALLBACK_MINUTES,
  policyCategoryForPoi,
  resolveStopDuration,
};

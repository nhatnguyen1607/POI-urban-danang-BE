const { DA_NANG_BBOX, DEFAULT_CITY_ID } = require('../../services/canonicalCsvPoiRepository');

const CITY_CONFIGS = {
  [DEFAULT_CITY_ID]: {
    cityId: DEFAULT_CITY_ID,
    displayNameVi: 'Da Nang',
    displayNameEn: 'Da Nang',
    countryCode: 'VN',
    timezone: 'Asia/Ho_Chi_Minh',
    currency: 'VND',
    center: {
      lat: 16.0544,
      lon: 108.2022,
    },
    bbox: DA_NANG_BBOX,
    status: 'READY_FOR_BETA',
    adminBoundaryVersion: 'pending_spatial_join',
  },
};

function getCityConfig(cityId = DEFAULT_CITY_ID) {
  return CITY_CONFIGS[cityId] || null;
}

function listCityConfigs() {
  return Object.values(CITY_CONFIGS);
}

function bboxToPolygonWkt(bbox) {
  return [
    'POLYGON((',
    `${bbox.west} ${bbox.south},`,
    `${bbox.east} ${bbox.south},`,
    `${bbox.east} ${bbox.north},`,
    `${bbox.west} ${bbox.north},`,
    `${bbox.west} ${bbox.south}`,
    '))',
  ].join(' ');
}

module.exports = {
  CITY_CONFIGS,
  bboxToPolygonWkt,
  getCityConfig,
  listCityConfigs,
};

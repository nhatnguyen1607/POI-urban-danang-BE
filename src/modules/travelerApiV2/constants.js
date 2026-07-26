const API_VERSION = 'v2';
const CONTRACT_VERSION = 'phase2-traveler-api-v2-draft-1';
const DATASET_VERSION = 'urbanagent-poi-master-v1';
const DEFAULT_LIMIT = 20;
const MAX_LIMIT = 100;

const CAPABILITY_STATES = Object.freeze({
  UNAVAILABLE: 'unavailable',
  PLANNED: 'planned',
  EXPERIMENTAL: 'experimental',
  AVAILABLE: 'available',
});

const PHASE2_CITY_STATUS = 'EXPERIMENTAL';

module.exports = {
  API_VERSION,
  CAPABILITY_STATES,
  CONTRACT_VERSION,
  DATASET_VERSION,
  DEFAULT_LIMIT,
  MAX_LIMIT,
  PHASE2_CITY_STATUS,
};

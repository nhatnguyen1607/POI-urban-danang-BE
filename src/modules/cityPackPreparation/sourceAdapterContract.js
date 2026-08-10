const ADAPTER_CONTRACT_VERSION = 'phase4-stage4c-v1';

function createSourceAdapter({ source, policyClass, normalize }) {
  if (!source || typeof source !== 'string') {
    throw new Error('Source adapter requires a stable source name.');
  }

  if (!policyClass || typeof policyClass !== 'string') {
    throw new Error(`Source adapter ${source} requires a policy class.`);
  }

  if (typeof normalize !== 'function') {
    throw new Error(`Source adapter ${source} requires a normalize() function.`);
  }

  return {
    contractVersion: ADAPTER_CONTRACT_VERSION,
    source,
    policyClass,
    normalize,
  };
}

module.exports = {
  ADAPTER_CONTRACT_VERSION,
  createSourceAdapter,
};

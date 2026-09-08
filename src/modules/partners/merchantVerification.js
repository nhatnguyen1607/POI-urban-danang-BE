const CLAIM_STATES = Object.freeze({
  UNCLAIMED: 'UNCLAIMED',
  PENDING: 'PENDING',
  VERIFIED: 'VERIFIED',
  REJECTED: 'REJECTED',
  SUSPENDED: 'SUSPENDED',
});

function cleanString(value, max = 500) {
  return typeof value === 'string' ? value.trim().slice(0, max) : '';
}

function normalizeMerchantRecord(input = {}) {
  const claimStatus = Object.values(CLAIM_STATES).includes(input.claimStatus)
    ? input.claimStatus
    : CLAIM_STATES.UNCLAIMED;
  return {
    merchantId: cleanString(input.merchantId, 160) || null,
    placeId: cleanString(input.placeId, 220) || null,
    claimStatus,
    verifiedAt: claimStatus === CLAIM_STATES.VERIFIED ? cleanString(input.verifiedAt, 40) || null : null,
    verifiedBy: claimStatus === CLAIM_STATES.VERIFIED ? cleanString(input.verifiedBy, 160) || null : null,
    submittedStatus: cleanString(input.submittedStatus, 80) || null,
    openingHours: Array.isArray(input.openingHours) ? input.openingHours.slice(0, 28) : [],
    temporaryClosure: input.temporaryClosure === true,
    serviceLinks: Array.isArray(input.serviceLinks) ? input.serviceLinks.slice(0, 12) : [],
    evidence: Array.isArray(input.evidence) ? input.evidence.slice(0, 40) : [],
    auditTrail: Array.isArray(input.auditTrail) ? input.auditTrail.slice(0, 100) : [],
  };
}

function merchantClaimResponse() {
  return {
    status: 'MERCHANT_CLAIM_MANUAL_VERIFICATION_REQUIRED',
    autoVerification: false,
    selfAssignmentAllowed: false,
  };
}

function createMerchantVerificationStore(records = []) {
  const items = records.map(normalizeMerchantRecord);
  return {
    list: () => items.map((item) => ({ ...item })),
    findVerified: (placeId) => items.find((item) => item.placeId === placeId && item.claimStatus === CLAIM_STATES.VERIFIED) || null,
    requestClaim: merchantClaimResponse,
  };
}

module.exports = {
  CLAIM_STATES,
  createMerchantVerificationStore,
  merchantClaimResponse,
  normalizeMerchantRecord,
};

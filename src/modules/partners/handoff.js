const { PROVIDER_CAPABILITIES } = require('./providerContract');

const DEFAULT_ALLOWED_DOMAINS = Object.freeze([
  'booking.com',
  'traveloka.com',
  'grab.com',
  'shopeefood.vn',
]);

function configuredDomains(env = process.env) {
  return String(env.URBANAGENT_PARTNER_HANDOFF_ALLOWED_DOMAINS || '')
    .split(',')
    .map((value) => value.trim().toLowerCase())
    .filter(Boolean);
}

function isAllowedHostname(hostname, allowedDomains) {
  const normalized = String(hostname || '').toLowerCase();
  return allowedDomains.some((domain) => normalized === domain || normalized.endsWith(`.${domain}`));
}

function validateHandoffUrl(value, { allowedDomains = [...DEFAULT_ALLOWED_DOMAINS, ...configuredDomains()] } = {}) {
  try {
    const url = new URL(String(value || ''));
    if (url.protocol !== 'https:' || !isAllowedHostname(url.hostname, allowedDomains)) {
      return { valid: false, url: null, reason: 'unsafe_or_unapproved_handoff_url' };
    }
    url.username = '';
    url.password = '';
    return { valid: true, url: url.toString(), reason: null };
  } catch {
    return { valid: false, url: null, reason: 'invalid_handoff_url' };
  }
}

function createHandoff(result, context = {}, options = {}) {
  const validated = validateHandoffUrl(result?.handoffUrl, options);
  if (!validated.valid) return null;
  const capability = result.capability;
  if (![PROVIDER_CAPABILITIES.BOOKING_HANDOFF, PROVIDER_CAPABILITIES.FOOD_HANDOFF, PROVIDER_CAPABILITIES.RIDE_HANDOFF].includes(capability)) {
    return null;
  }
  return {
    providerId: result.providerId,
    providerName: result.providerName,
    capability,
    url: validated.url,
    placeId: String(context.placeId || ''),
    destination: context.destination || null,
    requestedDates: context.requestedDates || null,
    guestCount: Number.isInteger(context.guestCount) && context.guestCount > 0 ? context.guestCount : null,
    evidenceState: result.status,
  };
}

module.exports = { DEFAULT_ALLOWED_DOMAINS, createHandoff, validateHandoffUrl };

const WARNING_ORDER = Object.freeze([
  'OPENING_HOURS_CONFLICT',
  'UNSCHEDULED_MUST_INCLUDE',
  'COORDINATES_MISSING',
  'OPENING_HOURS_UNPARSEABLE',
  'DAILY_WINDOW_TIGHT',
  'INSUFFICIENT_CANDIDATES',
  'PARTIAL_PREVIEW',
  'OPENING_HOURS_UNKNOWN',
  'DURATION_ESTIMATED',
  'TRAVEL_TIME_ESTIMATED',
  'ORIGIN_NOT_PROVIDED',
  'MAX_STOPS_APPLIED',
  'BUDGET_DATA_UNKNOWN',
]);

const WARNING_DEFINITIONS = Object.freeze({
  OPENING_HOURS_CONFLICT: {
    severity: 'warning',
    message: 'A known opening-hours conflict affects the preview.',
  },
  UNSCHEDULED_MUST_INCLUDE: {
    severity: 'warning',
    message: 'A requested must-include POI could not be scheduled.',
  },
  COORDINATES_MISSING: {
    severity: 'warning',
    message: 'Coordinates are missing for one or more POIs; route math is partial.',
  },
  OPENING_HOURS_UNPARSEABLE: {
    severity: 'warning',
    message: 'Opening-hours data exists but cannot be safely interpreted.',
  },
  DAILY_WINDOW_TIGHT: {
    severity: 'warning',
    message: 'The requested daily window is tight for the selected stops.',
  },
  INSUFFICIENT_CANDIDATES: {
    severity: 'warning',
    message: 'Fewer eligible recommendation candidates were available than requested.',
  },
  PARTIAL_PREVIEW: {
    severity: 'warning',
    message: 'The preview is incomplete and includes public-safe omissions.',
  },
  OPENING_HOURS_UNKNOWN: {
    severity: 'info',
    message: 'Verified opening-hours data is not available for one or more stops.',
  },
  DURATION_ESTIMATED: {
    severity: 'info',
    message: 'One or more stop durations are estimated.',
  },
  TRAVEL_TIME_ESTIMATED: {
    severity: 'info',
    message: 'Travel time uses a local Haversine approximation.',
  },
  ORIGIN_NOT_PROVIDED: {
    severity: 'info',
    message: 'Start location was not provided; first-leg distance and travel time are unknown.',
  },
  MAX_STOPS_APPLIED: {
    severity: 'info',
    message: 'The max-stops limit was applied.',
  },
  BUDGET_DATA_UNKNOWN: {
    severity: 'info',
    message: 'Budget fit cannot be fully evaluated with the approved runtime data.',
  },
});

function warningOrder(code) {
  const index = WARNING_ORDER.indexOf(code);
  return index === -1 ? WARNING_ORDER.length : index;
}

function normalizeScopePosition(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 999999;
}

function normalizeWarning(warning) {
  const definition = WARNING_DEFINITIONS[warning.code] || {
    severity: 'warning',
    message: 'A preview warning was emitted.',
  };
  return {
    code: warning.code,
    severity: definition.severity,
    message: warning.message || definition.message,
    scope: warning.scope || 'preview',
    ...(warning.dayNumber !== undefined ? { dayNumber: warning.dayNumber } : {}),
    ...(warning.stopOrder !== undefined ? { stopOrder: warning.stopOrder } : {}),
    ...(warning.legOrder !== undefined ? { legOrder: warning.legOrder } : {}),
    ...(warning.poiId ? { poiId: warning.poiId } : {}),
  };
}

function sortWarnings(warnings = []) {
  return [...warnings].sort((a, b) => (
    warningOrder(a.code) - warningOrder(b.code) ||
    normalizeScopePosition(a.dayNumber) - normalizeScopePosition(b.dayNumber) ||
    normalizeScopePosition(a.stopOrder) - normalizeScopePosition(b.stopOrder) ||
    normalizeScopePosition(a.legOrder) - normalizeScopePosition(b.legOrder) ||
    String(a.poiId || '').localeCompare(String(b.poiId || ''), 'en') ||
    String(a.scope || '').localeCompare(String(b.scope || ''), 'en')
  ));
}

function uniqueWarnings(warnings = []) {
  const seen = new Set();
  const result = [];
  for (const warning of sortWarnings(warnings).map(normalizeWarning)) {
    const key = [
      warning.code,
      warning.scope,
      warning.dayNumber ?? '',
      warning.stopOrder ?? '',
      warning.legOrder ?? '',
      warning.poiId || '',
    ].join('|');
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(warning);
  }
  return result;
}

function warningCodes(warnings = []) {
  return uniqueWarnings(warnings).map((warning) => warning.code);
}

module.exports = {
  WARNING_DEFINITIONS,
  WARNING_ORDER,
  sortWarnings,
  uniqueWarnings,
  warningCodes,
};

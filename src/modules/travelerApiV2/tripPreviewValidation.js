const {
  DEFAULT_TRIP_PREVIEW_RECOMMENDATION_LIMIT,
  MAX_TRIP_PREVIEW_RECOMMENDATION_LIMIT,
  TRIP_PREVIEW_PACE_DEFAULT_STOPS,
} = require('./constants');

const SUPPORTED_CITY_ID = 'da-nang';
const TIME_PATTERN = /^([01]\d|2[0-3]):([0-5]\d)$/;
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
const TRANSPORT_MODES = new Set(['walk', 'motorbike', 'car', 'taxi']);
const PACES = new Set(['relaxed', 'balanced', 'packed']);
const BUDGETS = new Set(['budget', 'moderate', 'premium', 'unknown']);

function isPlainObject(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function fieldError(field, rule) {
  return { field, rule };
}

function parseInteger(value, { field, min, max, required = false }) {
  if (value === undefined || value === null || value === '') {
    return required ? { error: fieldError(field, 'required') } : { value: null };
  }
  const number = Number(value);
  if (!Number.isInteger(number) || number < min || number > max) {
    return { error: fieldError(field, `integer_between_${min}_and_${max}`) };
  }
  return { value: number };
}

function parsePositiveNumber(value, { field, required = false }) {
  if (value === undefined || value === null || value === '') {
    return required ? { error: fieldError(field, 'required') } : { value: null };
  }
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) {
    return { error: fieldError(field, 'positive_number') };
  }
  return { value: number };
}

function parseTime(value, field) {
  if (value === undefined || value === null || value === '') return { value: null };
  const text = String(value).trim();
  if (!TIME_PATTERN.test(text)) return { error: fieldError(field, 'time_HH_mm') };
  const [, hours, minutes] = text.match(TIME_PATTERN);
  return {
    value: text,
    minutes: Number(hours) * 60 + Number(minutes),
  };
}

function parseDate(value, field) {
  if (value === undefined || value === null || value === '') return { value: null };
  const text = String(value).trim();
  if (!DATE_PATTERN.test(text) || Number.isNaN(Date.parse(`${text}T00:00:00Z`))) {
    return { error: fieldError(field, 'date_YYYY_MM_DD') };
  }
  return { value: text };
}

function parseCoordinates(location) {
  if (location === undefined || location === null) return { value: null };
  if (!isPlainObject(location)) {
    return { error: fieldError('startLocation', 'object_with_lat_lon') };
  }
  if (location.lat === undefined || location.lon === undefined) {
    return { error: fieldError('startLocation', 'lat_lon_required') };
  }
  const lat = Number(location.lat);
  const lon = Number(location.lon);
  if (!Number.isFinite(lat) || lat < -90 || lat > 90) {
    return { error: fieldError('startLocation.lat', 'number_between_-90_and_90') };
  }
  if (!Number.isFinite(lon) || lon < -180 || lon > 180) {
    return { error: fieldError('startLocation.lon', 'number_between_-180_and_180') };
  }
  return {
    value: {
      lat,
      lon,
      label: typeof location.label === 'string' ? location.label.trim() || null : null,
    },
  };
}

function uniqueStringArray(value, { field, max }) {
  if (value === undefined || value === null) return { value: [] };
  if (!Array.isArray(value)) return { error: fieldError(field, 'array') };
  if (value.length > max) return { error: fieldError(field, `maximum_${max}`) };
  const seen = new Set();
  const result = [];
  for (const item of value) {
    const text = String(item || '').trim();
    if (!text) return { error: fieldError(field, 'non_empty_string_items') };
    if (seen.has(text)) continue;
    seen.add(text);
    result.push(text);
  }
  return { value: result };
}

function validateTripPreviewRequest(body = {}) {
  const errors = [];
  if (!isPlainObject(body)) {
    return { errors: [fieldError('body', 'object')] };
  }

  const cityId = String(body.cityId || '').trim();
  if (!cityId) errors.push(fieldError('cityId', 'required'));

  const query = String(body.query || '').trim();
  if (!query || query.length > 500) errors.push(fieldError('query', 'string_length_1_to_500'));

  const trip = body.trip === undefined ? {} : body.trip;
  if (!isPlainObject(trip)) errors.push(fieldError('trip', 'object'));
  const safeTrip = isPlainObject(trip) ? trip : {};

  const date = parseDate(safeTrip.date, 'trip.date');
  if (date.error) errors.push(date.error);

  const startTime = parseTime(safeTrip.startTime, 'trip.startTime');
  if (startTime.error) errors.push(startTime.error);

  const duration = parseInteger(safeTrip.durationMinutes, {
    field: 'trip.durationMinutes',
    min: 15,
    max: 480,
  });
  if (duration.error) errors.push(duration.error);

  const dayCount = parseInteger(safeTrip.dayCount, {
    field: 'trip.dayCount',
    min: 1,
    max: 7,
  });
  if (dayCount.error) errors.push(dayCount.error);

  const transport = String(safeTrip.transport || 'motorbike').trim();
  if (!TRANSPORT_MODES.has(transport)) errors.push(fieldError('trip.transport', 'enum_walk_motorbike_car_taxi'));

  const pace = String(safeTrip.pace || 'balanced').trim();
  if (!PACES.has(pace)) errors.push(fieldError('trip.pace', 'enum_relaxed_balanced_packed'));

  const budget = String(safeTrip.budget || 'unknown').trim();
  if (!BUDGETS.has(budget)) errors.push(fieldError('trip.budget', 'enum_budget_moderate_premium_unknown'));

  let dailyWindow = null;
  if (safeTrip.dailyWindow !== undefined && safeTrip.dailyWindow !== null) {
    if (!isPlainObject(safeTrip.dailyWindow)) {
      errors.push(fieldError('trip.dailyWindow', 'object_with_start_end'));
    } else {
      const start = parseTime(safeTrip.dailyWindow.start, 'trip.dailyWindow.start');
      const end = parseTime(safeTrip.dailyWindow.end, 'trip.dailyWindow.end');
      if (start.error) errors.push(start.error);
      if (end.error) errors.push(end.error);
      if (!start.error && !end.error) {
        if (start.value === null || end.value === null) {
          errors.push(fieldError('trip.dailyWindow', 'start_end_required'));
        } else {
          const span = end.minutes - start.minutes;
          if (span <= 0) {
            errors.push(fieldError('trip.dailyWindow', 'same_day_end_after_start'));
          } else if (span < 15 || span > 480) {
            errors.push(fieldError('trip.dailyWindow', 'span_minutes_between_15_and_480'));
          } else {
            dailyWindow = {
              start: start.value,
              end: end.value,
              startMinutes: start.minutes,
              endMinutes: end.minutes,
              spanMinutes: span,
            };
          }
        }
      }
    }
  }

  const startLocation = parseCoordinates(body.startLocation);
  if (startLocation.error) errors.push(startLocation.error);

  const constraints = body.constraints === undefined ? {} : body.constraints;
  if (!isPlainObject(constraints)) errors.push(fieldError('constraints', 'object'));
  const safeConstraints = isPlainObject(constraints) ? constraints : {};

  const maxStops = parseInteger(safeConstraints.maxStopsPerDay, {
    field: 'constraints.maxStopsPerDay',
    min: 1,
    max: 6,
  });
  if (maxStops.error) errors.push(maxStops.error);

  const maxDistance = parsePositiveNumber(safeConstraints.maxDistanceKm, {
    field: 'constraints.maxDistanceKm',
  });
  if (maxDistance.error) errors.push(maxDistance.error);

  const mustInclude = uniqueStringArray(safeConstraints.mustIncludePoiIds, {
    field: 'constraints.mustIncludePoiIds',
    max: 20,
  });
  if (mustInclude.error) errors.push(mustInclude.error);

  const exclude = uniqueStringArray(safeConstraints.excludePoiIds, {
    field: 'constraints.excludePoiIds',
    max: 100,
  });
  if (exclude.error) errors.push(exclude.error);

  const recommendationOptions = body.recommendationOptions === undefined ? {} : body.recommendationOptions;
  if (!isPlainObject(recommendationOptions)) errors.push(fieldError('recommendationOptions', 'object'));
  const safeRecommendationOptions = isPlainObject(recommendationOptions) ? recommendationOptions : {};
  const recommendationLimit = parseInteger(safeRecommendationOptions.limit, {
    field: 'recommendationOptions.limit',
    min: 1,
    max: MAX_TRIP_PREVIEW_RECOMMENDATION_LIMIT,
  });
  if (recommendationLimit.error) errors.push(recommendationLimit.error);

  if (errors.length) return { errors };

  const normalizedPace = PACES.has(pace) ? pace : 'balanced';
  const defaultMaxStopsPerDay = TRIP_PREVIEW_PACE_DEFAULT_STOPS[normalizedPace];

  return {
    value: {
      cityId,
      query,
      trip: {
        date: date.value,
        startTime: startTime.value,
        startTimeMinutes: startTime.minutes ?? null,
        durationMinutes: duration.value,
        transport: TRANSPORT_MODES.has(transport) ? transport : 'motorbike',
        pace: normalizedPace,
        dayCount: dayCount.value || 1,
        dailyWindow,
        party: isPlainObject(safeTrip.party) ? safeTrip.party : {},
        budget: BUDGETS.has(budget) ? budget : 'unknown',
      },
      startLocation: startLocation.value,
      preferences: isPlainObject(body.preferences) ? body.preferences : {},
      constraints: {
        maxStopsPerDay: maxStops.value || defaultMaxStopsPerDay,
        mustIncludePoiIds: mustInclude.value,
        excludePoiIds: exclude.value,
        maxDistanceKm: maxDistance.value,
      },
      recommendationOptions: {
        limit: recommendationLimit.value || DEFAULT_TRIP_PREVIEW_RECOMMENDATION_LIMIT,
      },
    },
  };
}

module.exports = {
  TIME_PATTERN,
  validateTripPreviewRequest,
};

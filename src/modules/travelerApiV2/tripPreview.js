const { loadPOIs, normalizeText } = require('../../services/poiDataService');
const {
  DATASET_VERSION,
  MAX_TRIP_PREVIEW_RECOMMENDATION_LIMIT,
} = require('./constants');
const {
  getTravelerRecommendationCandidates,
} = require('./recommendations');
const { serializePoi } = require('./serializers');
const {
  DURATION_POLICY_VERSION,
  policyCategoryForPoi,
  resolveStopDuration,
} = require('./tripPreviewDurationPolicy');
const {
  TRAVEL_ESTIMATION_METHOD,
  TRAVEL_POLICY_VERSION,
  estimateTravelLeg,
  isValidCoordinate,
  unknownLeg,
} = require('./tripPreviewTravelPolicy');
const {
  uniqueWarnings,
  warningCodes,
} = require('./tripPreviewWarnings');

const CONTRACT_VERSION = 'phase2-batch3-trip-preview-v1';
const DEFAULT_DAY_START = '09:00';

function canonicalIdFromPoi(poi = {}) {
  return String(poi.globalId || poi.id || '').trim();
}

function minutesToTime(minutes) {
  const normalized = ((minutes % 1440) + 1440) % 1440;
  const hours = Math.floor(normalized / 60);
  const mins = normalized % 60;
  return `${String(hours).padStart(2, '0')}:${String(mins).padStart(2, '0')}`;
}

function parseTimeToMinutes(value) {
  if (!value) return null;
  const [hours, minutes] = String(value).split(':').map(Number);
  if (!Number.isInteger(hours) || !Number.isInteger(minutes)) return null;
  return hours * 60 + minutes;
}

function normalizeOpeningHours(value) {
  const text = String(value || '').trim();
  if (!text) return { status: 'unknown', ranges: [] };
  const ranges = [];
  for (const part of text.split('|')) {
    const match = part.trim().match(/^([01]\d|2[0-3]):([0-5]\d)\s*-\s*([01]\d|2[0-3]):([0-5]\d)$/);
    if (!match) {
      return { status: 'unparseable', ranges: [] };
    }
    const start = Number(match[1]) * 60 + Number(match[2]);
    const end = Number(match[3]) * 60 + Number(match[4]);
    if (end <= start) return { status: 'unparseable', ranges: [] };
    ranges.push({ start, end });
  }
  return { status: 'known', ranges };
}

function isWithinOpeningHours({ openingHoursRaw, arrivalMinutes, departureMinutes }) {
  const parsed = normalizeOpeningHours(openingHoursRaw);
  if (parsed.status !== 'known') return parsed.status;
  const fits = parsed.ranges.some((range) => arrivalMinutes >= range.start && departureMinutes <= range.end);
  return fits ? 'open' : 'conflict';
}

function publicPoiWithOpeningHours(poi) {
  return {
    ...serializePoi(poi),
    openingHours: {
      status: poi.openingHoursRaw ? 'available_unverified_runtime' : 'unknown',
      raw: null,
    },
  };
}

function buildCandidateFromRecommendation(item, rank) {
  return {
    id: item.poi.globalId || item.poi.id,
    poi: item.poi,
    score: Number(item.score) || 0,
    rank,
    reason: item.reason || 'Selected from deterministic recommendation candidates.',
    reasonCodes: Array.isArray(item.reasonCodes) ? item.reasonCodes : [],
    provenance: item.provenance || item.poi.provenance || null,
    selectionSource: 'recommendation',
  };
}

function buildCandidateFromPoi(poi, rank) {
  return {
    id: canonicalIdFromPoi(poi),
    poi,
    score: 100,
    rank,
    reason: 'Included because it was requested as a must-include POI.',
    reasonCodes: ['must_include', 'canonical_dataset_match'],
    provenance: serializePoi(poi).provenance,
    selectionSource: 'must_include',
  };
}

function deduplicateCandidates(candidates, mustIncludeIds) {
  const mustIncludeSet = new Set(mustIncludeIds);
  const byId = new Map();
  for (const candidate of candidates) {
    const id = String(candidate.id);
    const existing = byId.get(id);
    if (!existing) {
      byId.set(id, candidate);
      continue;
    }
    if (mustIncludeSet.has(id) && existing.selectionSource !== 'must_include') {
      byId.set(id, {
        ...candidate,
        selectionSource: 'must_include',
        reasonCodes: [...new Set(['must_include', ...candidate.reasonCodes])],
      });
    }
  }
  return Array.from(byId.values());
}

function compareCandidate(a, b) {
  return (
    Number(b.score || 0) - Number(a.score || 0) ||
    Number(a.rank || 0) - Number(b.rank || 0) ||
    String(a.id).localeCompare(String(b.id), 'en') ||
    normalizeText(a.poi?.name || '').localeCompare(normalizeText(b.poi?.name || ''), 'en')
  );
}

function coordinateOfCandidate(candidate) {
  const location = candidate.poi?.location || candidate.poi || {};
  return {
    lat: location.lat,
    lon: location.lon,
  };
}

function orderCandidatesGeographically(candidates, startLocation) {
  const remaining = [...candidates].sort(compareCandidate);
  const ordered = [];
  let currentPoint = startLocation && isValidCoordinate(startLocation) ? startLocation : null;

  while (remaining.length) {
    if (!currentPoint) {
      const first = remaining.shift();
      ordered.push(first);
      currentPoint = isValidCoordinate(coordinateOfCandidate(first)) ? coordinateOfCandidate(first) : null;
      continue;
    }

    let bestIndex = 0;
    let bestDistance = Infinity;
    for (let index = 0; index < remaining.length; index += 1) {
      const candidatePoint = coordinateOfCandidate(remaining[index]);
      if (!isValidCoordinate(candidatePoint)) continue;
      const leg = estimateTravelLeg({
        from: currentPoint,
        to: candidatePoint,
        transport: 'motorbike',
      });
      const distance = leg.distanceMeters ?? Infinity;
      if (
        distance < bestDistance ||
        (distance === bestDistance && compareCandidate(remaining[index], remaining[bestIndex]) < 0)
      ) {
        bestIndex = index;
        bestDistance = distance;
      }
    }
    const [next] = remaining.splice(bestIndex, 1);
    ordered.push(next);
    currentPoint = isValidCoordinate(coordinateOfCandidate(next)) ? coordinateOfCandidate(next) : currentPoint;
  }

  return ordered;
}

function resolveDayWindow(request, dayNumber) {
  const configured = request.trip.dailyWindow;
  if (configured) {
    const spanMinutes = request.trip.durationMinutes
      ? Math.min(configured.spanMinutes, request.trip.durationMinutes)
      : configured.spanMinutes;
    return {
      ...configured,
      endMinutes: configured.startMinutes + spanMinutes,
      spanMinutes,
    };
  }
  if (request.trip.startTime) {
    const startMinutes = request.trip.startTimeMinutes;
    const spanMinutes = request.trip.durationMinutes || 480;
    return {
      start: request.trip.startTime,
      end: minutesToTime(startMinutes + spanMinutes),
      startMinutes,
      endMinutes: startMinutes + spanMinutes,
      spanMinutes,
    };
  }
  return {
    start: null,
    end: null,
    startMinutes: null,
    endMinutes: null,
    spanMinutes: request.trip.durationMinutes || 480,
    dayNumber,
  };
}

function addUnscheduled(unscheduled, item) {
  unscheduled.push({
    poiId: item.poiId || null,
    reasonCode: item.reasonCode,
    message: item.message,
    requested: Boolean(item.requested),
  });
}

function publicTravelLeg(leg) {
  const { legOrder, ...publicLeg } = leg;
  return publicLeg;
}

function routeSummaryFromLegs(stops) {
  const legs = stops.map((stop) => stop.travelFromPrevious);
  const knownLegs = legs.filter((leg) => leg.distanceKnown && leg.travelTimeKnown);
  const unknownLegs = legs.filter((leg) => !leg.distanceKnown || !leg.travelTimeKnown);
  const totalStayMinutes = stops.reduce((sum, stop) => sum + stop.durationMinutes, 0);
  if (unknownLegs.length) {
    return {
      totalDistanceKm: null,
      totalTravelMinutes: null,
      totalStayMinutes,
      totalPlanMinutes: null,
      distanceFullyKnown: false,
      travelTimeFullyKnown: false,
      knownLegCount: knownLegs.length,
      unknownLegCount: unknownLegs.length,
      calculationSource: 'partial-local-haversine-estimate',
      status: 'partial',
      warnings: warningCodes(legs.flatMap((leg) => leg.warnings || [])),
    };
  }
  const distanceMeters = knownLegs.reduce((sum, leg) => sum + leg.distanceMeters, 0);
  const totalTravelMinutes = knownLegs.reduce((sum, leg) => sum + leg.travelDurationMinutes, 0);
  return {
    totalDistanceKm: Number((distanceMeters / 1000).toFixed(2)),
    totalTravelMinutes,
    totalStayMinutes,
    totalPlanMinutes: totalTravelMinutes + totalStayMinutes,
    distanceFullyKnown: true,
    travelTimeFullyKnown: true,
    knownLegCount: knownLegs.length,
    unknownLegCount: 0,
    calculationSource: TRAVEL_ESTIMATION_METHOD,
    status: 'known',
    warnings: warningCodes(legs.flatMap((leg) => leg.warnings || [])),
  };
}

function previewStatus({ stops, warnings, unscheduled }) {
  if (!stops.length) return 'INFEASIBLE';
  if (unscheduled.length) return 'PARTIAL';
  const warningObjects = uniqueWarnings(warnings);
  if (warningObjects.length) return 'FEASIBLE_WITH_WARNINGS';
  return 'FEASIBLE';
}

function dayStatus(day) {
  if (!day.stops.length) return 'INFEASIBLE';
  if (day.unscheduled.length) return 'PARTIAL';
  if (day.warnings.length) return 'FEASIBLE_WITH_WARNINGS';
  return 'FEASIBLE';
}

function collectStopWarningObjects(stop) {
  return (stop.warnings || []).map((code) => ({
    code,
    scope: 'stop',
    dayNumber: stop.dayNumber,
    stopOrder: stop.order,
    poiId: stop.poi.globalId,
  }));
}

function scheduleCandidates({ candidates, request, mustIncludeIds }) {
  const mustIncludeSet = new Set(mustIncludeIds);
  const ordered = orderCandidatesGeographically(candidates, request.startLocation);
  const maxStopsPerDay = request.constraints.maxStopsPerDay;
  const totalStopLimit = maxStopsPerDay * request.trip.dayCount;
  const selected = ordered.slice(0, totalStopLimit);
  const unscheduled = [];
  const allWarnings = [];

  if (ordered.length > totalStopLimit) {
    allWarnings.push({ code: 'MAX_STOPS_APPLIED', scope: 'preview' });
  }
  if (ordered.length < totalStopLimit) {
    allWarnings.push({ code: 'INSUFFICIENT_CANDIDATES', scope: 'preview' });
  }

  const days = Array.from({ length: request.trip.dayCount }, (_, index) => {
    const dayNumber = index + 1;
    const window = resolveDayWindow(request, dayNumber);
    return {
      dayNumber,
      date: request.trip.date || null,
      dailyWindow: window.start && window.end ? { start: window.start, end: window.end } : null,
      feasibilityStatus: 'INFEASIBLE',
      stops: [],
      unscheduled: [],
      warnings: [],
      _cursorMinutes: window.startMinutes,
      _window: window,
      _previousPoint: index === 0 ? request.startLocation : null,
    };
  });

  let globalOrder = 1;
  for (const candidate of selected) {
    const targetDay = days.find((day) => day.stops.length < maxStopsPerDay) || days[days.length - 1];
    const duration = resolveStopDuration({ poi: candidate.poi });
    const point = coordinateOfCandidate(candidate);
    const leg = targetDay.stops.length === 0 && !targetDay._previousPoint
      ? unknownLeg({
          transport: request.trip.transport,
          calculationSource: 'missing-origin',
          legOrder: globalOrder,
        })
      : estimateTravelLeg({
          from: targetDay._previousPoint,
          to: point,
          transport: request.trip.transport,
          legOrder: globalOrder,
        });

    const legWarnings = [];
    if (!leg.distanceKnown || !leg.travelTimeKnown) {
      if (leg.calculationSource === 'missing-origin') {
        legWarnings.push({ code: 'ORIGIN_NOT_PROVIDED', scope: 'leg', legOrder: globalOrder, poiId: candidate.id });
      } else {
        legWarnings.push({ code: 'COORDINATES_MISSING', scope: 'leg', legOrder: globalOrder, poiId: candidate.id });
      }
    } else {
      legWarnings.push({ code: 'TRAVEL_TIME_ESTIMATED', scope: 'leg', legOrder: globalOrder, poiId: candidate.id });
    }
    leg.warnings = legWarnings;

    const knownTravelMinutes = leg.travelTimeKnown ? leg.travelDurationMinutes : 0;
    const arrivalMinutes = targetDay._cursorMinutes === null ? null : targetDay._cursorMinutes + knownTravelMinutes;
    const departureMinutes = arrivalMinutes === null ? null : arrivalMinutes + duration.durationMinutes;
    const wouldOverflow = departureMinutes !== null &&
      targetDay._window.endMinutes !== null &&
      departureMinutes > targetDay._window.endMinutes;
    const hardMustInclude = mustIncludeSet.has(candidate.id);

    if (wouldOverflow) {
      const item = {
        poiId: candidate.id,
        reasonCode: hardMustInclude ? 'must_include_does_not_fit_window' : 'daily_window_overflow',
        message: hardMustInclude
          ? 'A requested must-include POI could not fit within the requested daily window.'
          : 'An optional recommendation could not fit within the requested daily window.',
        requested: hardMustInclude,
      };
      addUnscheduled(unscheduled, item);
      targetDay.unscheduled.push(item);
      allWarnings.push({ code: hardMustInclude ? 'UNSCHEDULED_MUST_INCLUDE' : 'DAILY_WINDOW_TIGHT', scope: 'day', dayNumber: targetDay.dayNumber, poiId: candidate.id });
      continue;
    }

    const openingStatus = arrivalMinutes === null
      ? (candidate.poi.openingHoursRaw ? normalizeOpeningHours(candidate.poi.openingHoursRaw).status : 'unknown')
      : isWithinOpeningHours({
          openingHoursRaw: candidate.poi.openingHoursRaw,
          arrivalMinutes,
          departureMinutes,
        });
    const openingWarnings = [];
    if (openingStatus === 'unknown') {
      openingWarnings.push({ code: 'OPENING_HOURS_UNKNOWN', scope: 'stop', dayNumber: targetDay.dayNumber, stopOrder: globalOrder, poiId: candidate.id });
    } else if (openingStatus === 'unparseable') {
      openingWarnings.push({ code: 'OPENING_HOURS_UNPARSEABLE', scope: 'stop', dayNumber: targetDay.dayNumber, stopOrder: globalOrder, poiId: candidate.id });
    } else if (openingStatus === 'conflict') {
      if (!hardMustInclude) {
        const item = {
          poiId: candidate.id,
          reasonCode: 'opening_hours_conflict',
          message: 'A known opening-hours conflict prevented scheduling this optional stop.',
          requested: false,
        };
        addUnscheduled(unscheduled, item);
        targetDay.unscheduled.push(item);
        allWarnings.push({ code: 'OPENING_HOURS_CONFLICT', scope: 'stop', dayNumber: targetDay.dayNumber, stopOrder: globalOrder, poiId: candidate.id });
        continue;
      }
      openingWarnings.push({ code: 'OPENING_HOURS_CONFLICT', scope: 'stop', dayNumber: targetDay.dayNumber, stopOrder: globalOrder, poiId: candidate.id });
    }

    const stopWarningObjects = uniqueWarnings([
      ...legWarnings,
      ...openingWarnings,
      ...duration.warningCodes.map((code) => ({ code, scope: 'stop', dayNumber: targetDay.dayNumber, stopOrder: globalOrder, poiId: candidate.id })),
    ]);
    const stop = {
      stopId: `stop_${globalOrder}`,
      order: globalOrder,
      dayNumber: targetDay.dayNumber,
      poi: publicPoiWithOpeningHours(candidate.poi),
      arrivalTime: arrivalMinutes === null ? null : minutesToTime(arrivalMinutes),
      departureTime: departureMinutes === null ? null : minutesToTime(departureMinutes),
      durationMinutes: duration.durationMinutes,
      durationSource: duration.durationSource,
      durationPolicyVersion: DURATION_POLICY_VERSION,
      travelFromPrevious: publicTravelLeg(leg),
      reason: candidate.reason,
      reasonCodes: [...new Set([...candidate.reasonCodes, 'route_preview'])],
      warnings: warningCodes(stopWarningObjects),
    };
    targetDay.stops.push(stop);
    targetDay.warnings.push(...stopWarningObjects);
    allWarnings.push(...stopWarningObjects);
    targetDay._cursorMinutes = departureMinutes;
    targetDay._previousPoint = isValidCoordinate(point) ? point : targetDay._previousPoint;
    globalOrder += 1;
  }

  if (request.trip.budget !== 'unknown') {
    allWarnings.push({ code: 'BUDGET_DATA_UNKNOWN', scope: 'preview' });
  }
  if (unscheduled.length) {
    allWarnings.push({ code: 'PARTIAL_PREVIEW', scope: 'preview' });
  }

  const publicDays = days.map((day) => {
    const warnings = uniqueWarnings(day.warnings);
    const publicDay = {
      dayNumber: day.dayNumber,
      date: day.date,
      dailyWindow: day.dailyWindow,
      feasibilityStatus: dayStatus({ ...day, warnings }),
      stops: day.stops.map((stop) => stop.stopId),
      stopCount: day.stops.length,
      unscheduled: day.unscheduled,
      warnings,
    };
    return publicDay;
  });

  const stops = days.flatMap((day) => day.stops);
  const warnings = uniqueWarnings(allWarnings);
  return {
    stops,
    days: publicDays,
    warnings,
    unscheduled,
    routeSummary: routeSummaryFromLegs(stops),
    feasibilityStatus: previewStatus({ stops, warnings, unscheduled }),
  };
}

function hardConstraintOverlap(request) {
  const excludeSet = new Set(request.constraints.excludePoiIds);
  return request.constraints.mustIncludePoiIds.filter((id) => excludeSet.has(id));
}

async function buildTripPreview(request) {
  const overlap = hardConstraintOverlap(request);
  if (overlap.length) {
    return {
      error: {
        status: 422,
        code: 'NO_FEASIBLE_ITINERARY',
        message: 'The requested hard trip constraints cannot produce a feasible itinerary.',
        details: {
          cityId: request.cityId,
          hardConstraintsSatisfied: false,
          reasonCode: 'include_exclude_overlap',
          poiIds: overlap,
        },
      },
    };
  }

  const allPois = await loadPOIs({ cityId: request.cityId });
  const poiById = new Map(allPois.map((poi) => [canonicalIdFromPoi(poi), poi]));
  const excludeSet = new Set(request.constraints.excludePoiIds);
  const mustIncludePois = [];
  const unscheduled = [];
  const baseWarnings = [];

  for (const poiId of request.constraints.mustIncludePoiIds) {
    const poi = poiById.get(poiId);
    if (!poi) {
      addUnscheduled(unscheduled, {
        poiId,
        reasonCode: 'unknown_canonical_id',
        message: 'A requested must-include POI ID is not in the approved canonical dataset.',
        requested: true,
      });
      baseWarnings.push({ code: 'UNSCHEDULED_MUST_INCLUDE', scope: 'preview', poiId });
    } else if (!excludeSet.has(poiId)) {
      mustIncludePois.push(poi);
    }
  }

  const recommendations = await getTravelerRecommendationCandidates({
    query: request.query,
    context: {
      location: request.startLocation || undefined,
      maxDistanceKm: request.constraints.maxDistanceKm || undefined,
      budget: request.trip.budget,
      preferences: request.preferences,
    },
    limit: request.recommendationOptions.limit,
    cityId: request.cityId,
    maxCandidateLimit: MAX_TRIP_PREVIEW_RECOMMENDATION_LIMIT,
  });

  const recommendationCandidates = recommendations.recommendations
    .map((item, index) => {
      const candidate = buildCandidateFromRecommendation(item, index);
      return {
        ...candidate,
        poi: poiById.get(candidate.id) || candidate.poi,
      };
    })
    .filter((candidate) => !excludeSet.has(candidate.id));
  const mustIncludeCandidates = mustIncludePois.map((poi, index) => buildCandidateFromPoi(poi, -1000 + index));
  const candidates = deduplicateCandidates([
    ...mustIncludeCandidates,
    ...recommendationCandidates,
  ], request.constraints.mustIncludePoiIds).sort((a, b) => {
    const aMust = request.constraints.mustIncludePoiIds.includes(a.id) ? 0 : 1;
    const bMust = request.constraints.mustIncludePoiIds.includes(b.id) ? 0 : 1;
    return aMust - bMust || compareCandidate(a, b);
  });

  if (!candidates.length) {
    return {
      trip: {
        tripId: null,
        preview: true,
        persisted: false,
        authenticated: false,
        saveEligible: false,
        cityId: request.cityId,
        query: request.query,
        durationMinutes: request.trip.durationMinutes,
        transport: request.trip.transport,
        pace: request.trip.pace,
        date: request.trip.date,
        startTime: request.trip.startTime,
        dayCount: request.trip.dayCount,
        dailyWindow: request.trip.dailyWindow ? { start: request.trip.dailyWindow.start, end: request.trip.dailyWindow.end } : null,
        timeKnown: Boolean(request.trip.startTime || request.trip.dailyWindow),
        feasibilityStatus: 'INFEASIBLE',
        stops: [],
        days: [],
        routeSummary: routeSummaryFromLegs([]),
        alternatives: [],
        unscheduled,
        explanation: {
          summary: 'No eligible canonical POIs could be scheduled for the requested preview.',
          reasonCodes: ['no_eligible_candidates'],
        },
        dataFreshness: {
          source: 'canonical_dataset',
          observedAt: null,
          lastVerifiedAt: null,
          status: 'unknown',
        },
        warnings: uniqueWarnings([
          ...baseWarnings,
          { code: 'INSUFFICIENT_CANDIDATES', scope: 'preview' },
        ]),
        provenance: {
          source: 'canonical',
          datasetVersion: DATASET_VERSION,
          externalLiveDataUsed: false,
        },
        contract: {
          version: CONTRACT_VERSION,
          durationPolicyVersion: DURATION_POLICY_VERSION,
          travelPolicyVersion: TRAVEL_POLICY_VERSION,
        },
      },
    };
  }

  const scheduled = scheduleCandidates({
    candidates,
    request,
    mustIncludeIds: request.constraints.mustIncludePoiIds,
  });
  scheduled.unscheduled.unshift(...unscheduled);
  const combinedWarnings = uniqueWarnings([
    ...baseWarnings,
    ...scheduled.warnings,
    ...(scheduled.unscheduled.length ? [{ code: 'PARTIAL_PREVIEW', scope: 'preview' }] : []),
  ]);
  const finalStatus = previewStatus({
    stops: scheduled.stops,
    warnings: combinedWarnings,
    unscheduled: scheduled.unscheduled,
  });

  if (request.constraints.mustIncludePoiIds.length &&
    scheduled.unscheduled.some((item) => item.requested && item.reasonCode !== 'unknown_canonical_id') &&
    !scheduled.stops.some((stop) => request.constraints.mustIncludePoiIds.includes(stop.poi.globalId))) {
    return {
      error: {
        status: 422,
        code: 'NO_FEASIBLE_ITINERARY',
        message: 'The requested hard trip constraints cannot produce a feasible itinerary.',
        details: {
          cityId: request.cityId,
          hardConstraintsSatisfied: false,
        },
      },
    };
  }

  return {
    trip: {
      tripId: null,
      preview: true,
      persisted: false,
      authenticated: false,
      saveEligible: false,
      cityId: request.cityId,
      query: request.query,
      durationMinutes: request.trip.durationMinutes,
      transport: request.trip.transport,
      pace: request.trip.pace,
      date: request.trip.date,
      startTime: request.trip.startTime || request.trip.dailyWindow?.start || null,
      dayCount: request.trip.dayCount,
      dailyWindow: request.trip.dailyWindow ? { start: request.trip.dailyWindow.start, end: request.trip.dailyWindow.end } : null,
      timeKnown: Boolean(request.trip.startTime || request.trip.dailyWindow),
      feasibilityStatus: finalStatus,
      stops: scheduled.stops,
      days: scheduled.days,
      routeSummary: scheduled.routeSummary,
      alternatives: [],
      unscheduled: scheduled.unscheduled,
      explanation: {
        summary: 'Preview generated from deterministic recommendation candidates and approximate local travel estimates.',
        reasonCodes: ['intent_match', 'route_preview'],
      },
      dataFreshness: {
        source: 'canonical_dataset',
        observedAt: null,
        lastVerifiedAt: null,
        status: 'unknown',
      },
      warnings: combinedWarnings,
      provenance: {
        source: 'canonical',
        datasetVersion: DATASET_VERSION,
        externalLiveDataUsed: false,
      },
      contract: {
        version: CONTRACT_VERSION,
        durationPolicyVersion: DURATION_POLICY_VERSION,
        travelPolicyVersion: TRAVEL_POLICY_VERSION,
      },
    },
  };
}

module.exports = {
  CONTRACT_VERSION,
  DEFAULT_DAY_START,
  buildTripPreview,
  isWithinOpeningHours,
  minutesToTime,
  normalizeOpeningHours,
  parseTimeToMinutes,
  policyCategoryForPoi,
  scheduleCandidates,
};

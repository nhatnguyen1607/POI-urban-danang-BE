/**
 * =============================================================================
 *  HỆ CHUYÊN GIA (Expert System) - Backward Chaining Inference Engine
 *  Tối ưu chỉ đường tuân theo luật giao thông Đà Nẵng
 * =============================================================================
 * 
 *  Rule Base: 3 file CSV luật giao thông
 *    1. cam_queo.csv       - Luật cấm quẹo / quay đầu
 *    2. cam_theo_gio.csv   - Luật cấm theo khung giờ
 *    3. duong_1_chieu.csv  - Đường một chiều
 * 
 *  Inference Engine: Backward Chaining
 *    GOAL: route_is_legal(origin → destination)
 *      ├── SUB-GOAL: all_segments_legal
 *      │   ├── not_wrong_way_one_way(segment)
 *      │   ├── not_time_restricted(segment, current_time)
 *      │   └── turn_legal(prev_segment → segment)
 */

const fs = require('fs');
const csv = require('csv-parser');
const path = require('path');

// ============================================================================
//  RULE BASE - Load & Parse CSV Files
// ============================================================================

class RuleBase {
  constructor() {
    this.turnRestrictions = [];    // cam_queo.csv
    this.timeRestrictions = [];    // cam_theo_gio.csv
    this.oneWayStreets = [];       // duong_1_chieu.csv
    this.loaded = false;
  }

  /**
   * Load all 3 CSV rule files
   */
  async load() {
    const ruleDir = path.join(__dirname, 'rule');

    const [turnRules, timeRules, oneWayRules] = await Promise.all([
      this._readCSV(path.join(ruleDir, 'cam_queo.csv')),
      this._readCSV(path.join(ruleDir, 'cam_theo_gio.csv')),
      this._readCSV(path.join(ruleDir, 'duong_1_chieu.csv')),
    ]);

    // Parse turn restrictions
    this.turnRestrictions = turnRules
      .filter(row => row['Loại cấm'] && row['Loại cấm'] !== 'N/A')
      .map(row => ({
        type: this._normalizeTurnType(row['Loại cấm']),
        fromRoad: this._normalizeRoadName(row['Đang đi trên đường']),
        toRoad: this._normalizeRoadName(row['Cấm rẽ vào đường']),
        rawType: row['Loại cấm'],
      }));

    // Parse time restrictions
    this.timeRestrictions = timeRules.map(row => ({
      roadName: this._normalizeRoadName(row['Tên đường']),
      timeRanges: this._parseTimeRanges(row['Khung giờ']),
      restrictionType: row['Loại cấm'],
      rawTimeStr: row['Khung giờ'],
    }));

    // Parse one-way streets
    this.oneWayStreets = oneWayRules.map(row => {
      const startCoord = this._parseCoord(row['Tọa độ bắt đầu']);
      const endCoord = this._parseCoord(row['Tọa độ kết thúc']);
      return {
        roadName: this._normalizeRoadName(row['Tên đường']),
        startLat: startCoord.lat,
        startLng: startCoord.lng,
        endLat: endCoord.lat,
        endLng: endCoord.lng,
      };
    });

    this.loaded = true;
    console.log(`[ES] Rule Base loaded: ${this.turnRestrictions.length} turn rules, ${this.timeRestrictions.length} time rules, ${this.oneWayStreets.length} one-way segments`);
  }

  // --- Parsing helpers ---

  _readCSV(filePath) {
    return new Promise((resolve, reject) => {
      const results = [];
      fs.createReadStream(filePath, { encoding: 'utf-8' })
        .pipe(csv())
        .on('data', (data) => results.push(data))
        .on('end', () => resolve(results))
        .on('error', (err) => reject(err));
    });
  }

  _normalizeRoadName(name) {
    if (!name) return '';
    return name
      .trim()
      .replace(/^Đường\s+/i, '')
      .replace(/^Cầu\s+/i, 'Cầu ')
      .replace(/^Hầm chui\s+/i, 'Hầm chui ')
      .toLowerCase()
      .normalize('NFC');
  }

  _normalizeTurnType(type) {
    if (!type) return 'unknown';
    const t = type.toLowerCase().trim();
    if (t.includes('quẹo phải') || t.includes('rẽ phải')) return 'no_right_turn';
    if (t.includes('quẹo trái') || t.includes('rẽ trái')) return 'no_left_turn';
    if (t.includes('quay đầu')) return 'no_u_turn';
    if (t.includes('đi thẳng')) return 'no_straight';
    if (t.includes('chỉ được quẹo phải') || t.includes('chỉ được rẽ phải')) return 'only_right_turn';
    return 'unknown';
  }

  _parseTimeRanges(timeStr) {
    if (!timeStr) return [];
    // Handle formats like "05:00- 07:00,16:00 - 20:00" or "06:30-08:30 AND 16:30-18:30"
    const ranges = [];
    const parts = timeStr.replace(/AND/gi, ',').split(',');
    for (const part of parts) {
      const match = part.trim().match(/(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})/);
      if (match) {
        ranges.push({
          startHour: parseInt(match[1]),
          startMin: parseInt(match[2]),
          endHour: parseInt(match[3]),
          endMin: parseInt(match[4]),
        });
      }
    }
    return ranges;
  }

  _parseCoord(coordStr) {
    if (!coordStr) return { lat: 0, lng: 0 };
    // Format: "16.0722113,108.2130547"
    const clean = coordStr.replace(/"/g, '').trim();
    const parts = clean.split(',');
    return {
      lat: parseFloat(parts[0]) || 0,
      lng: parseFloat(parts[1]) || 0,
    };
  }
}

// ============================================================================
//  BACKWARD CHAINING INFERENCE ENGINE
// ============================================================================

class InferenceEngine {
  constructor(ruleBase) {
    this.ruleBase = ruleBase;
  }

  /**
   * Main entry: Validate a route using backward chaining
   * @param {Object} routeData - OSRM route data with steps
   * @returns {Object} - { valid, warnings, ruleTrace }
   */
  validateRoute(routeData) {
    const currentTime = new Date();
    const steps = routeData.steps || [];
    
    // ===== BACKWARD CHAINING =====
    // GOAL: route_is_legal
    // Strategy: Break into sub-goals, check each one
    
    const ruleTrace = [];  // Trace of inference steps
    const warnings = [];
    
    ruleTrace.push({
      step: 'INIT',
      goal: 'route_is_legal',
      description: 'Bắt đầu suy diễn lùi: Kiểm tra tuyến đường có hợp pháp không?',
    });

    // Extract road segments from OSRM steps
    const segments = this._extractSegments(steps);
    
    ruleTrace.push({
      step: 'DECOMPOSE',
      goal: 'all_segments_legal',
      description: `Phân rã thành ${segments.length} đoạn đường cần kiểm tra`,
      segments: segments.map(s => s.roadName).filter(Boolean),
    });

    // Deduplication: track warned roads to avoid repeating same warning
    const warnedKeys = new Set();

    // Check each segment
    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      const prevSegment = i > 0 ? segments[i - 1] : null;

      // SUB-GOAL 1: Check one-way street
      const oneWayResult = this._checkOneWay(segment, ruleTrace);
      if (oneWayResult) {
        const key = `oneway:${oneWayResult.road}`;
        if (!warnedKeys.has(key)) {
          warnedKeys.add(key);
          warnings.push(oneWayResult);
        }
      }

      // SUB-GOAL 2: Check time restrictions
      const timeResult = this._checkTimeRestriction(segment, currentTime, ruleTrace);
      if (timeResult) {
        const key = `time:${timeResult.road}`;
        if (!warnedKeys.has(key)) {
          warnedKeys.add(key);
          warnings.push(timeResult);
        }
      }

      // SUB-GOAL 3: Check turn restrictions
      if (prevSegment) {
        const turnResult = this._checkTurnRestriction(prevSegment, segment, ruleTrace);
        if (turnResult) {
          const key = `turn:${turnResult.road}->${turnResult.toRoad}`;
          if (!warnedKeys.has(key)) {
            warnedKeys.add(key);
            warnings.push(turnResult);
          }
        }
      }
    }

    // Final conclusion
    const isLegal = warnings.length === 0;
    ruleTrace.push({
      step: 'CONCLUDE',
      goal: 'route_is_legal',
      result: isLegal,
      description: isLegal
        ? '✅ Tuyến đường hợp pháp - Không vi phạm luật giao thông nào'
        : `⚠️ Phát hiện ${warnings.length} cảnh báo vi phạm luật giao thông`,
    });

    return {
      valid: isLegal,
      warnings,
      ruleTrace,
      totalRulesChecked: ruleTrace.length,
    };
  }

  // --- Sub-goal checkers ---

  /**
   * SUB-GOAL: Check if segment violates one-way street rules
   */
  _checkOneWay(segment, ruleTrace) {
    if (!segment.roadName || !segment.startLat) return null;

    const normalizedName = this.ruleBase._normalizeRoadName(segment.roadName);
    
    // Find matching one-way rules for this road
    const matchingRules = this.ruleBase.oneWayStreets.filter(ow =>
      this._fuzzyMatch(normalizedName, ow.roadName)
    );

    if (matchingRules.length === 0) {
      ruleTrace.push({
        step: 'CHECK_ONE_WAY',
        road: segment.roadName,
        result: true,
        description: `✓ ${segment.roadName}: Không phải đường một chiều hoặc không có trong dữ liệu`,
      });
      return null;
    }

    // Check if segment direction matches allowed direction
    for (const rule of matchingRules) {
      // Check if this specific segment of the one-way road is relevant
      // by checking proximity of coordinates
      const isNearby = this._isSegmentNearOneWayRule(segment, rule);
      if (!isNearby) continue;

      // Check direction: allowed direction is from start → end
      const allowedBearing = this._bearing(rule.startLat, rule.startLng, rule.endLat, rule.endLng);
      const segmentBearing = this._bearing(segment.startLat, segment.startLng, segment.endLat, segment.endLng);
      
      const bearingDiff = Math.abs(allowedBearing - segmentBearing);
      const normalizedDiff = bearingDiff > 180 ? 360 - bearingDiff : bearingDiff;

      // If bearing difference > 120 degrees, likely going wrong way
      if (normalizedDiff > 120) {
        ruleTrace.push({
          step: 'CHECK_ONE_WAY',
          road: segment.roadName,
          result: false,
          description: `✗ ${segment.roadName}: Đi ngược chiều đường một chiều!`,
          rule: `Đường ${segment.roadName} chỉ cho phép đi theo hướng từ tọa độ (${rule.startLat},${rule.startLng}) đến (${rule.endLat},${rule.endLng})`,
        });

        return {
          type: 'ONE_WAY_VIOLATION',
          severity: 'high',
          road: segment.roadName,
          message: `Đi ngược chiều đường một chiều trên ${segment.roadName}`,
          law: 'Luật đường một chiều - Đà Nẵng',
          location: { lat: segment.startLat, lng: segment.startLng },
        };
      }
    }

    ruleTrace.push({
      step: 'CHECK_ONE_WAY',
      road: segment.roadName,
      result: true,
      description: `✓ ${segment.roadName}: Đường một chiều - đi đúng hướng`,
    });
    return null;
  }

  /**
   * SUB-GOAL: Check if segment has time-based restrictions
   */
  _checkTimeRestriction(segment, currentTime, ruleTrace) {
    if (!segment.roadName) return null;

    const normalizedName = this.ruleBase._normalizeRoadName(segment.roadName);
    const hour = currentTime.getHours();
    const minute = currentTime.getMinutes();
    const currentMinutes = hour * 60 + minute;

    const matchingRules = this.ruleBase.timeRestrictions.filter(tr =>
      this._fuzzyMatch(normalizedName, tr.roadName)
    );

    if (matchingRules.length === 0) {
      return null; // No time restrictions for this road
    }

    for (const rule of matchingRules) {
      // Only check restrictions relevant to personal vehicles
      const restriction = rule.restrictionType.toLowerCase();
      const isRelevant = restriction.includes('cấm xe cơ giới') || 
                         restriction.includes('tốc độ tối đa');
      
      if (!isRelevant) continue;

      for (const range of rule.timeRanges) {
        const startMinutes = range.startHour * 60 + range.startMin;
        const endMinutes = range.endHour * 60 + range.endMin;

        if (currentMinutes >= startMinutes && currentMinutes <= endMinutes) {
          ruleTrace.push({
            step: 'CHECK_TIME_RESTRICTION',
            road: segment.roadName,
            result: false,
            description: `✗ ${segment.roadName}: Vi phạm luật cấm theo giờ! (${rule.restrictionType} - ${rule.rawTimeStr})`,
          });

          return {
            type: 'TIME_RESTRICTION',
            severity: restriction.includes('cấm xe cơ giới') ? 'high' : 'medium',
            road: segment.roadName,
            message: `${rule.restrictionType} trên ${segment.roadName} (khung giờ: ${rule.rawTimeStr})`,
            law: 'Luật cấm theo khung giờ - Đà Nẵng',
            timeRange: rule.rawTimeStr,
            location: { lat: segment.startLat, lng: segment.startLng },
          };
        }
      }
    }

    ruleTrace.push({
      step: 'CHECK_TIME_RESTRICTION',
      road: segment.roadName,
      result: true,
      description: `✓ ${segment.roadName}: Không vi phạm luật cấm theo giờ tại thời điểm hiện tại`,
    });
    return null;
  }

  /**
   * SUB-GOAL: Check if turn from prevSegment to segment is legal
   */
  _checkTurnRestriction(prevSegment, segment, ruleTrace) {
    if (!prevSegment.roadName || !segment.roadName) return null;
    
    // Same road - no turn to check
    if (prevSegment.roadName === segment.roadName) return null;

    const fromRoad = this.ruleBase._normalizeRoadName(prevSegment.roadName);
    const toRoad = this.ruleBase._normalizeRoadName(segment.roadName);

    const matchingRules = this.ruleBase.turnRestrictions.filter(tr =>
      this._fuzzyMatch(fromRoad, tr.fromRoad) && this._fuzzyMatch(toRoad, tr.toRoad)
    );

    if (matchingRules.length === 0) {
      return null; // No turn restrictions at this intersection
    }

    // For simplicity, if there's any restriction between these two roads, warn
    for (const rule of matchingRules) {
      ruleTrace.push({
        step: 'CHECK_TURN_RESTRICTION',
        fromRoad: prevSegment.roadName,
        toRoad: segment.roadName,
        result: false,
        description: `✗ ${rule.rawType}: Từ ${prevSegment.roadName} → ${segment.roadName}`,
        rule: `${rule.rawType} khi đi từ ${prevSegment.roadName} vào ${segment.roadName}`,
      });

      return {
        type: 'TURN_RESTRICTION',
        severity: 'high',
        road: prevSegment.roadName,
        toRoad: segment.roadName,
        restrictionType: rule.rawType,
        message: `${rule.rawType} khi đi từ ${prevSegment.roadName} rẽ vào ${segment.roadName}`,
        law: 'Luật cấm quẹo - Đà Nẵng',
        location: { lat: segment.startLat, lng: segment.startLng },
      };
    }

    return null;
  }

  // --- Helper methods ---

  /**
   * Extract road segments from OSRM steps
   */
  _extractSegments(steps) {
    return steps
      .filter(step => step.name && step.name.trim() !== '')
      .map(step => ({
        roadName: step.name,
        startLat: step.maneuver?.location?.[1] || 0,
        startLng: step.maneuver?.location?.[0] || 0,
        endLat: step.intersections?.[step.intersections.length - 1]?.location?.[1] || step.maneuver?.location?.[1] || 0,
        endLng: step.intersections?.[step.intersections.length - 1]?.location?.[0] || step.maneuver?.location?.[0] || 0,
        distance: step.distance,
        duration: step.duration,
        modifier: step.maneuver?.modifier || '',
        maneuverType: step.maneuver?.type || '',
      }));
  }

  /**
   * Fuzzy match road names (handling Vietnamese variations)
   */
  _fuzzyMatch(name1, name2) {
    if (!name1 || !name2) return false;
    
    const n1 = name1.toLowerCase().normalize('NFC').trim();
    const n2 = name2.toLowerCase().normalize('NFC').trim();

    // Exact match
    if (n1 === n2) return true;

    // Remove common prefixes and compare
    const strip = (s) => s
      .replace(/^(đường|cầu|hầm chui|đại lộ|ga đi|ga đến)\s+/i, '')
      .trim();

    const s1 = strip(n1);
    const s2 = strip(n2);

    if (s1 === s2) return true;

    // One contains the other
    if (s1.includes(s2) || s2.includes(s1)) return true;

    return false;
  }

  /**
   * Check if segment coordinates are near a one-way rule's coordinates
   */
  _isSegmentNearOneWayRule(segment, rule) {
    const threshold = 0.005; // ~500m
    
    const segMidLat = (segment.startLat + segment.endLat) / 2;
    const segMidLng = (segment.startLng + segment.endLng) / 2;
    const ruleMidLat = (rule.startLat + rule.endLat) / 2;
    const ruleMidLng = (rule.startLng + rule.endLng) / 2;

    return Math.abs(segMidLat - ruleMidLat) < threshold && 
           Math.abs(segMidLng - ruleMidLng) < threshold;
  }

  /**
   * Calculate bearing between two coordinates
   */
  _bearing(lat1, lng1, lat2, lng2) {
    const toRad = (deg) => deg * Math.PI / 180;
    const toDeg = (rad) => rad * 180 / Math.PI;

    const dLng = toRad(lng2 - lng1);
    const y = Math.sin(dLng) * Math.cos(toRad(lat2));
    const x = Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) -
              Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(dLng);
    
    return (toDeg(Math.atan2(y, x)) + 360) % 360;
  }
}

// ============================================================================
//  OSRM ROUTE FETCHER
// ============================================================================

/**
 * Fetch real driving route from OSRM public API
 */
async function fetchOSRMRoute(originLat, originLng, destLat, destLng) {
  const url = `https://router.project-osrm.org/route/v1/driving/${originLng},${originLat};${destLng},${destLat}?overview=full&geometries=geojson&steps=true&annotations=true`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`OSRM API error: ${response.status}`);
  }

  const data = await response.json();
  if (data.code !== 'Ok' || !data.routes || data.routes.length === 0) {
    throw new Error('No route found by OSRM');
  }

  const route = data.routes[0];
  return {
    geometry: route.geometry,   // GeoJSON LineString
    distance: route.distance,   // meters
    duration: route.duration,   // seconds
    steps: route.legs[0].steps, // Turn-by-turn steps
  };
}

// ============================================================================
//  MAIN EXPORT: Route + ES Validation
// ============================================================================

// Singleton rule base (loaded once at startup)
const ruleBase = new RuleBase();
const engine = new InferenceEngine(ruleBase);

/**
 * Initialize the Expert System (call once at server startup)
 */
async function initExpertSystem() {
  await ruleBase.load();
  return { ruleBase, engine };
}

/**
 * Find and validate route from origin to destination
 * @param {number} originLat
 * @param {number} originLng
 * @param {number} destLat
 * @param {number} destLng
 * @returns {Object} Route + ES validation results
 */
async function findOptimalRoute(originLat, originLng, destLat, destLng) {
  if (!ruleBase.loaded) {
    await ruleBase.load();
  }

  // Step 1: Fetch real route from OSRM
  const routeData = await fetchOSRMRoute(originLat, originLng, destLat, destLng);

  // Step 2: Run backward chaining inference engine
  const esResult = engine.validateRoute(routeData);

  // Step 3: Build response with steps in Vietnamese
  const steps = routeData.steps
    .filter(s => s.name && s.distance > 0)
    .map(s => ({
      instruction: _buildVietnameseInstruction(s),
      road: s.name,
      distance: s.distance,
      duration: s.duration,
      maneuver: s.maneuver?.type || '',
      modifier: s.maneuver?.modifier || '',
      location: s.maneuver?.location || [],
    }));

  return {
    route: routeData.geometry,
    distance: routeData.distance,
    duration: routeData.duration,
    steps,
    esValidation: esResult,
  };
}

/**
 * Build Vietnamese instruction from OSRM step
 */
function _buildVietnameseInstruction(step) {
  const type = step.maneuver?.type || '';
  const modifier = step.maneuver?.modifier || '';
  const road = step.name || 'đường không tên';
  const distKm = (step.distance / 1000).toFixed(1);

  const modMap = {
    'left': 'rẽ trái',
    'right': 'rẽ phải',
    'slight left': 'rẽ nhẹ sang trái',
    'slight right': 'rẽ nhẹ sang phải',
    'sharp left': 'rẽ gấp sang trái',
    'sharp right': 'rẽ gấp sang phải',
    'straight': 'đi thẳng',
    'uturn': 'quay đầu',
  };

  const typeMap = {
    'depart': `Xuất phát trên ${road}`,
    'arrive': `Đến nơi - ${road}`,
    'turn': `${modMap[modifier] || modifier} vào ${road}`,
    'new name': `Tiếp tục trên ${road}`,
    'merge': `Nhập vào ${road}`,
    'fork': `${modMap[modifier] || 'rẽ'} vào ${road}`,
    'roundabout': `Đi qua vòng xuyến, ra ${road}`,
    'end of road': `Cuối đường, ${modMap[modifier] || ''} vào ${road}`,
    'continue': `Tiếp tục trên ${road}`,
  };

  const instruction = typeMap[type] || `${modMap[modifier] || type} trên ${road}`;
  return `${instruction} (${distKm} km)`;
}

module.exports = {
  initExpertSystem,
  findOptimalRoute,
  RuleBase,
  InferenceEngine,
};

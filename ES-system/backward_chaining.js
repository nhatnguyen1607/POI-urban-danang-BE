const FuzzyLogic = require('./fuzzy_logic');

// ============================================================================
//  BACKWARD CHAINING INFERENCE ENGINE
// ============================================================================

class InferenceEngine {
  constructor(reteNetwork) {
    this.reteNetwork = reteNetwork;
  }

  validateRoute(routeData, weather) {
    console.log(`\n=================== BẮT ĐẦU ĐÁNH GIÁ LỘ TRÌNH ===================`);
    const currentTime = new Date();
    const steps = routeData.steps || [];
    
    const ruleTrace = [];
    const warnings = [];
    const fuzzyInsights = [];
    
    ruleTrace.push({
      step: 'INIT',
      goal: 'route_is_legal',
      description: 'Bắt đầu suy diễn lùi qua Rete Network',
    });

    const segments = this._extractSegments(steps);
    const warnedKeys = new Set();

    for (let i = 0; i < segments.length; i++) {
      const segment = segments[i];
      const prevSegment = i > 0 ? segments[i - 1] : null;

      // 1. One-way
      const oneWayResult = this._checkOneWay(segment, ruleTrace);
      if (oneWayResult) {
        const key = `oneway:${oneWayResult.road}`;
        if (!warnedKeys.has(key)) {
          warnedKeys.add(key);
          warnings.push(oneWayResult);
        }
      }

      // 2. Time restrictions + Fuzzy Logic
      const timeResult = this._checkTimeRestriction(segment, currentTime, ruleTrace, fuzzyInsights, weather);
      if (timeResult) {
        const key = `time:${timeResult.road}`;
        if (!warnedKeys.has(key)) {
          warnedKeys.add(key);
          warnings.push(timeResult);
        }
      }

      // 3. Turn restrictions
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

    const isLegal = warnings.length === 0;
    return {
      valid: isLegal,
      warnings,
      ruleTrace,
      fuzzyInsights, // Thêm insights mờ
      totalRulesChecked: ruleTrace.length,
    };
  }

  // --- Sub-goal checkers ---

  _checkOneWay(segment, ruleTrace) {
    if (!segment.roadName || !segment.startLat) return null;

    // Sử dụng Rete Network thay vì duyệt qua toàn bộ RuleBase
    const matchingRules = this.reteNetwork.matchOneWayStreet(segment.roadName);

    if (matchingRules.length === 0) {
      ruleTrace.push({ step: 'CHECK_ONE_WAY', road: segment.roadName, result: true, description: `✓ ${segment.roadName}: Không thuộc đường một chiều` });
      return null;
    }

    if (segment.distance < 15) {
      ruleTrace.push({ step: 'CHECK_ONE_WAY', road: segment.roadName, result: true, description: `✓ ${segment.roadName}: Bỏ qua kiểm tra ngược chiều (đoạn chuyển tiếp quá ngắn: ${segment.distance}m)` });
      return null;
    }

    // Kiểm tra TẤT CẢ các đoạn one-way gần đó
    // Trên đường chia dải (divided highway), mỗi làn là 1 đoạn one-way riêng
    // Nếu CÓ bất kỳ đoạn nào cùng hướng → xe đang ở làn đúng → OK
    let hasMatchingDirection = false;
    let worstViolation = null;

    for (const rule of matchingRules) {
      const isNearby = this._isSegmentNearOneWayRule(segment, rule);
      if (!isNearby) continue;

      const allowedBearing = this._bearing(rule.startLat, rule.startLng, rule.endLat, rule.endLng);
      
      // Ưu tiên dùng bearing_after từ OSRM (hướng đi thực tế sau khi rẽ)
      const segmentBearing = segment.bearingAfter != null
        ? segment.bearingAfter
        : this._bearing(segment.startLat, segment.startLng, segment.endLat, segment.endLng);
      
      const bearingDiff = Math.abs(allowedBearing - segmentBearing);
      const normalizedDiff = bearingDiff > 180 ? 360 - bearingDiff : bearingDiff;

      console.log(`[ONE_WAY DEBUG] ${segment.roadName}: allowed=${allowedBearing.toFixed(1)} segment=${segmentBearing.toFixed(1)} diff=${normalizedDiff.toFixed(1)} dist=${segment.distance}m`);

      if (normalizedDiff <= 90) {
        // Có đoạn one-way cùng hướng → xe đang ở làn đúng
        hasMatchingDirection = true;
        break;
      }

      if (normalizedDiff > 150 && !worstViolation) {
        worstViolation = {
          type: 'ONE_WAY_VIOLATION',
          severity: 'high',
          road: segment.roadName,
          message: `Đi ngược chiều đường một chiều trên ${segment.roadName}`,
          location: { lat: segment.startLat, lng: segment.startLng },
        };
      }
    }

    // Chỉ báo lỗi nếu KHÔNG có đoạn nào cùng hướng
    if (!hasMatchingDirection && worstViolation) {
      ruleTrace.push({ step: 'CHECK_ONE_WAY', road: segment.roadName, result: false, description: `✗ ${segment.roadName}: Đi ngược chiều đường một chiều!` });
      return worstViolation;
    }
    
    ruleTrace.push({ step: 'CHECK_ONE_WAY', road: segment.roadName, result: true, description: `✓ ${segment.roadName}: Đi đúng hướng đường một chiều` });
    return null;
  }

  _checkTimeRestriction(segment, currentTime, ruleTrace, fuzzyInsights, weather) {
    if (!segment.roadName) return null;

    const matchingRules = this.reteNetwork.matchTimeRestriction(segment.roadName);
    if (matchingRules.length === 0) {
      ruleTrace.push({ step: 'CHECK_TIME_RESTRICTION', road: segment.roadName, result: true, description: `✓ ${segment.roadName}: Không có luật cấm giờ` });
      return null;
    }

    const hour = currentTime.getHours();
    const minute = currentTime.getMinutes();
    const currentMinutes = hour * 60 + minute;

    for (const rule of matchingRules) {
      // Ứng dụng Fuzzy Logic vào đây để đánh giá tuyến đường
      const fuzzyResult = FuzzyLogic.applyFuzzyTimeRestriction(rule, currentTime, segment.roadName, this.reteNetwork, weather);
      if (fuzzyResult.fuzzyLabel) {
        fuzzyInsights.push({
          road: segment.roadName,
          label: fuzzyResult.fuzzyLabel,
          score: fuzzyResult.score,
        });
      }

      // Luật crisp cho cấm cứng
      for (const range of rule.timeRanges) {
        const startMinutes = range.startHour * 60 + range.startMin;
        const endMinutes = range.endHour * 60 + range.endMin;

        if (currentMinutes >= startMinutes && currentMinutes <= endMinutes) {
          ruleTrace.push({ step: 'CHECK_TIME_RESTRICTION', road: segment.roadName, result: false, description: `✗ ${segment.roadName}: Vi phạm luật cấm giờ (${rule.restrictionType})` });
          return {
            type: 'TIME_RESTRICTION',
            severity: 'high',
            road: segment.roadName,
            message: `${rule.restrictionType} trên ${segment.roadName} (khung giờ: ${rule.rawTimeStr})`,
            timeRange: rule.rawTimeStr,
            location: { lat: segment.startLat, lng: segment.startLng },
          };
        }
      }
    }
    
    ruleTrace.push({ step: 'CHECK_TIME_RESTRICTION', road: segment.roadName, result: true, description: `✓ ${segment.roadName}: Hợp lệ trong thời điểm hiện tại` });
    return null;
  }

  _checkTurnRestriction(prevSegment, segment, ruleTrace) {
    if (!prevSegment.roadName || !segment.roadName || prevSegment.roadName === segment.roadName) return null;

    // O(1) hoặc O(K) Rete Network Query
    const rule = this.reteNetwork.matchTurnRestriction(prevSegment.roadName, segment.roadName);

    if (rule) {
      ruleTrace.push({ step: 'CHECK_TURN_RESTRICTION', result: false, description: `✗ ${rule.rawType}: Từ ${prevSegment.roadName} → ${segment.roadName}` });
      return {
        type: 'TURN_RESTRICTION',
        severity: 'high',
        road: prevSegment.roadName,
        toRoad: segment.roadName,
        restrictionType: rule.rawType,
        message: `${rule.rawType} khi đi từ ${prevSegment.roadName} rẽ vào ${segment.roadName}`,
        location: { lat: segment.startLat, lng: segment.startLng },
      };
    }
    
    ruleTrace.push({ step: 'CHECK_TURN_RESTRICTION', result: true, description: `✓ Rẽ hợp lệ: ${prevSegment.roadName} → ${segment.roadName}` });
    return null;
  }

  // Helper utils
  _extractSegments(steps) {
    return steps.filter(step => step.name && step.name.trim() !== '').map(step => ({
      roadName: step.name,
      distance: step.distance || 0,
      bearingAfter: step.maneuver?.bearing_after != null ? step.maneuver.bearing_after : null,
      startLat: step.maneuver?.location?.[1] || 0,
      startLng: step.maneuver?.location?.[0] || 0,
      endLat: step.intersections?.[step.intersections.length - 1]?.location?.[1] || step.maneuver?.location?.[1] || 0,
      endLng: step.intersections?.[step.intersections.length - 1]?.location?.[0] || step.maneuver?.location?.[0] || 0,
    }));
  }

  _isSegmentNearOneWayRule(segment, rule) {
    const threshold = 0.005; // ~500m
    const segMidLat = (segment.startLat + segment.endLat) / 2;
    const segMidLng = (segment.startLng + segment.endLng) / 2;
    const ruleMidLat = (rule.startLat + rule.endLat) / 2;
    const ruleMidLng = (rule.startLng + rule.endLng) / 2;
    return Math.abs(segMidLat - ruleMidLat) < threshold && Math.abs(segMidLng - ruleMidLng) < threshold;
  }

  _bearing(lat1, lng1, lat2, lng2) {
    const toRad = (deg) => deg * Math.PI / 180;
    const toDeg = (rad) => rad * 180 / Math.PI;
    const dLng = toRad(lng2 - lng1);
    const y = Math.sin(dLng) * Math.cos(toRad(lat2));
    const x = Math.cos(toRad(lat1)) * Math.sin(toRad(lat2)) - Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(dLng);
    return (toDeg(Math.atan2(y, x)) + 360) % 360;
  }
}

module.exports = InferenceEngine;

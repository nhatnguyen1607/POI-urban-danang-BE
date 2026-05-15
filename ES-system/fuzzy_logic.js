// ============================================================================
//  FUZZY LOGIC - Hệ mờ đa biến (Multi-variable Fuzzy Inference System)
//  Biến đầu vào:
//    1. Giờ hiện tại (so với đỉnh cao điểm, phụ thuộc loại đường)
//    2. Mật độ POI thực tế trên đường (suy luận từ dữ liệu)
//    3. Lượng mưa từ API Thời tiết
// ============================================================================

class FuzzyLogic {
  static evaluateTrafficCondition(currentTime, segmentId, poiDensity, weatherData = { rain_1h: 0 }, roadClass = 'secondary') {
    const hour = currentTime.getHours() + currentTime.getMinutes() / 60;
    const rain = weatherData.rain_1h;

    // ===== BIẾN ĐẦU VÀO 1: Giờ cao điểm (Peak Hour) =====
    const peak = this._getPeakConfig(roadClass);

    const peak1Degree = this.degreeOfMembership(hour,
      peak.p1s - 1, peak.p1s, peak.p1e, peak.p1e + 1
    );
    const peak2Degree = this.degreeOfMembership(hour,
      peak.p2s - 1, peak.p2s, peak.p2e, peak.p2e + 1
    );
    const muPeakHour = Math.max(peak1Degree, peak2Degree);

    // ===== BIẾN ĐẦU VÀO 2: Mật độ POI (POI Density) =====
    // Hàm thành viên hình thang cho mật độ POI
    const muPoiLow    = this.degreeOfMembership(poiDensity, 0, 0, 5, 15);
    const muPoiMedium = this.degreeOfMembership(poiDensity, 10, 20, 35, 50);
    const muPoiHigh   = this.degreeOfMembership(poiDensity, 35, 50, 300, 500);

    // ===== BIẾN ĐẦU VÀO 3: Thời tiết (Weather) =====
    const muRainLight = this.degreeOfMembership(rain, 0, 0, 2, 5);
    const muRainHeavy = this.degreeOfMembership(rain, 2, 5, 50, 100); 

    // ===== LUẬT MỜ TỔNG HỢP (Fuzzy Rule Base) =====
    // Toán tử AND = MIN, OR = MAX (Mamdani)
    const r1 = Math.min(muPeakHour, muPoiHigh, muRainHeavy);       // Cao điểm + POI cao + Mưa to → CỰC CAO
    const r2 = Math.min(muPeakHour, muPoiHigh, muRainLight);        // Cao điểm + POI cao + Mưa nhẹ → CAO
    const r3 = Math.min(muPeakHour, muPoiHigh, 1 - muRainHeavy);    // Cao điểm + POI cao + Không mưa → CAO
    const r4 = Math.min(muPeakHour, muPoiMedium);                   // Cao điểm + POI vừa → VỪA
    const r5 = Math.min(1 - muPeakHour, muPoiHigh, muRainHeavy);    // Ngoài giờ + POI cao + Mưa to → CAO
    const r6 = Math.min(1 - muPeakHour, muPoiHigh);                 // Ngoài giờ + POI cao → VỪA
    const r7 = Math.min(muPeakHour, muPoiLow);                      // Cao điểm + POI thấp → VỪA (do mật độ thấp nhưng rơi vào cao điểm)
    const r8 = Math.min(1 - muPeakHour, muPoiLow);                  // Ngoài giờ + POI thấp → THẤP
    const r9 = Math.min(1 - muPeakHour, muPoiMedium);               // Ngoài giờ + POI vừa → THẤP

    // Aggregation: gộp kết quả luật theo mức output
    const muOutHigh   = Math.max(r1, r2, r3, r5);
    const muOutMedium = Math.max(r4, r6, r7);
    const muOutLow    = Math.max(r8, r9);

    // ===== PHI MỜ HÓA (Centroid Defuzzification) =====
    const defuzzifiedValue = this.defuzzifyCentroid(muOutLow, muOutMedium, muOutHigh);

    const classMap = { 'primary': 'Trục chính/Quốc lộ', 'secondary': 'Đường nội thành', 'residential': 'Đường dân cư' };
    const classVn = classMap[roadClass] || roadClass;

    console.log(`\n--- [MỜ HOÁ] ${segmentId} | POI: ${poiDensity} | Loại: ${classVn} ---`);
    console.log(`  Giờ: ${hour.toFixed(2)}H | Mưa: ${rain}mm | μPeak=${muPeakHour.toFixed(2)} | μPOI(L,M,H)=${muPoiLow.toFixed(2)},${muPoiMedium.toFixed(2)},${muPoiHigh.toFixed(2)} | μRain(L,H)=${muRainLight.toFixed(2)},${muRainHeavy.toFixed(2)} → Điểm: ${defuzzifiedValue.toFixed(1)}/100`);

    if (defuzzifiedValue > 70) {
      return { level: 'cao', label: `Khả năng kẹt xe cao (${poiDensity} POI khu vực)`, score: defuzzifiedValue };
    } else if (defuzzifiedValue > 30) {
      return { level: 'vừa', label: `Giao thông hơi đông đúc (${poiDensity} POI khu vực)`, score: defuzzifiedValue };
    }
    return { level: 'thấp', label: 'Giao thông thông thoáng', score: defuzzifiedValue };
  }

  /**
   * primary (QL, trục chính): cao điểm dài, traffic cao
   * secondary (đường nội thành): cao điểm chuẩn
   * residential (đường dân cư): cao điểm ngắn
   */
  static _getPeakConfig(roadClass) {
    switch (roadClass) {
      case 'primary':
        return { p1s: 6.5, p1e: 9.0, p2s: 16.0, p2e: 19.0 };
      case 'secondary':
        return { p1s: 7.0, p1e: 8.5, p2s: 16.5, p2e: 18.5 };
      default: // residential
        return { p1s: 7.0, p1e: 8.0, p2s: 17.0, p2e: 18.0 };
    }
  }

  static defuzzifyCentroid(degreeLow, degreeMedium, degreeHigh) {
    let numerator = 0;
    let denominator = 0;
    
    for (let x = 0; x <= 100; x += 1) {
      const outLow = this.degreeOfMembership(x, -20, 0, 20, 40);
      const outMedium = this.degreeOfMembership(x, 20, 40, 60, 80);
      const outHigh = this.degreeOfMembership(x, 60, 80, 100, 120);
      
      const aggLow = Math.min(outLow, degreeLow);
      const aggMedium = Math.min(outMedium, degreeMedium);
      const aggHigh = Math.min(outHigh, degreeHigh);
      
      const maxMu = Math.max(aggLow, aggMedium, aggHigh);
      
      numerator += x * maxMu;
      denominator += maxMu;
    }
    
    if (denominator === 0) return 0;
    return numerator / denominator;
  }

  static degreeOfMembership(x, a, b, c, d) {
    if (x <= a || x >= d) return 0;
    if (x >= b && x <= c) return 1;
    if (x > a && x < b) return (x - a) / (b - a);
    if (x > c && x < d) return (d - x) / (d - c);
    return 0;
  }
}

module.exports = FuzzyLogic;

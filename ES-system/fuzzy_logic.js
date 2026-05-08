// ============================================================================
//  FUZZY LOGIC - Đánh giá giao thông bằng logic mờ (Định tính)
// ============================================================================

class FuzzyLogic {
  /**
   * Đánh giá mức độ ùn tắc hiện tại trên đoạn đường
   * Output: "Khu vực hơi đông", "Khả năng kẹt xe cao", "Bình thường"
   */
  static evaluateTrafficCondition(currentTime, segmentId, reteNetwork, restrictionType = '') {
    const hour = currentTime.getHours() + currentTime.getMinutes() / 60;
    
    // 1. Xin Rete Network thông tin POI của đoạn đường này
    const profile = reteNetwork.getProfile(segmentId);
    
    let rushDegree = 0;

    // 2. Tính độ mờ dựa trên khung giờ ĐỘNG từ file CSV
    let peak1Degree = this.degreeOfMembership(hour, 
        profile.peak_1_start - 1, profile.peak_1_start, 
        profile.peak_1_end, profile.peak_1_end + 1
    );

    let peak2Degree = 0;
    if(profile.peak_2_start > 0) {
        peak2Degree = this.degreeOfMembership(hour, 
            profile.peak_2_start - 1, profile.peak_2_start, 
            profile.peak_2_end, profile.peak_2_end + 1
        );
    }

    // Lấy mức độ cao nhất
    let rawRushDegree = Math.max(peak1Degree, peak2Degree);
    
    // Scale mức độ kẹt xe thực tế theo độ sầm uất (density_level)
    // Đường High sầm uất -> kẹt cứng (1.0)
    // Đường Medium -> đông đúc (0.75)
    // Đường Low -> hơi chậm (0.5)
    if (profile.density_level === 'Medium') {
        rushDegree = rawRushDegree * 0.75;
    } else if (profile.density_level === 'Low') {
        rushDegree = rawRushDegree * 0.5;
    } else {
        rushDegree = rawRushDegree; // High
    }
    
    // 3. Phi mờ hóa (Defuzzification) - Tính trọng tâm
    // Giả sử có 3 mức độ kẹt xe: Thấp (0-40), Vừa (20-80), Cao (60-100)
    // Ta lấy degree của từng mức dựa trên rushDegree tổng
    const muHigh = Math.max(0, rushDegree - 0.5) * 2; 
    const muMedium = rushDegree > 0.3 && rushDegree <= 0.75 ? 1 : (rushDegree > 0.75 ? 1 - (rushDegree-0.75)*4 : rushDegree * 3);
    const muLow = Math.max(0, 1 - rushDegree * 2);
    
    const defuzzifiedValue = this.defuzzifyCentroid(muLow, muMedium, muHigh);

    // In giá trị mờ hóa và phi mờ hóa ra console với format dễ nhìn
    console.log(`\n--- [MỜ HOÁ] Đoạn đường: ${segmentId} (${profile.density_level}) ---`);
    console.log(`Giờ: ${hour.toFixed(2)}H | Độ kẹt (Rush): ${rushDegree.toFixed(2)} | Điểm trọng tâm: ${defuzzifiedValue.toFixed(1)}/100`);

    // 4. Kết luận
    if (defuzzifiedValue > 70) {
      return { level: 'cao', label: 'Khả năng kẹt xe cao (Khu vực nhiều nhà hàng/cafe)', score: defuzzifiedValue };
    } else if (defuzzifiedValue > 30) {
      return { level: 'vừa', label: 'Giao thông hơi đông đúc', score: defuzzifiedValue };
    }
    return { level: 'thấp', label: 'Giao thông thông thoáng', score: defuzzifiedValue };
  }

  /**
   * Tính trọng tâm (Centroid Defuzzification)
   */
  static defuzzifyCentroid(degreeLow, degreeMedium, degreeHigh) {
    let numerator = 0;
    let denominator = 0;
    
    for (let x = 0; x <= 100; x += 5) {
      // Membership cho các mức độ đầu ra (Từ 0 đến 100 điểm)
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

  /**
   * Hàm thành viên hình thang mờ (Trapezoidal Membership Function)
   */
  static degreeOfMembership(x, a, b, c, d) {
    if (x <= a || x >= d) return 0;
    if (x >= b && x <= c) return 1;
    if (x > a && x < b) return (x - a) / (b - a);
    if (x > c && x < d) return (d - x) / (d - c);
    return 0;
  }
  
  /**
   * Tính toán và đưa ra kết luận mờ dựa trên luật cấm
   */
  static applyFuzzyTimeRestriction(rule, currentTime, segmentId, reteNetwork) {
    const trafficEval = this.evaluateTrafficCondition(currentTime, segmentId, reteNetwork, rule.restrictionType);
    return {
      warning: true,
      fuzzyLabel: trafficEval.label,
      score: trafficEval.score,
      rawRule: rule,
    };
  }
}

module.exports = FuzzyLogic;

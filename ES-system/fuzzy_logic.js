// ============================================================================
//  FUZZY LOGIC - Đánh giá giao thông bằng logic mờ (Định tính)
// ============================================================================

class FuzzyLogic {
  /**
   * Đánh giá mức độ ùn tắc hiện tại trên đoạn đường
   * Output: "Khu vực hơi đông", "Khả năng kẹt xe cao", "Bình thường"
   */
  static evaluateTrafficCondition(currentTime, restrictionType = '') {
    const hour = currentTime.getHours();
    
    // Membership function: Giờ cao điểm
    const isMorningRush = this.degreeOfMembership(hour, 6, 7, 8, 9);
    const isEveningRush = this.degreeOfMembership(hour, 16, 17, 18, 19);
    
    const rushDegree = Math.max(isMorningRush, isEveningRush);
    
    if (rushDegree > 0.7) {
      return { level: 'cao', label: 'Tuyến đường này hiện tại có khả năng kẹt xe cao (Giờ cao điểm)' };
    } else if (rushDegree > 0.3) {
      return { level: 'vừa', label: 'Khu vực hơi đông, có thể di chuyển chậm' };
    }
    return { level: 'thấp', label: 'Giao thông bình thường' };
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
  static applyFuzzyTimeRestriction(rule, currentTime) {
    // Thay vì cấm tuyệt đối, ta có thể dùng hệ logic mờ để đưa ra nhãn cảnh báo
    const trafficEval = this.evaluateTrafficCondition(currentTime, rule.restrictionType);
    return {
      warning: true,
      fuzzyLabel: trafficEval.label,
      rawRule: rule,
    };
  }
}

module.exports = FuzzyLogic;

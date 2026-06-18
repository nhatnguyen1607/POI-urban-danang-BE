const { normalizeText } = require('./poiDataService');
const { detectIntent } = require('./intentService');

function unique(values) {
  return Array.from(new Set(values.filter(Boolean)));
}

function includesAny(text, terms) {
  return terms.some((term) => text.includes(normalizeText(term)));
}

function parseBusinessConcept(concept = '', language = 'vi') {
  const normalized = normalizeText(concept);
  const intent = detectIntent(concept);
  const businessType =
    intent?.label ||
    (includesAny(normalized, ['restaurant', 'nha hang']) ? 'Restaurant' : language === 'en' ? 'General business' : 'Kinh doanh tổng quát');

  const targetCustomers = [];
  if (includesAny(normalized, ['student', 'sinh vien', 'hoc bai', 'university', 'dai hoc'])) {
    targetCustomers.push(language === 'en' ? 'Students' : 'Sinh viên');
  }
  if (includesAny(normalized, ['tourist', 'du lich', 'traveler', 'khach du lich', 'gan bien', 'beach'])) {
    targetCustomers.push(language === 'en' ? 'Tourists' : 'Khách du lịch');
  }
  if (includesAny(normalized, ['family', 'gia dinh'])) targetCustomers.push(language === 'en' ? 'Families' : 'Gia đình');
  if (includesAny(normalized, ['office', 'van phong'])) targetCustomers.push(language === 'en' ? 'Office workers' : 'Dân văn phòng');

  let budgetLevel = language === 'en' ? 'Unspecified' : 'Chưa rõ';
  if (includesAny(normalized, ['binh dan', 'gia re', 'budget', 'low cost', 'tiet kiem'])) budgetLevel = language === 'en' ? 'Budget' : 'Tiết kiệm';
  if (includesAny(normalized, ['cao cap', 'premium', 'luxury'])) budgetLevel = language === 'en' ? 'Premium' : 'Cao cấp';
  if (includesAny(normalized, ['vua phai', 'moderate', 'mid range'])) budgetLevel = language === 'en' ? 'Moderate' : 'Vừa phải';

  const priorities = [];
  if (includesAny(normalized, ['quiet', 'yen tinh', 'hoc bai'])) priorities.push(language === 'en' ? 'Quiet study-friendly space' : 'Không gian yên tĩnh/học bài');
  if (includesAny(normalized, ['parking', 'do xe'])) priorities.push(language === 'en' ? 'Parking availability' : 'Có chỗ đỗ xe');
  if (includesAny(normalized, ['beach', 'gan bien', 'bien'])) priorities.push(language === 'en' ? 'Near beach/tourist flow' : 'Gần biển/luồng khách du lịch');
  if (includesAny(normalized, ['school', 'truong', 'university', 'dai hoc'])) priorities.push(language === 'en' ? 'Near schools/universities' : 'Gần trường học/đại học');
  if (includesAny(normalized, ['traffic', 'giao thong', 'access'])) priorities.push(language === 'en' ? 'Good accessibility' : 'Dễ tiếp cận/giao thông thuận tiện');

  return {
    businessType,
    targetCustomers: unique(targetCustomers.length ? targetCustomers : [language === 'en' ? 'Broad local market' : 'Khách địa phương rộng']),
    budgetLevel,
    priorities: unique(priorities.length ? priorities : [language === 'en' ? 'No explicit constraint detected' : 'Chưa phát hiện ràng buộc rõ']),
    parser: {
      mode: 'deterministic_keyword_parser',
      hallucinationRisk: 'low',
      note:
        language === 'en'
          ? 'Parsed from explicit concept keywords. No external facts are inferred.'
          : 'Tách từ khóa trực tiếp trong concept. Không suy diễn dữ kiện bên ngoài.',
    },
  };
}

module.exports = {
  parseBusinessConcept,
};

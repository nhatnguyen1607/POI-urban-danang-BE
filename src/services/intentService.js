const { normalizeText } = require('./poiDataService');

const INTENTS = [
  {
    id: 'cafe',
    label: 'Cafe',
    queryTerms: ['cafe', 'ca phe', 'coffee', 'coffe', 'dessert', 'tra sua', 'hoc bai', 'yen tinh'],
    categoryTerms: ['cafe', 'ca phe', 'coffee', 'dessert', 'tra sua'],
  },
  {
    id: 'seafood',
    label: 'Hai san',
    queryTerms: ['hai san', 'seafood'],
    categoryTerms: ['hai san', 'seafood', 'nha hang'],
  },
  {
    id: 'food',
    label: 'An uong',
    queryTerms: [
      'quan an', 'an vat', 'via he', 'mon an', 'binh dan', 'dac san',
      'am thuc', 'an toi', 'an sang', 'an trua', 'mon dia phuong', 'do an',
    ],
    categoryTerms: ['quan an', 'an vat', 'via he', 'food'],
  },
  {
    id: 'pub',
    label: 'Quan nhau',
    queryTerms: ['quan nhau', 'nhau', 'bia', 'beer', 'dem'],
    categoryTerms: ['quan nhau', 'nhau', 'bar', 'beer'],
  },
  {
    id: 'travel',
    label: 'Diem di choi',
    queryTerms: [
      'di choi', 'check in', 'tham quan', 'lich trinh', 'du lich',
      'dia diem', 'noi tieng', 'danh thang', 'di tich', 'ngam bien',
      'hoang hon', 'ngu hanh son',
    ],
    categoryTerms: [
      'diem du lich', 'bao tang', 'bai bien', 'park', 'cong vien',
      'attraction', 'di tich', 'danh thang', 'chua',
    ],
  },
];

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function normalizedBoundaryText(value) {
  return normalizeText(value).replace(/[^a-z0-9]+/g, ' ').trim();
}

function preparedTextIncludesTerm(text, term) {
  const normalizedTerm = normalizeText(term).replace(/[^a-z0-9]+/g, ' ').trim();
  if (!text || !normalizedTerm) return false;
  return new RegExp(`(?:^| )${escapeRegExp(normalizedTerm)}(?:$| )`).test(text);
}

function includesNormalizedTerm(value, term) {
  return preparedTextIncludesTerm(normalizedBoundaryText(value), term);
}

function detectIntent(query) {
  return detectIntents(query)[0] || null;
}

function detectIntents(query) {
  const normalized = normalizeText(query);
  const matches = INTENTS.map((intent) => {
    const score = intent.queryTerms.reduce(
      (sum, term) => sum + (includesNormalizedTerm(normalized, term) ? 1 : 0),
      0,
    );
    return { ...intent, score };
  })
    .filter((intent) => intent.score > 0)
    .sort((a, b) => b.score - a.score);

  return matches;
}

function categoryMatchScore(poi, intent) {
  if (!intent) return 0.5;
  const category = normalizedBoundaryText(poi.category);
  const text = normalizedBoundaryText(`${poi.name} ${poi.text}`);
  const categoryHit = intent.categoryTerms.some((term) => preparedTextIncludesTerm(category, term));
  const textHit = intent.queryTerms.some((term) => preparedTextIncludesTerm(text, term));
  if (categoryHit) return 1;
  if (textHit) return 0.72;
  return 0.12;
}

module.exports = {
  detectIntent,
  detectIntents,
  categoryMatchScore,
  INTENTS,
  includesNormalizedTerm,
};

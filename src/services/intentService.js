const { normalizeText } = require('./poiDataService');

const INTENTS = [
  {
    id: 'cafe',
    label: 'Cafe',
    queryTerms: ['cafe', 'ca phe', 'coffee', 'dessert', 'tra sua', 'hoc bai', 'yen tinh'],
    categoryTerms: ['cafe', 'ca phe', 'coffee', 'dessert', 'tra sua'],
  },
  {
    id: 'seafood',
    label: 'Hai san',
    queryTerms: ['hai san', 'seafood', 'gan bien', 'bien'],
    categoryTerms: ['hai san', 'seafood', 'nha hang'],
  },
  {
    id: 'food',
    label: 'An uong',
    queryTerms: ['quan an', 'an vat', 'via he', 'mon an', 'binh dan'],
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
    queryTerms: ['di choi', 'check in', 'tham quan', 'lich trinh', 'du lich'],
    categoryTerms: ['diem du lich', 'bao tang', 'bien', 'park'],
  },
];

function detectIntent(query) {
  return detectIntents(query)[0] || null;
}

function detectIntents(query) {
  const normalized = normalizeText(query);
  const matches = INTENTS.map((intent) => {
    const score = intent.queryTerms.reduce(
      (sum, term) => sum + (normalized.includes(normalizeText(term)) ? 1 : 0),
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
  const category = normalizeText(poi.category);
  const text = normalizeText(`${poi.name} ${poi.text}`);
  const categoryHit = intent.categoryTerms.some((term) => category.includes(normalizeText(term)));
  const textHit = intent.queryTerms.some((term) => text.includes(normalizeText(term)));
  if (categoryHit) return 1;
  if (textHit) return 0.72;
  return 0.12;
}

module.exports = {
  detectIntent,
  detectIntents,
  categoryMatchScore,
  INTENTS,
};

const { buildBusinessEvidencePack } = require('./businessEvidenceService');
const { parseBusinessConcept } = require('./businessConceptParserService');

function pct(value) {
  return `${Math.round((value || 0) * 100)}%`;
}

function topNames(items, limit = 3) {
  return items.slice(0, limit).map((item) => item.name || item.category).filter(Boolean);
}

function buildInsightPrompt({ concept, evidence, language }) {
  return {
    system:
      'You are a grounded data-to-text writer for Danang UrbanAgent AI. Only verbalize the provided EVIDENCE_JSON. Do not invent places, counts, scores, streets, or customer density.',
    user: {
      concept,
      language,
      evidence,
      outputRules: [
        'Use demand proxy wording, never real footfall wording.',
        'Mention missing evidence when route, rent, opening hour, or live traffic data is unavailable.',
        'Every major claim must be traceable to evidence ids.',
      ],
    },
  };
}

function generateLocalInsight(area, language = 'vi') {
  const evidence = area.evidence;
  const signals = evidence.signals;
  const topCategories = evidence.topCategories.slice(0, 3);
  const complementary = evidence.complementaryPOIs.slice(0, 4);
  const competitors = evidence.competitors.slice(0, 4);
  const routeWarnings = evidence.routeWarnings;
  const evidenceIds = [
    ...topCategories.map((item) => item.evidenceId),
    ...complementary.map((item) => item.evidenceId),
    ...competitors.map((item) => item.evidenceId),
    ...routeWarnings.map((item) => item.evidenceId),
  ];

  if (language === 'en') {
    return {
      summary: `This area scores ${area.score}/100. The opportunity is supported by a ${pct(signals.demandProxy)} demand proxy, ${pct(signals.conceptFit)} concept fit, and ${evidence.rawCounts.poiTotalInArea} POIs in the local cluster.`,
      area_potential: `The strongest category signals are ${topCategories.map((item) => `${item.category} (${item.count})`).join(', ') || 'not enough category evidence'}. This suggests the area already has commercial activity that can support the concept, but the signal remains a proxy from POI/review/rating data.`,
      complementary_poi_analysis: complementary.length
        ? `Complementary POIs include ${topNames(complementary).join(', ')}. These nearby categories can create cross-visits and make the area more useful for the target concept.`
        : 'There is not enough complementary POI evidence in the current CSV to make a strong claim.',
      risk_warnings: [
        competitors.length
          ? `Direct competition is visible: ${topNames(competitors).join(', ')}. Differentiation is required.`
          : 'Direct competition is not high in the current evidence pack.',
        ...routeWarnings.map((item) => item.warning),
      ],
      recommended_actions: [
        'Survey rent and frontage quality before deciding.',
        'Validate peak-hour accessibility and parking.',
        'Collect live opening-hour and footfall evidence before investment.',
      ],
      used_evidence_ids: evidenceIds,
      missing_evidence: ['real footfall', 'rent price', 'live traffic', 'opening hours'],
    };
  }

  return {
    summary: `Khu vực này đạt ${area.score}/100. Điểm mạnh đến từ demand proxy ${pct(signals.demandProxy)}, độ khớp concept ${pct(signals.conceptFit)} và ${evidence.rawCounts.poiTotalInArea} POI trong cụm dữ liệu hiện có.`,
    area_potential: `Các danh mục nổi bật là ${topCategories.map((item) => `${item.category} (${item.count})`).join(', ') || 'chưa đủ bằng chứng danh mục'}. Điều này cho thấy khu vực đã có hoạt động thương mại nền, nhưng đây vẫn là tín hiệu nhu cầu ước lượng từ POI/review/rating.`,
    complementary_poi_analysis: complementary.length
      ? `POI bổ trợ đáng chú ý gồm ${topNames(complementary).join(', ')}. Những điểm này có thể tạo luồng ghé chéo và làm concept dễ được phát hiện hơn.`
      : 'Dữ liệu CSV hiện chưa đủ POI bổ trợ để kết luận mạnh.',
    risk_warnings: [
      competitors.length
        ? `Có cạnh tranh trực tiếp: ${topNames(competitors).join(', ')}. Concept cần khác biệt hóa rõ.`
        : 'Cạnh tranh trực tiếp chưa cao trong evidence pack hiện tại.',
      ...routeWarnings.map((item) => item.warning),
    ],
    recommended_actions: [
      'Khảo sát giá thuê, mặt tiền và khả năng đỗ xe.',
      'Kiểm tra accessibility vào giờ cao điểm.',
      'Bổ sung dữ liệu giờ mở cửa, traffic sống và footfall thật trước khi ra quyết định đầu tư.',
    ],
    used_evidence_ids: evidenceIds,
    missing_evidence: ['mật độ khách thật', 'giá thuê', 'traffic thời gian thực', 'giờ mở cửa đầy đủ'],
  };
}

function collectAllowedTokens(area) {
  const evidence = area.evidence;
  return new Set([
    area.id,
    String(area.score),
    ...evidence.topCategories.flatMap((item) => [item.category, String(item.count), item.evidenceId]),
    ...evidence.complementaryPOIs.flatMap((item) => [item.name, item.category, item.poiId, item.evidenceId]),
    ...evidence.competitors.flatMap((item) => [item.name, item.category, item.poiId, item.evidenceId]),
    ...evidence.routeWarnings.flatMap((item) => [item.warning, item.evidenceId]),
    ...evidence.samplePOIs.flatMap((item) => [item.name, item.category, item.id, item.evidenceId]),
  ].filter(Boolean));
}

function validateInsight(area, insight) {
  const allowed = collectAllowedTokens(area);
  const usedIds = insight.used_evidence_ids || [];
  const unknownEvidenceIds = usedIds.filter((id) => !allowed.has(id));
  const unsupportedClaims = [];
  const text = JSON.stringify(insight).toLowerCase();
  ['mật độ khách thật', 'real footfall', 'doanh thu chắc chắn', 'guaranteed revenue'].forEach((phrase) => {
    if (text.includes(phrase) && !String(insight.missing_evidence || '').toLowerCase().includes(phrase)) {
      unsupportedClaims.push(`Unsafe claim phrase: ${phrase}`);
    }
  });

  return {
    hallucinationChecked: true,
    unknownEvidenceIds,
    unsupportedClaims,
    passed: unknownEvidenceIds.length === 0 && unsupportedClaims.length === 0,
  };
}

async function generateBusinessInsights({ concept, limit = 5, language = 'vi' }) {
  const pack = await buildBusinessEvidencePack({ concept, limit });
  const parsedConstraints = parseBusinessConcept(concept, language);
  const areas = pack.areas.map((area) => {
    const insight = generateLocalInsight(area, language);
    return {
      ...area,
      llmInsight: insight,
      insightPrompt: buildInsightPrompt({ concept, evidence: area.evidence, language }),
      guardrail: validateInsight(area, insight),
    };
  });

  return {
    ...pack,
    language,
    parsedConstraints,
    areas,
    stagePipeline: [
      { id: 1, name: 'Business Concept Input', status: 'complete' },
      { id: 2, name: 'Concept Parser', status: 'complete', output: parsedConstraints },
      { id: 3, name: 'Candidate Area Scorer', status: 'complete', safeguard: 'Mathematical scoring only; no LLM text.' },
      { id: 4, name: 'Evidence Pack Builder', status: 'complete', safeguard: 'Evidence IDs are attached to every table row and POI.' },
      { id: 5, name: 'Business Insight Generator', status: 'complete', safeguard: 'LLM/data-to-text may only interpret evidence JSON.' },
      { id: 6, name: 'Report Dashboard', status: 'ready' },
    ],
    guardrails: {
      hallucinationChecked: true,
      unsupportedClaims: areas.flatMap((area) => area.guardrail.unsupportedClaims),
      passed: areas.every((area) => area.guardrail.passed),
      mode: 'local_data_to_text',
      mathBeforeLlm: true,
      evidenceOnlyInterpretation: true,
      note: 'MVP uses deterministic grounded data-to-text. A real LLM can replace this layer after API credentials are configured.',
    },
  };
}

module.exports = {
  generateBusinessInsights,
  generateLocalInsight,
  validateInsight,
  buildInsightPrompt,
};

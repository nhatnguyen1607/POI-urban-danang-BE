const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.resolve(__dirname, '..');
const REPORT_FILE = path.join(ROOT_DIR, 'reports', 'evaluation', 'learning_loop_eval_latest.json');
const OUT_DIR = path.join(ROOT_DIR, 'reports', 'evaluation', 'figures');

const METRICS = [
  { key: 'exactPoiHitRate', label: 'Exact POI Hit Rate', good: 'higher' },
  { key: 'recallAtReturnedK', label: 'Recall@ReturnedK', good: 'higher' },
  { key: 'intentCoverage', label: 'Intent Coverage', good: 'higher' },
  { key: 'precisionAtExpectedK', label: 'Precision@ExpectedK', good: 'higher' },
  { key: 'f1', label: 'F1', good: 'higher' },
];

function readReport() {
  if (!fs.existsSync(REPORT_FILE)) {
    throw new Error(`Missing report: ${REPORT_FILE}. Run npm run evaluate:learning first.`);
  }
  return JSON.parse(fs.readFileSync(REPORT_FILE, 'utf8'));
}

function barChart(report) {
  const width = 1120;
  const height = 660;
  const chartX = 260;
  const chartY = 110;
  const chartW = 760;
  const rowH = 86;
  const beforeColor = '#64748b';
  const afterColor = '#22d3ee';
  const deltaColor = '#34d399';
  const text = [];

  text.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`);
  text.push('<rect width="100%" height="100%" fill="#020617"/>');
  text.push('<text x="48" y="54" fill="#f8fafc" font-family="Inter, Arial, sans-serif" font-size="30" font-weight="800">Danang UrbanAgent Learning Evaluation</text>');
  text.push(`<text x="48" y="84" fill="#94a3b8" font-family="Inter, Arial, sans-serif" font-size="15">Before vs After memory + reranker training | ${report.createdAt}</text>`);

  [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
    const x = chartX + tick * chartW;
    text.push(`<line x1="${x}" y1="${chartY - 20}" x2="${x}" y2="${chartY + rowH * METRICS.length}" stroke="#1e293b" stroke-width="1"/>`);
    text.push(`<text x="${x - 10}" y="${chartY - 32}" fill="#64748b" font-family="Inter, Arial, sans-serif" font-size="12">${tick.toFixed(2)}</text>`);
  });

  METRICS.forEach((metric, index) => {
    const y = chartY + index * rowH;
    const before = Number(report.before?.[metric.key] || 0);
    const after = Number(report.after?.[metric.key] || 0);
    const delta = Number(report.delta?.[metric.key] || 0);
    const beforeW = before * chartW;
    const afterW = after * chartW;
    text.push(`<text x="48" y="${y + 30}" fill="#e2e8f0" font-family="Inter, Arial, sans-serif" font-size="17" font-weight="700">${metric.label}</text>`);
    text.push(`<rect x="${chartX}" y="${y + 8}" width="${beforeW}" height="22" rx="5" fill="${beforeColor}" opacity="0.72"/>`);
    text.push(`<rect x="${chartX}" y="${y + 38}" width="${afterW}" height="22" rx="5" fill="${afterColor}" opacity="0.95"/>`);
    text.push(`<text x="${chartX + Math.max(beforeW, 4) + 10}" y="${y + 25}" fill="#cbd5e1" font-family="Inter, Arial, sans-serif" font-size="13">before ${before.toFixed(4)}</text>`);
    text.push(`<text x="${chartX + Math.max(afterW, 4) + 10}" y="${y + 55}" fill="#cffafe" font-family="Inter, Arial, sans-serif" font-size="13">after ${after.toFixed(4)}</text>`);
    text.push(`<text x="${chartX + chartW + 36}" y="${y + 44}" fill="${delta >= 0 ? deltaColor : '#fb7185'}" font-family="Inter, Arial, sans-serif" font-size="15" font-weight="700">Delta ${delta >= 0 ? '+' : ''}${delta.toFixed(4)}</text>`);
  });

  text.push(`<rect x="48" y="${height - 104}" width="${width - 96}" height="62" rx="14" fill="#0f172a" stroke="#1e293b"/>`);
  text.push(`<text x="72" y="${height - 76}" fill="#f8fafc" font-family="Inter, Arial, sans-serif" font-size="16" font-weight="700">Interpretation</text>`);
  text.push(`<text x="72" y="${height - 50}" fill="#cbd5e1" font-family="Inter, Arial, sans-serif" font-size="14">A useful agent should improve Exact POI Hit Rate, Recall, and maintain Intent Coverage after learning from persona memory and feedback.</text>`);
  text.push('</svg>');
  return text.join('\n');
}

function summaryCard(report) {
  const width = 1120;
  const height = 620;
  const cards = [
    { label: 'Exact POI Hit Rate', before: report.before.exactPoiHitRate, after: report.after.exactPoiHitRate },
    { label: 'Recall@ReturnedK', before: report.before.recallAtReturnedK, after: report.after.recallAtReturnedK },
    { label: 'Intent Coverage', before: report.before.intentCoverage, after: report.after.intentCoverage },
  ];
  const text = [];
  text.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">`);
  text.push('<rect width="100%" height="100%" fill="#020617"/>');
  text.push('<text x="56" y="62" fill="#f8fafc" font-family="Inter, Arial, sans-serif" font-size="34" font-weight="850">How We Know The Agent Learned</text>');
  text.push('<text x="56" y="96" fill="#94a3b8" font-family="Inter, Arial, sans-serif" font-size="16">Memory + feedback training should improve retrieval quality before any neural fine-tuning.</text>');
  cards.forEach((card, index) => {
    const x = 56 + index * 344;
    const delta = Number((card.after - card.before).toFixed(4));
    text.push(`<rect x="${x}" y="150" width="310" height="250" rx="22" fill="#0f172a" stroke="#1e293b"/>`);
    text.push(`<text x="${x + 24}" y="194" fill="#cbd5e1" font-family="Inter, Arial, sans-serif" font-size="17" font-weight="700">${card.label}</text>`);
    text.push(`<text x="${x + 24}" y="260" fill="#64748b" font-family="Inter, Arial, sans-serif" font-size="16">Before: ${Number(card.before).toFixed(4)}</text>`);
    text.push(`<text x="${x + 24}" y="318" fill="#22d3ee" font-family="Inter, Arial, sans-serif" font-size="44" font-weight="850">${Number(card.after).toFixed(4)}</text>`);
    text.push(`<text x="${x + 24}" y="360" fill="${delta >= 0 ? '#34d399' : '#fb7185'}" font-family="Inter, Arial, sans-serif" font-size="18" font-weight="800">Delta ${delta >= 0 ? '+' : ''}${delta.toFixed(4)}</text>`);
  });
  text.push('<rect x="56" y="448" width="1008" height="96" rx="18" fill="#082f49" stroke="#0e7490"/>');
  text.push('<text x="84" y="486" fill="#e0f2fe" font-family="Inter, Arial, sans-serif" font-size="17" font-weight="750">Research/product message</text>');
  text.push('<text x="84" y="520" fill="#bae6fd" font-family="Inter, Arial, sans-serif" font-size="15">The agent contribution is not just routing: it learns long-term user/persona preferences, then reranks Danang POIs with measurable before/after gains.</text>');
  text.push('</svg>');
  return text.join('\n');
}

function writeVisualizations() {
  const report = readReport();
  fs.mkdirSync(OUT_DIR, { recursive: true });
  const chartFile = path.join(OUT_DIR, 'learning_before_after.svg');
  const summaryFile = path.join(OUT_DIR, 'learning_summary_card.svg');
  fs.writeFileSync(chartFile, barChart(report), 'utf8');
  fs.writeFileSync(summaryFile, summaryCard(report), 'utf8');
  return { chartFile, summaryFile };
}

if (require.main === module) {
  try {
    const result = writeVisualizations();
    console.log(JSON.stringify(result, null, 2));
  } catch (error) {
    console.error(error);
    process.exit(1);
  }
}

module.exports = {
  writeVisualizations,
};

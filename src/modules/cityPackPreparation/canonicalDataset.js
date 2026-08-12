const crypto = require('node:crypto');
const fs = require('node:fs');

const { normalizeCategory, numberOrNull } = require('./sourceRecord');

const EXPECTED_CANONICAL_SHA =
  'e1f7fd635087eecb56dac8a2f3ed810f481ff129a12d19332e2d68f08ed56f96';

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];

    if (char === '"') {
      if (inQuotes && next === '"') {
        value += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (char === ',' && !inQuotes) {
      row.push(value);
      value = '';
      continue;
    }

    if ((char === '\n' || char === '\r') && !inQuotes) {
      if (char === '\r' && next === '\n') i += 1;
      row.push(value);
      if (row.some((cell) => cell !== '')) rows.push(row);
      row = [];
      value = '';
      continue;
    }

    value += char;
  }

  if (value || row.length > 0) {
    row.push(value);
    rows.push(row);
  }

  return rows;
}

function readCanonicalPois(canonicalPath) {
  const rows = parseCsv(fs.readFileSync(canonicalPath, 'utf8'));
  const headers = rows[0].map((header, index) =>
    index === 0 ? header.replace(/^\uFEFF/, '').trim() : header.trim(),
  );

  return rows.slice(1).map((cells) => {
    const record = Object.fromEntries(headers.map((header, index) => [header, cells[index] ?? '']));
    return {
      id: record.Global_ID,
      name: record['Restaurant Name'],
      category: normalizeCategory(record.Category_Normalized || record.Category),
      sourceCategory: record.Category,
      latitude: numberOrNull(record.Lat),
      longitude: numberOrNull(record.Lon),
      address: record.Address_Current || record.Address_Raw || '',
      district: record.District || record.District_Raw || '',
      source: record.Source,
    };
  });
}

function inspectCanonicalDataset(canonicalPath) {
  const rows = readCanonicalPois(canonicalPath);
  const digest = sha256(canonicalPath);

  return {
    path: canonicalPath,
    rows: rows.length,
    sha256: digest,
    shaMatchesExpected: digest === EXPECTED_CANONICAL_SHA,
  };
}

module.exports = {
  EXPECTED_CANONICAL_SHA,
  inspectCanonicalDataset,
  parseCsv,
  readCanonicalPois,
  sha256,
};

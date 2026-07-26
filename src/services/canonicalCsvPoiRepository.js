const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const csv = require('csv-parser');

const DATA_DIR = path.resolve(__dirname, '..', '..', 'data');
const DEFAULT_CITY_ID = 'da-nang';
const DEFAULT_CANONICAL_POI_CSV = path.join(DATA_DIR, 'canonical', 'urbanagent_poi_master_v1.csv');
const CANONICAL_POI_CSV_PATH = process.env.URBANAGENT_CANONICAL_POI_CSV || DEFAULT_CANONICAL_POI_CSV;

const DA_NANG_BBOX = {
  west: 107.8,
  south: 15.8,
  east: 108.5,
  north: 16.3,
};

const EXPECTED_CANONICAL_COLUMNS = [
  'Global_ID',
  'Alias_Global_IDs',
  'City_ID',
  'Entity_Type',
  'RestaurantID',
  'Source_IDs',
  'Restaurant Name',
  'District',
  'District_Raw',
  'Admin_Normalization_Status',
  'Address_Raw',
  'Address_Current',
  'Category',
  'Category_Normalized',
  'Lat',
  'Lon',
  'Overall Rating',
  'Rating_Scale',
  'Rating_Count',
  'Review_Sample_Count',
  'Google_Rating',
  'Google_Rating_Count',
  'Foody_Rating_10',
  'Foody_Review_Sample_Count',
  'Primary_Rating_Source',
  'Price',
  'Price_Min_VND',
  'Price_Max_VND',
  'Opening_Hours_Raw',
  'Image_URL',
  'LLM_Input_Text',
  'Source',
  'Merge_Status',
  'Data_Quality_Flags',
];

function normalizeText(value) {
  return String(value || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/\u0111/g, 'd')
    .replace(/\u0110/g, 'D')
    .toLowerCase();
}

function cleanString(value) {
  const text = String(value ?? '').trim();
  if (!text || ['nan', 'none', 'null', 'n/a'].includes(text.toLowerCase())) return '';
  return text;
}

function optionalNumber(value) {
  const text = cleanString(value).replace(',', '.');
  if (!text) return null;
  const n = Number.parseFloat(text);
  return Number.isFinite(n) ? n : null;
}

function parseList(value) {
  const text = cleanString(value);
  if (!text) return [];
  return text
    .split(/[|;]/)
    .map((item) => cleanString(item))
    .filter(Boolean);
}

function parseImageUrls(value) {
  const seen = new Set();
  return cleanString(value)
    .split(',')
    .map((item) => item.trim())
    .filter(Boolean)
    .filter((item) => {
      try {
        const url = new URL(item);
        return url.protocol === 'http:' || url.protocol === 'https:';
      } catch (_) {
        return false;
      }
    })
    .filter((item) => {
      if (seen.has(item)) return false;
      seen.add(item);
      return true;
    });
}

function isValidCoordinate(lat, lon) {
  return Number.isFinite(lat) && Number.isFinite(lon) && lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
}

function isInsideDaNang(lat, lon) {
  return isValidCoordinate(lat, lon) &&
    lat >= DA_NANG_BBOX.south &&
    lat <= DA_NANG_BBOX.north &&
    lon >= DA_NANG_BBOX.west &&
    lon <= DA_NANG_BBOX.east;
}

function readCSV(filePath) {
  return new Promise((resolve, reject) => {
    const rows = [];
    fs.createReadStream(filePath)
      .pipe(csv({
        mapHeaders: ({ header, index }) => {
          const cleanHeader = String(header || '').trim();
          return index === 0
            ? cleanHeader.replace(/^\uFEFF/, '')
            : cleanHeader;
        },
      }))
      .on('headers', (headers) => {
        rows.headers = headers;
      })
      .on('data', (row) => rows.push(row))
      .on('end', () => resolve({ rows, headers: rows.headers || [] }))
      .on('error', reject);
  });
}

function fileSha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function normalizeCanonicalPoiRow(row, { index = 0, file = CANONICAL_POI_CSV_PATH } = {}) {
  const globalId = cleanString(row.Global_ID);
  const cityId = cleanString(row.City_ID);
  const entityType = cleanString(row.Entity_Type).toLowerCase();
  const name = cleanString(row['Restaurant Name']);
  const lat = optionalNumber(row.Lat);
  const lon = optionalNumber(row.Lon);
  const category = cleanString(row.Category);
  const categoryNormalized = cleanString(row.Category_Normalized);
  const addressRaw = cleanString(row.Address_Raw);
  const addressCurrent = cleanString(row.Address_Current);
  const district = cleanString(row.District);
  const districtRaw = cleanString(row.District_Raw);
  const rating = optionalNumber(row['Overall Rating']);
  const ratingCount = optionalNumber(row.Rating_Count);
  const source = cleanString(row.Source);
  const imageUrls = parseImageUrls(row.Image_URL);
  const issues = [];

  if (entityType !== 'poi') issues.push('non_poi_entity_type');
  if (!globalId) issues.push('missing_global_id');
  if (!cityId) issues.push('missing_city_id');
  if (!name) issues.push('missing_name');
  if (!isValidCoordinate(lat, lon)) issues.push('invalid_coordinates');
  if (isValidCoordinate(lat, lon) && cityId === DEFAULT_CITY_ID && !isInsideDaNang(lat, lon)) {
    issues.push('outside_da_nang_bbox');
  }

  const validForApplication = entityType === 'poi' &&
    Boolean(globalId) &&
    Boolean(cityId) &&
    Boolean(name) &&
    isValidCoordinate(lat, lon);

  return {
    validForApplication,
    issues,
    raw: row,
    poi: validForApplication
      ? {
          id: globalId,
          globalId,
          legacyId: globalId,
          cityId,
          entityType,
          source,
          sourceId: cleanString(row.RestaurantID) || null,
          sourceIds: parseList(row.Source_IDs),
          aliasGlobalIds: parseList(row.Alias_Global_IDs),
          name,
          category: category || categoryNormalized || 'unknown',
          categoryNormalized: categoryNormalized || null,
          district,
          districtRaw,
          adminNormalizationStatus: cleanString(row.Admin_Normalization_Status) || null,
          address: addressCurrent || addressRaw || '',
          addressRaw: addressRaw || null,
          addressCurrent: addressCurrent || null,
          lat,
          lon,
          hasCoordinates: true,
          coordinateStatus: issues.includes('outside_da_nang_bbox') ? 'outside_city_bbox' : 'valid',
          rating,
          ratingScale: optionalNumber(row.Rating_Scale),
          ratingCount,
          reviewCount: ratingCount,
          reviewSampleCount: optionalNumber(row.Review_Sample_Count),
          googleRating: optionalNumber(row.Google_Rating),
          googleRatingCount: optionalNumber(row.Google_Rating_Count),
          foodyRating10: optionalNumber(row.Foody_Rating_10),
          foodyReviewSampleCount: optionalNumber(row.Foody_Review_Sample_Count),
          primaryRatingSource: cleanString(row.Primary_Rating_Source) || null,
          price: cleanString(row.Price) || null,
          priceMinVnd: optionalNumber(row.Price_Min_VND),
          priceMaxVnd: optionalNumber(row.Price_Max_VND),
          openingHoursRaw: cleanString(row.Opening_Hours_Raw) || null,
          imageUrls,
          imageUrl: imageUrls[0] || null,
          text: cleanString(row.LLM_Input_Text),
          normalized: normalizeText([
            name,
            category,
            categoryNormalized,
            district,
            addressCurrent,
            addressRaw,
            row.LLM_Input_Text,
          ].join(' ')),
          mergeStatus: cleanString(row.Merge_Status) || null,
          dataQualityFlags: parseList(row.Data_Quality_Flags),
          rowIndex: index,
          sourceFile: path.relative(path.resolve(__dirname, '..', '..'), file).replace(/\\/g, '/'),
          raw: row,
        }
      : null,
  };
}

function emptyQualityReport(filePath = CANONICAL_POI_CSV_PATH) {
  return {
    generatedAt: new Date().toISOString(),
    dataset: {
      path: path.relative(path.resolve(__dirname, '..', '..'), filePath).replace(/\\/g, '/'),
      sha256: fs.existsSync(filePath) ? fileSha256(filePath) : null,
      expectedColumns: EXPECTED_CANONICAL_COLUMNS,
    },
    cityId: DEFAULT_CITY_ID,
    bbox: DA_NANG_BBOX,
    totals: {
      rows: 0,
      columns: 0,
      applicationPois: 0,
      excludedUrbanVoidRows: 0,
      invalidRows: 0,
      uniqueGlobalIds: 0,
      duplicateGlobalIdExtraRows: 0,
      aliasGlobalIds: 0,
      missingName: 0,
      missingCityId: 0,
      missingCoordinates: 0,
      invalidCoordinates: 0,
      outsideDaNangBBox: 0,
      missingAddress: 0,
      missingRating: 0,
      missingReviewCount: 0,
    },
    columns: [],
    categoryDistribution: {},
    entityTypeDistribution: {},
    duplicateGlobalIds: [],
    nonMergeDuplicateCandidates: [],
    headerMatchesExpected: false,
  };
}

function increment(map, key, amount = 1) {
  const safeKey = cleanString(key) || 'unknown';
  map.set(safeKey, (map.get(safeKey) || 0) + amount);
}

function buildQualityReport({ rows, headers }, filePath = CANONICAL_POI_CSV_PATH) {
  const report = emptyQualityReport(filePath);
  const globalIdCounts = new Map();
  const categoryCounts = new Map();
  const entityTypeCounts = new Map();
  const aliasSet = new Set();
  const sourceIdGroups = new Map();

  report.totals.rows = rows.length;
  report.totals.columns = headers.length;
  report.columns = headers;
  report.headerMatchesExpected = JSON.stringify(headers) === JSON.stringify(EXPECTED_CANONICAL_COLUMNS);

  rows.forEach((row, index) => {
    const normalized = normalizeCanonicalPoiRow(row, { index, file: filePath });
    const globalId = cleanString(row.Global_ID);
    const entityType = cleanString(row.Entity_Type).toLowerCase() || 'unknown';
    const lat = optionalNumber(row.Lat);
    const lon = optionalNumber(row.Lon);
    const address = cleanString(row.Address_Current) || cleanString(row.Address_Raw);
    const sourceIds = parseList(row.Source_IDs);

    if (globalId) increment(globalIdCounts, globalId);
    parseList(row.Alias_Global_IDs).forEach((id) => aliasSet.add(id));
    increment(entityTypeCounts, entityType);

    if (entityType === 'urban_void') report.totals.excludedUrbanVoidRows += 1;
    if (normalized.validForApplication) {
      report.totals.applicationPois += 1;
      increment(categoryCounts, normalized.poi.categoryNormalized || normalized.poi.category);
      const duplicateKey = sourceIds.join('|') || normalized.poi.sourceId || '';
      if (duplicateKey) {
        const bucket = sourceIdGroups.get(duplicateKey) || [];
        bucket.push(normalized.poi.globalId);
        sourceIdGroups.set(duplicateKey, bucket);
      }
    } else {
      report.totals.invalidRows += 1;
    }

    if (normalized.issues.includes('missing_name')) report.totals.missingName += 1;
    if (normalized.issues.includes('missing_city_id')) report.totals.missingCityId += 1;
    if (lat === null || lon === null) report.totals.missingCoordinates += 1;
    if (normalized.issues.includes('invalid_coordinates')) report.totals.invalidCoordinates += 1;
    if (normalized.issues.includes('outside_da_nang_bbox')) report.totals.outsideDaNangBBox += 1;
    if (!address) report.totals.missingAddress += 1;
    if (optionalNumber(row['Overall Rating']) === null) report.totals.missingRating += 1;
    if (optionalNumber(row.Rating_Count) === null) report.totals.missingReviewCount += 1;
  });

  report.totals.uniqueGlobalIds = globalIdCounts.size;
  report.totals.duplicateGlobalIdExtraRows = Array.from(globalIdCounts.values())
    .reduce((sum, count) => sum + Math.max(0, count - 1), 0);
  report.totals.aliasGlobalIds = aliasSet.size;
  report.duplicateGlobalIds = Array.from(globalIdCounts.entries())
    .filter(([, count]) => count > 1)
    .sort((a, b) => b[1] - a[1])
    .map(([globalId, count]) => ({ globalId, count }));
  report.nonMergeDuplicateCandidates = Array.from(sourceIdGroups.entries())
    .filter(([, ids]) => ids.length > 1)
    .map(([sourceIds, globalIds]) => ({ sourceIds, globalIds }))
    .slice(0, 100);
  report.categoryDistribution = Object.fromEntries(
    Array.from(categoryCounts.entries()).sort((a, b) => b[1] - a[1]),
  );
  report.entityTypeDistribution = Object.fromEntries(
    Array.from(entityTypeCounts.entries()).sort((a, b) => b[1] - a[1]),
  );

  return report;
}

class CanonicalCsvPoiRepository {
  constructor({ filePath = CANONICAL_POI_CSV_PATH } = {}) {
    this.filePath = filePath;
    this.cache = null;
    this.qualityCache = null;
  }

  async loadAll() {
    if (this.cache) return this.cache;
    const source = await readCSV(this.filePath);
    this.qualityCache = buildQualityReport(source, this.filePath);
    this.cache = source.rows
      .map((row, index) => normalizeCanonicalPoiRow(row, { index, file: this.filePath }))
      .filter((item) => item.validForApplication)
      .map((item) => item.poi);
    return this.cache;
  }

  async findByCity(cityId = DEFAULT_CITY_ID) {
    const pois = await this.loadAll();
    return pois.filter((poi) => poi.cityId === cityId);
  }

  async getQualityReport() {
    if (this.qualityCache) return this.qualityCache;
    const source = await readCSV(this.filePath);
    this.qualityCache = buildQualityReport(source, this.filePath);
    return this.qualityCache;
  }

  clearCache() {
    this.cache = null;
    this.qualityCache = null;
  }
}

module.exports = {
  CANONICAL_POI_CSV_PATH,
  DA_NANG_BBOX,
  DEFAULT_CANONICAL_POI_CSV,
  DEFAULT_CITY_ID,
  EXPECTED_CANONICAL_COLUMNS,
  CanonicalCsvPoiRepository,
  buildQualityReport,
  cleanString,
  fileSha256,
  isInsideDaNang,
  isValidCoordinate,
  normalizeCanonicalPoiRow,
  normalizeText,
  optionalNumber,
  parseImageUrls,
  parseList,
  readCSV,
};

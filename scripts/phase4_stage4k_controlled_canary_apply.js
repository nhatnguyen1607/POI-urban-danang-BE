const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const { parseCsv } = require('../src/modules/cityPackPreparation/canonicalDataset');
const { EXPECTED_CANONICAL_COLUMNS } = require('../src/services/canonicalCsvPoiRepository');

const EXPECTED_BASELINE_SHA = '5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae';
const EXPECTED_ROWS = 4166;
const EXPECTED_DECISION_COUNTS = Object.freeze({ APPROVE: 10, REJECT: 5, DEFER: 10 });
const ALLOWED_MUTATION_FIELDS = Object.freeze(['address', 'phone', 'website']);
const LOCKED_FIELDS = Object.freeze([
  'name', 'coordinates', 'latitude', 'longitude', 'category', 'externalIds',
  'canonicalId', 'sourceIdentity', 'rowIdentity',
]);
const FIELD_TO_CANONICAL_COLUMN = Object.freeze({
  address: 'Address_Current',
  phone: null,
  website: null,
});

const APPROVED_CASES = Object.freeze([
  {
    caseId: 'MATCH:overture:overture:0b5d5134-5b87-4109-a53e-5887be0ceaea',
    canonicalId: 'google_maps_241',
    canonicalName: 'All Seasons Buffet Da Nang',
    fields: ['address', 'website', 'phone'],
  },
  {
    caseId: 'MATCH:overture:overture:19f0bbd2-2f66-4a58-bafd-963ec3fc2c9b',
    canonicalId: 'google_maps_1557',
    canonicalName: '40Plus Cafe & Vinyl Music',
    fields: ['address', 'phone'],
  },
  {
    caseId: 'MATCH:overture:overture:1e1966bb-c0c8-40df-8e5d-db423558f2b2',
    canonicalId: 'google_maps_2368',
    canonicalName: 'M\u1ef3 Qu\u1ea3ng C\u00f4 S\u00e1u',
    fields: ['address', 'website', 'phone'],
  },
  {
    caseId: 'MATCH:overture:overture:9c0ec2ea-fc55-4d8c-ac0d-f863a66cdf42',
    canonicalId: 'google_maps_2088',
    canonicalName: 'BonPas Bakery & Cafe',
    fields: ['address', 'phone'],
  },
  {
    caseId: 'MATCH:overture:overture:a4ebc563-8eb9-48ca-b2bd-8b2508318c4e',
    canonicalId: 'google_maps_2716',
    canonicalName: 'Roost Coffee Roasters | Da Nang',
    fields: ['address', 'phone'],
  },
  {
    caseId: 'MATCH:overture:overture:e8b8593a-c202-4274-8aaa-03070caf69f9',
    canonicalId: 'google_maps_1156',
    canonicalName: 'H\u1ea3i S\u1ea3n Qu\u00e1n C\u00f3',
    fields: ['address', 'website', 'phone'],
  },
  {
    caseId: 'MATCH:overture:overture:ff3c30b4-6a4e-4c90-965e-60e170533224',
    canonicalId: 'google_maps_1603',
    canonicalName: "Ly's Bakery",
    fields: ['address', 'website', 'phone'],
  },
  {
    caseId: 'MATCH:overture:overture:08955212-c94e-4d50-89ad-ab73a0defb0e',
    canonicalId: 'google_maps_1602',
    canonicalName: 'Bakery Ho\u00e0ng Ph\u00e1t',
    fields: ['address', 'phone'],
  },
  {
    caseId: 'MATCH:overture:overture:4b42243a-e10d-4898-8bd1-74b355694a2b',
    canonicalId: 'google_maps_5247',
    canonicalName: 'Th\u00eca G\u1ed7 \u0110\u00e0 N\u1eb5ng',
    fields: ['address', 'website', 'phone'],
  },
  {
    caseId: 'MATCH:overture:overture:fbacb852-0983-4801-85fe-e345f7c3fce8',
    canonicalId: 'google_maps_1738',
    canonicalName: 'Gear Cafe\u2019 And Bistro',
    fields: ['address', 'website', 'phone'],
  },
]);

function sha256Buffer(value) {
  return crypto.createHash('sha256').update(value).digest('hex');
}

function sha256File(filePath) {
  return sha256Buffer(fs.readFileSync(filePath));
}

function stableJson(value) {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${stableJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

function stableHash(value) {
  return sha256Buffer(Buffer.from(stableJson(value), 'utf8'));
}

function parseCsvObjects(text) {
  const rows = parseCsv(text);
  if (rows.length === 0) return { headers: [], rows: [] };
  const headers = rows[0].map((header, index) => (
    index === 0 ? header.replace(/^\uFEFF/, '').trim() : header.trim()
  ));
  return {
    headers,
    rows: rows.slice(1).map((cells) => Object.fromEntries(
      headers.map((header, index) => [header, cells[index] ?? '']),
    )),
  };
}

function readCsvObjects(filePath) {
  return parseCsvObjects(fs.readFileSync(filePath, 'utf8')).rows;
}

function readJsonl(filePath) {
  return fs.readFileSync(filePath, 'utf8').split(/\r?\n/).filter(Boolean).map(JSON.parse);
}

function valueCounts(rows, field) {
  return rows.reduce((counts, row) => {
    const key = String(row[field] || '').trim();
    counts[key] = (counts[key] || 0) + 1;
    return counts;
  }, {});
}

function assertEqualSet(actual, expected, message) {
  const left = [...actual].sort();
  const right = [...expected].sort();
  if (JSON.stringify(left) !== JSON.stringify(right)) {
    throw new Error(`${message}: expected ${right.join('|')}, got ${left.join('|')}`);
  }
}

function validateDecisionInputs({ decisions, reviewRows, evidenceCases }) {
  if (decisions.length !== 25) throw new Error(`decision_row_count:${decisions.length}`);
  const decisionIds = decisions.map((row) => row.case_id);
  if (new Set(decisionIds).size !== 25) throw new Error('duplicate_decision_case_id');
  const malformed = decisions.filter((row) => !['APPROVE', 'REJECT', 'DEFER']
    .includes(String(row.reviewer_decision || '').trim()));
  if (malformed.length) throw new Error(`malformed_decisions:${malformed.map((row) => row.case_id).join('|')}`);

  const counts = valueCounts(decisions, 'reviewer_decision');
  for (const [decision, expected] of Object.entries(EXPECTED_DECISION_COUNTS)) {
    if ((counts[decision] || 0) !== expected) {
      throw new Error(`decision_count_${decision}:${counts[decision] || 0}`);
    }
  }

  const approvedIds = decisions
    .filter((row) => row.reviewer_decision === 'APPROVE')
    .map((row) => row.case_id);
  assertEqualSet(approvedIds, APPROVED_CASES.map((item) => item.caseId), 'approved_case_ids');

  const reviewById = new Map(reviewRows.map((row) => [row.case_id, row]));
  const evidenceById = new Map(evidenceCases.map((item) => [item.caseId, item]));
  if (reviewById.size !== reviewRows.length) throw new Error('duplicate_review_case_id');
  if (reviewRows.length !== 25) throw new Error(`review_row_count:${reviewRows.length}`);

  for (const approved of APPROVED_CASES) {
    const review = reviewById.get(approved.caseId);
    const evidence = evidenceById.get(approved.caseId);
    if (!review) throw new Error(`missing_review_case:${approved.caseId}`);
    if (!evidence) throw new Error(`missing_evidence_case:${approved.caseId}`);
    if (review.evidence_confidence !== 'STRONG') throw new Error(`review_confidence:${approved.caseId}`);
    if (review.proposed_operation !== 'ENRICH_EXISTING_SAFE_FIELDS_ONLY') {
      throw new Error(`review_operation:${approved.caseId}`);
    }
    assertEqualSet(
      String(review.safe_addition_fields || '').split('|').filter(Boolean),
      approved.fields,
      `safe_field_whitelist:${approved.caseId}`,
    );
    if (review.canonical_id !== approved.canonicalId || review.canonical_name !== approved.canonicalName) {
      throw new Error(`canonical_target_contract:${approved.caseId}`);
    }
    if (evidence.evidence?.confidence !== 'STRONG' || evidence.evidence?.label !== 'MATCH') {
      throw new Error(`evidence_gate:${approved.caseId}`);
    }
    if (!evidence.provenanceComplete || evidence.source?.source !== 'overture') {
      throw new Error(`source_provenance_gate:${approved.caseId}`);
    }
  }

  return { counts, approvedIds, reviewById, evidenceById };
}

function validateProvenance(change, approved) {
  const provenance = change?.provenance || {};
  const expectedSourceId = approved.caseId.replace(/^MATCH:overture:/, '');
  const required = ['source', 'sourceId', 'snapshotRef', 'license', 'policyClass', 'attribution', 'licenseUrl'];
  const missing = required.filter((field) => !provenance[field]);
  if (missing.length) return { valid: false, reason: `missing:${missing.join('|')}` };
  if (provenance.source !== 'overture' || provenance.sourceId !== expectedSourceId) {
    return { valid: false, reason: 'unexpected_source_identity' };
  }
  if (provenance.license !== 'CDLA-Permissive-2.0'
    || provenance.policyClass !== 'OPEN_PERMISSIVE_CANDIDATE') {
    return { valid: false, reason: 'unexpected_license_policy' };
  }
  if (/google/i.test(provenance.source)) return { valid: false, reason: 'google_source_not_allowed' };
  return { valid: true, reason: null };
}

function validSourceValue(field, value) {
  const text = String(value || '').trim();
  if (!text || /[\u0000-\u0008\u000B\u000C\u000E-\u001F]/.test(text)) return false;
  if (field === 'address') return text.length >= 3 && text.length <= 500;
  if (field === 'phone') return /^\+?[0-9][0-9 ()-]{7,24}$/.test(text);
  if (field === 'website') {
    try {
      return ['http:', 'https:'].includes(new URL(text).protocol);
    } catch {
      return false;
    }
  }
  return false;
}

function inspectCanonicalText(text, { requireBaselineSha = false } = {}) {
  const parsed = parseCsvObjects(text);
  if (JSON.stringify(parsed.headers) !== JSON.stringify(EXPECTED_CANONICAL_COLUMNS)) {
    throw new Error('canonical_header_mismatch');
  }
  if (parsed.rows.length !== EXPECTED_ROWS) throw new Error(`canonical_row_count:${parsed.rows.length}`);
  const ids = parsed.rows.map((row) => row.Global_ID);
  if (new Set(ids).size !== EXPECTED_ROWS) throw new Error('duplicate_canonical_id');
  const digest = sha256Buffer(Buffer.from(text, 'utf8'));
  if (requireBaselineSha && digest !== EXPECTED_BASELINE_SHA) {
    throw new Error(`canonical_baseline_sha:${digest}`);
  }
  return { ...parsed, sha256: digest };
}

function buildApplyPlan({ canonicalText, decisions, reviewRows, evidenceCases, requireBaselineSha = true }) {
  const canonical = inspectCanonicalText(canonicalText, { requireBaselineSha });
  const input = validateDecisionInputs({ decisions, reviewRows, evidenceCases });
  const canonicalById = new Map(canonical.rows.map((row) => [row.Global_ID, row]));
  const decisionById = new Map(decisions.map((row) => [row.case_id, row]));
  const mutations = [];
  const skipped = [];

  for (const approved of APPROVED_CASES) {
    const canonicalRow = canonicalById.get(approved.canonicalId);
    const evidence = input.evidenceById.get(approved.caseId);
    if (!canonicalRow) throw new Error(`missing_canonical_target:${approved.canonicalId}`);
    if (canonicalRow['Restaurant Name'] !== approved.canonicalName) {
      throw new Error(`canonical_name_changed:${approved.canonicalId}`);
    }
    if (decisionById.get(approved.caseId)?.reviewer_decision !== 'APPROVE') {
      throw new Error(`approval_missing:${approved.caseId}`);
    }

    const changesByField = new Map((evidence.fieldChanges || []).map((change) => [change.field, change]));
    for (const field of approved.fields) {
      const change = changesByField.get(field);
      const canonicalColumn = FIELD_TO_CANONICAL_COLUMN[field];
      const newValue = change?.newValue === null || change?.newValue === undefined
        ? '' : String(change.newValue).trim();
      const base = {
        case_id: approved.caseId,
        canonical_id: approved.canonicalId,
        canonical_name: approved.canonicalName,
        field,
        old_value: canonicalColumn ? canonicalRow[canonicalColumn] : null,
        new_value: newValue || null,
        source: evidence.source.source,
        source_id: evidence.source.sourceId,
        provenance: change?.provenance || null,
        license: change?.provenance?.license || null,
        human_decision: 'APPROVE',
      };

      if (!canonicalColumn || !canonical.headers.includes(canonicalColumn)) {
        skipped.push({ ...base, status: 'SKIPPED_UNSUPPORTED_CANONICAL_FIELD' });
        continue;
      }
      if (!change || change.state !== 'SAFE_ADDITION' || !validSourceValue(field, newValue)) {
        skipped.push({ ...base, status: 'SKIPPED_INVALID_OR_UNSAFE_SOURCE_VALUE' });
        continue;
      }
      const provenance = validateProvenance(change, approved);
      if (!provenance.valid) {
        skipped.push({ ...base, status: 'SKIPPED_PROVENANCE_LICENSE_FAILURE', reason: provenance.reason });
        continue;
      }
      if (String(canonicalRow[canonicalColumn] || '').trim()) {
        skipped.push({ ...base, status: canonicalRow[canonicalColumn] === newValue
          ? 'SKIPPED_EXISTING_VALUE' : 'CONFLICT_NOT_APPLIED' });
        continue;
      }
      mutations.push({ ...base, canonical_column: canonicalColumn, status: 'PLANNED_SAFE_ADDITION' });
    }
  }

  validatePlanSafety({ mutations, decisions });
  const summaryCore = {
    baselineSha256: canonical.sha256,
    beforeRows: canonical.rows.length,
    decisions: input.counts,
    approvedCaseIds: APPROVED_CASES.map((item) => item.caseId),
    approvedCases: APPROVED_CASES.length,
    canonicalTargets: [...new Set(mutations.map((item) => item.canonical_id))].length,
    plannedByField: Object.fromEntries(ALLOWED_MUTATION_FIELDS.map((field) => [
      field, mutations.filter((item) => item.field === field).length,
    ])),
    plannedFieldAdditions: mutations.length,
    skippedByStatus: valueCounts(skipped, 'status'),
    createNew: 0,
    deletedRows: 0,
    mergedRows: 0,
    lockedFields: LOCKED_FIELDS,
  };
  return {
    status: 'PRE_APPLY_PLAN_VALIDATED_NOT_YET_APPLIED',
    ...summaryCore,
    planHash: stableHash({ mutations, summaryCore }),
    mutations,
    skipped,
  };
}

function validatePlanSafety({ mutations, decisions }) {
  if (new Set(mutations.map((item) => item.case_id)).size > 10) throw new Error('too_many_approved_cases');
  if (mutations.length > 26) throw new Error('too_many_field_additions');
  const approvedIds = new Set(APPROVED_CASES.map((item) => item.caseId));
  const decisionById = new Map(decisions.map((row) => [row.case_id, row.reviewer_decision]));
  for (const mutation of mutations) {
    if (!approvedIds.has(mutation.case_id)) throw new Error(`unapproved_mutation:${mutation.case_id}`);
    if (decisionById.get(mutation.case_id) !== 'APPROVE') throw new Error(`non_approved_mutation:${mutation.case_id}`);
    if (!ALLOWED_MUTATION_FIELDS.includes(mutation.field)) throw new Error(`locked_field_mutation:${mutation.field}`);
    const contract = APPROVED_CASES.find((item) => item.caseId === mutation.case_id);
    if (!contract.fields.includes(mutation.field)) throw new Error(`non_whitelisted_mutation:${mutation.case_id}:${mutation.field}`);
  }
  return true;
}

function parseCsvLineTokens(line) {
  const tokens = [];
  let start = 0;
  let inQuotes = false;
  for (let index = 0; index <= line.length; index += 1) {
    const char = line[index];
    if (char === '"') {
      if (inQuotes && line[index + 1] === '"') index += 1;
      else inQuotes = !inQuotes;
    }
    if ((char === ',' && !inQuotes) || index === line.length) {
      const raw = line.slice(start, index);
      const value = raw.startsWith('"') && raw.endsWith('"')
        ? raw.slice(1, -1).replace(/""/g, '"') : raw;
      tokens.push({ raw, value });
      start = index + 1;
    }
  }
  if (inQuotes) throw new Error('unterminated_csv_quote');
  return tokens;
}

function csvEscape(value) {
  const text = value === null || value === undefined ? '' : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

function applyMutationsToText(canonicalText, mutations, { rollback = false } = {}) {
  const lineEnding = canonicalText.includes('\r\n') ? '\r\n' : '\n';
  if (lineEnding !== '\r\n') throw new Error('unexpected_canonical_line_endings');
  const hasFinalNewline = canonicalText.endsWith(lineEnding);
  const lines = canonicalText.split(lineEnding);
  if (hasFinalNewline) lines.pop();
  const headers = parseCsvLineTokens(lines[0]).map((token, index) => (
    index === 0 ? token.value.replace(/^\uFEFF/, '') : token.value
  ));
  const idIndex = headers.indexOf('Global_ID');
  const byCanonicalId = new Map();
  for (const mutation of mutations) {
    if (!byCanonicalId.has(mutation.canonical_id)) byCanonicalId.set(mutation.canonical_id, []);
    byCanonicalId.get(mutation.canonical_id).push(mutation);
  }
  const touched = new Set();
  const output = lines.map((line, lineIndex) => {
    if (lineIndex === 0) return line;
    const tokens = parseCsvLineTokens(line);
    const id = tokens[idIndex]?.value;
    const rowMutations = byCanonicalId.get(id);
    if (!rowMutations) return line;
    for (const mutation of rowMutations) {
      const columnIndex = headers.indexOf(mutation.canonical_column);
      if (columnIndex < 0) throw new Error(`missing_mutation_column:${mutation.canonical_column}`);
      tokens[columnIndex].raw = csvEscape(rollback ? mutation.old_value : mutation.new_value);
      touched.add(`${mutation.case_id}:${mutation.field}`);
    }
    return tokens.map((token) => token.raw).join(',');
  });
  if (touched.size !== mutations.length) throw new Error('not_all_mutations_applied');
  return `${output.join(lineEnding)}${hasFinalNewline ? lineEnding : ''}`;
}

function validatePostApply({ beforeText, afterText, plan }) {
  const before = inspectCanonicalText(beforeText, { requireBaselineSha: true });
  const after = inspectCanonicalText(afterText);
  const expectedText = applyMutationsToText(beforeText, plan.mutations);
  if (afterText !== expectedText) throw new Error('exact_diff_mismatch');
  if (applyMutationsToText(beforeText, plan.mutations) !== expectedText) throw new Error('nondeterministic_apply');
  const rolledBack = applyMutationsToText(afterText, plan.mutations, { rollback: true });
  if (rolledBack !== beforeText) throw new Error('rollback_not_byte_identical');

  const targetIds = new Set(APPROVED_CASES.map((item) => item.canonicalId));
  const beforeById = new Map(before.rows.map((row) => [row.Global_ID, row]));
  const afterById = new Map(after.rows.map((row) => [row.Global_ID, row]));
  let unchangedNonTargets = 0;
  for (const [id, beforeRow] of beforeById) {
    const afterRow = afterById.get(id);
    if (!afterRow) throw new Error(`deleted_row:${id}`);
    const changedColumns = before.headers.filter((header) => beforeRow[header] !== afterRow[header]);
    if (!targetIds.has(id)) {
      if (changedColumns.length) throw new Error(`non_target_changed:${id}:${changedColumns.join('|')}`);
      unchangedNonTargets += 1;
    } else if (changedColumns.some((column) => column !== 'Address_Current')) {
      throw new Error(`locked_target_column_changed:${id}:${changedColumns.join('|')}`);
    }
  }
  if (unchangedNonTargets !== EXPECTED_ROWS - targetIds.size) {
    throw new Error(`non_target_count:${unchangedNonTargets}`);
  }

  return {
    beforeRows: before.rows.length,
    afterRows: after.rows.length,
    beforeSha256: before.sha256,
    afterSha256: after.sha256,
    duplicateCanonicalIds: after.rows.length - new Set(after.rows.map((row) => row.Global_ID)).size,
    canonicalTargetsModified: new Set(plan.mutations.map((item) => item.canonical_id)).size,
    actualByField: Object.fromEntries(ALLOWED_MUTATION_FIELDS.map((field) => [
      field, plan.mutations.filter((item) => item.field === field).length,
    ])),
    nonTargetRowsUnchanged: unchangedNonTargets,
    namesChanged: 0,
    coordinatesChanged: 0,
    categoriesChanged: 0,
    externalIdsChanged: 0,
    newRows: 0,
    deletedRows: 0,
    rollbackSha256: sha256Buffer(Buffer.from(rolledBack, 'utf8')),
    rollbackByteIdentical: true,
    deterministic: true,
  };
}

function csvArtifact(rows, fields) {
  return `${[fields.join(','), ...rows.map((row) => fields.map((field) => csvEscape(
    typeof row[field] === 'object' && row[field] !== null ? JSON.stringify(row[field]) : row[field],
  )).join(','))].join('\n')}\n`;
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function writePreApplyArtifacts(artifactDir, plan) {
  fs.mkdirSync(artifactDir, { recursive: true });
  const fields = [
    'case_id', 'canonical_id', 'canonical_name', 'field', 'old_value', 'new_value',
    'source', 'source_id', 'provenance', 'license', 'human_decision', 'status',
  ];
  fs.writeFileSync(path.join(artifactDir, 'pre_apply_plan.csv'), csvArtifact(plan.mutations, fields), 'utf8');
  writeJson(path.join(artifactDir, 'pre_apply_summary.json'), {
    ...plan,
    mutations: undefined,
    skipped: plan.skipped,
  });
}

function parseArgs(args) {
  const result = { command: args[0] || 'plan' };
  for (let index = 1; index < args.length; index += 2) {
    const key = String(args[index] || '').replace(/^--/, '').replace(/-([a-z])/g, (_, char) => char.toUpperCase());
    result[key] = args[index + 1];
  }
  return result;
}

function loadInputs(options) {
  const canonicalText = fs.readFileSync(options.canonical, 'utf8');
  return {
    canonicalText,
    decisions: readCsvObjects(options.decisions),
    reviewRows: readCsvObjects(options.review),
    evidenceCases: readJsonl(options.evidence),
  };
}

function runCli(args = process.argv.slice(2)) {
  const options = parseArgs(args);
  const required = ['canonical', 'decisions', 'review', 'evidence', 'artifactDir'];
  const missing = required.filter((field) => !options[field]);
  if (missing.length) throw new Error(`missing_arguments:${missing.join('|')}`);
  const inputs = loadInputs(options);
  const plan = buildApplyPlan(inputs);
  writePreApplyArtifacts(options.artifactDir, plan);
  if (options.command === 'plan') return plan;
  if (options.command !== 'apply') throw new Error(`unknown_command:${options.command}`);
  if (process.env.URBANAGENT_ALLOW_STAGE4K_CANONICAL_WRITE !== 'true') {
    throw new Error('Stage 4K write requires URBANAGENT_ALLOW_STAGE4K_CANONICAL_WRITE=true');
  }
  const afterText = applyMutationsToText(inputs.canonicalText, plan.mutations);
  const validation = validatePostApply({ beforeText: inputs.canonicalText, afterText, plan });
  fs.writeFileSync(options.canonical, afterText, 'utf8');
  writeJson(path.join(options.artifactDir, 'apply_summary.json'), { planHash: plan.planHash, ...validation });
  writeJson(path.join(options.artifactDir, 'rollback_verification.json'), {
    expectedSha256: EXPECTED_BASELINE_SHA,
    rolledBackSha256: validation.rollbackSha256,
    byteIdentical: validation.rollbackByteIdentical,
  });
  writeJson(path.join(options.artifactDir, 'canonical_sha_after.json'), {
    before: validation.beforeSha256,
    after: validation.afterSha256,
  });
  fs.writeFileSync(path.join(options.artifactDir, 'before_after_diff.csv'), csvArtifact(
    plan.mutations.map((item) => ({ ...item, before: item.old_value, after: item.new_value })),
    ['case_id', 'canonical_id', 'canonical_name', 'field', 'before', 'after', 'source', 'source_id'],
  ), 'utf8');
  return { plan, validation };
}

if (require.main === module) {
  try {
    const result = runCli();
    const summary = result.validation ? { plan: {
      approvedCases: result.plan.approvedCases,
      plannedByField: result.plan.plannedByField,
      plannedFieldAdditions: result.plan.plannedFieldAdditions,
      skippedByStatus: result.plan.skippedByStatus,
      planHash: result.plan.planHash,
    }, validation: result.validation } : {
      approvedCases: result.approvedCases,
      plannedByField: result.plannedByField,
      plannedFieldAdditions: result.plannedFieldAdditions,
      skippedByStatus: result.skippedByStatus,
      planHash: result.planHash,
    };
    console.log(JSON.stringify(summary, null, 2));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

module.exports = {
  ALLOWED_MUTATION_FIELDS,
  APPROVED_CASES,
  EXPECTED_BASELINE_SHA,
  EXPECTED_DECISION_COUNTS,
  EXPECTED_ROWS,
  FIELD_TO_CANONICAL_COLUMN,
  LOCKED_FIELDS,
  applyMutationsToText,
  buildApplyPlan,
  csvEscape,
  inspectCanonicalText,
  parseCsvLineTokens,
  readCsvObjects,
  readJsonl,
  stableHash,
  validateDecisionInputs,
  validatePlanSafety,
  validatePostApply,
  validateProvenance,
  validSourceValue,
};

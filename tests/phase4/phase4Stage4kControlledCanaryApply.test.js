const assert = require('node:assert/strict');
const fs = require('node:fs');
const test = require('node:test');

const {
  APPROVED_CASES,
  EXPECTED_BASELINE_SHA,
  FIELD_TO_CANONICAL_COLUMN,
  applyMutationsToText,
  buildApplyPlan,
  inspectCanonicalText,
  parseCsvLineTokens,
  readCsvObjects,
  readJsonl,
  validateDecisionInputs,
  validatePlanSafety,
  validatePostApply,
  validateProvenance,
  validSourceValue,
} = require('../../scripts/phase4_stage4k_controlled_canary_apply');

const EXPECTED_APPROVED_IDS = [
  'MATCH:overture:overture:0b5d5134-5b87-4109-a53e-5887be0ceaea',
  'MATCH:overture:overture:19f0bbd2-2f66-4a58-bafd-963ec3fc2c9b',
  'MATCH:overture:overture:1e1966bb-c0c8-40df-8e5d-db423558f2b2',
  'MATCH:overture:overture:9c0ec2ea-fc55-4d8c-ac0d-f863a66cdf42',
  'MATCH:overture:overture:a4ebc563-8eb9-48ca-b2bd-8b2508318c4e',
  'MATCH:overture:overture:e8b8593a-c202-4274-8aaa-03070caf69f9',
  'MATCH:overture:overture:ff3c30b4-6a4e-4c90-965e-60e170533224',
  'MATCH:overture:overture:08955212-c94e-4d50-89ad-ab73a0defb0e',
  'MATCH:overture:overture:4b42243a-e10d-4898-8bd1-74b355694a2b',
  'MATCH:overture:overture:fbacb852-0983-4801-85fe-e345f7c3fce8',
];

const inputPaths = {
  before: process.env.URBANAGENT_STAGE4K_BEFORE_CANONICAL,
  after: process.env.URBANAGENT_STAGE4K_AFTER_CANONICAL,
  decisions: process.env.URBANAGENT_STAGE4K_DECISIONS,
  review: process.env.URBANAGENT_STAGE4K_REVIEW,
  evidence: process.env.URBANAGENT_STAGE4K_EVIDENCE,
};
const integrationAvailable = Object.values(inputPaths).every((filePath) => filePath && fs.existsSync(filePath));

function loadIntegrationInputs() {
  return {
    canonicalText: fs.readFileSync(inputPaths.before, 'utf8'),
    decisions: readCsvObjects(inputPaths.decisions),
    reviewRows: readCsvObjects(inputPaths.review),
    evidenceCases: readJsonl(inputPaths.evidence),
  };
}

function integrationTest(name, callback) {
  test(name, { skip: !integrationAvailable }, callback);
}

test('Stage 4K pins exactly the ten human-approved case IDs', () => {
  assert.equal(APPROVED_CASES.length, 10);
  assert.deepEqual(APPROVED_CASES.map((item) => item.caseId).sort(), [...EXPECTED_APPROVED_IDS].sort());
  assert.equal(new Set(APPROVED_CASES.map((item) => item.canonicalId)).size, 10);
});

test('Stage 4K limits the field contract to the approved maximum of 26 additions', () => {
  assert.equal(APPROVED_CASES.reduce((total, item) => total + item.fields.length, 0), 26);
  assert.ok(APPROVED_CASES.every((item) => item.fields.every((field) => (
    ['address', 'phone', 'website'].includes(field)
  ))));
  assert.deepEqual(FIELD_TO_CANONICAL_COLUMN, {
    address: 'Address_Current', phone: null, website: null,
  });
});

test('Stage 4K CSV token parser preserves quoted values and escaped quotes', () => {
  const tokens = parseCsvLineTokens('id,"A, B","said ""hello""",tail');
  assert.deepEqual(tokens.map((item) => item.value), ['id', 'A, B', 'said "hello"', 'tail']);
  assert.deepEqual(tokens.map((item) => item.raw), ['id', '"A, B"', '"said ""hello"""', 'tail']);
});

test('Stage 4K validates address, phone and http(s) source formats', () => {
  assert.equal(validSourceValue('address', '12 Bach Dang, Da Nang'), true);
  assert.equal(validSourceValue('phone', '+842361234567'), true);
  assert.equal(validSourceValue('website', 'https://example.test/place'), true);
  assert.equal(validSourceValue('website', 'javascript:alert(1)'), false);
  assert.equal(validSourceValue('phone', 'unknown'), false);
});

test('Stage 4K provenance gate accepts only complete expected Overture provenance', () => {
  const approved = APPROVED_CASES[0];
  const sourceId = approved.caseId.replace(/^MATCH:overture:/, '');
  const valid = validateProvenance({ provenance: {
    source: 'overture', sourceId, snapshotRef: 'overture-fixed',
    license: 'CDLA-Permissive-2.0', policyClass: 'OPEN_PERMISSIVE_CANDIDATE',
    attribution: 'Overture Maps Foundation', licenseUrl: 'https://cdla.dev/permissive-2-0/',
  } }, approved);
  const google = validateProvenance({ provenance: {
    source: 'google', sourceId, snapshotRef: 'fixed', license: 'x', policyClass: 'x',
    attribution: 'x', licenseUrl: 'https://example.test',
  } }, approved);
  assert.equal(valid.valid, true);
  assert.equal(google.valid, false);
});

test('Stage 4K plan safety rejects locked and non-whitelisted fields', () => {
  const decisions = APPROVED_CASES.map((item) => ({ case_id: item.caseId, reviewer_decision: 'APPROVE' }));
  assert.throws(() => validatePlanSafety({
    decisions,
    mutations: [{ case_id: APPROVED_CASES[0].caseId, field: 'name' }],
  }), /locked_field_mutation/);
  assert.throws(() => validatePlanSafety({
    decisions,
    mutations: [{ case_id: APPROVED_CASES[1].caseId, field: 'website' }],
  }), /non_whitelisted_mutation/);
});

integrationTest('Stage 4K accepts exactly 10 APPROVE, 5 REJECT and 10 DEFER decisions', () => {
  const inputs = loadIntegrationInputs();
  const result = validateDecisionInputs(inputs);
  assert.deepEqual(result.counts, { APPROVE: 10, REJECT: 5, DEFER: 10 });
  assert.equal(result.approvedIds.length, 10);
});

integrationTest('Stage 4K rejects an altered approval set before planning', () => {
  const inputs = loadIntegrationInputs();
  const changed = inputs.decisions.map((row) => ({ ...row }));
  const approved = changed.find((row) => row.reviewer_decision === 'APPROVE');
  const rejected = changed.find((row) => row.reviewer_decision === 'REJECT');
  approved.reviewer_decision = 'REJECT';
  rejected.reviewer_decision = 'APPROVE';
  assert.throws(() => validateDecisionInputs({ ...inputs, decisions: changed }), /approved_case_ids/);
});

integrationTest('Stage 4K plans only 10 address additions and no create/delete/merge', () => {
  const plan = buildApplyPlan(loadIntegrationInputs());
  assert.deepEqual(plan.plannedByField, { address: 10, phone: 0, website: 0 });
  assert.equal(plan.plannedFieldAdditions, 10);
  assert.equal(plan.canonicalTargets, 10);
  assert.equal(plan.createNew, 0);
  assert.equal(plan.deletedRows, 0);
  assert.equal(plan.mergedRows, 0);
  assert.equal(plan.skippedByStatus.SKIPPED_UNSUPPORTED_CANONICAL_FIELD, 16);
  assert.ok(plan.mutations.every((item) => item.human_decision === 'APPROVE'));
});

integrationTest('Stage 4K gives REJECT and DEFER cases zero canonical mutations', () => {
  const inputs = loadIntegrationInputs();
  const plan = buildApplyPlan(inputs);
  const decisionById = new Map(inputs.decisions.map((row) => [row.case_id, row.reviewer_decision]));
  assert.ok(plan.mutations.every((item) => decisionById.get(item.case_id) === 'APPROVE'));
  assert.equal(plan.mutations.filter((item) => decisionById.get(item.case_id) === 'REJECT').length, 0);
  assert.equal(plan.mutations.filter((item) => decisionById.get(item.case_id) === 'DEFER').length, 0);
});

integrationTest('Stage 4K never overwrites a populated canonical address', () => {
  const inputs = loadIntegrationInputs();
  const initialPlan = buildApplyPlan(inputs);
  const oneMutation = initialPlan.mutations.slice(0, 1);
  const populatedText = applyMutationsToText(inputs.canonicalText, oneMutation);
  const replanned = buildApplyPlan({ ...inputs, canonicalText: populatedText, requireBaselineSha: false });
  assert.equal(replanned.plannedFieldAdditions, 9);
  assert.equal(
    (replanned.skippedByStatus.SKIPPED_EXISTING_VALUE || 0)
      + (replanned.skippedByStatus.CONFLICT_NOT_APPLIED || 0),
    1,
  );
});

integrationTest('Stage 4K application is deterministic and touches only target addresses', () => {
  const inputs = loadIntegrationInputs();
  const plan = buildApplyPlan(inputs);
  const first = applyMutationsToText(inputs.canonicalText, plan.mutations);
  const second = applyMutationsToText(inputs.canonicalText, plan.mutations);
  assert.equal(first, second);
  const result = validatePostApply({ beforeText: inputs.canonicalText, afterText: first, plan });
  assert.equal(result.canonicalTargetsModified, 10);
  assert.equal(result.nonTargetRowsUnchanged, 4156);
  assert.equal(result.namesChanged, 0);
  assert.equal(result.coordinatesChanged, 0);
  assert.equal(result.categoriesChanged, 0);
  assert.equal(result.externalIdsChanged, 0);
});

integrationTest('Stage 4K rollback restores the exact baseline bytes and SHA', () => {
  const inputs = loadIntegrationInputs();
  const plan = buildApplyPlan(inputs);
  const applied = applyMutationsToText(inputs.canonicalText, plan.mutations);
  const rolledBack = applyMutationsToText(applied, plan.mutations, { rollback: true });
  assert.equal(rolledBack, inputs.canonicalText);
  assert.equal(inspectCanonicalText(rolledBack).sha256, EXPECTED_BASELINE_SHA);
});

integrationTest('Stage 4K final canonical passes exact diff and integrity validation', () => {
  const inputs = loadIntegrationInputs();
  const plan = buildApplyPlan(inputs);
  const afterText = fs.readFileSync(inputPaths.after, 'utf8');
  const result = validatePostApply({ beforeText: inputs.canonicalText, afterText, plan });
  assert.equal(result.beforeRows, 4166);
  assert.equal(result.afterRows, 4166);
  assert.equal(result.duplicateCanonicalIds, 0);
  assert.notEqual(result.afterSha256, EXPECTED_BASELINE_SHA);
  assert.equal(result.rollbackSha256, EXPECTED_BASELINE_SHA);
  assert.equal(result.rollbackByteIdentical, true);
});

integrationTest('Stage 4K preserves the locked Thia Go canonical identity', () => {
  const inputs = loadIntegrationInputs();
  const plan = buildApplyPlan(inputs);
  const after = inspectCanonicalText(applyMutationsToText(inputs.canonicalText, plan.mutations));
  const target = after.rows.find((row) => row.Global_ID === 'google_maps_5247');
  assert.equal(target['Restaurant Name'], 'Th\u00eca G\u1ed7 \u0110\u00e0 N\u1eb5ng');
});

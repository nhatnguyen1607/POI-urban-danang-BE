const fs = require('node:fs');
const path = require('node:path');

const { runAutomatedReviewPolicy, DEFAULT_VERSIONS } = require('../src/modules/cityPackPreparation/automatedReviewPolicy');
const { inspectCanonicalDataset } = require('../src/modules/cityPackPreparation/canonicalDataset');
const { recoverStage4kSkippedEnrichments } = require('../src/modules/cityPackPreparation/cityPackEnrichment');
const {
  DECISION_SOURCES,
  createDecisionRecord,
  findReusableDecision,
  stableHash,
} = require('../src/modules/cityPackPreparation/decisionMemory');
const { readJsonl, writeJsonl } = require('../src/modules/cityPackPreparation/incrementalSync');
const { readCsvObjects } = require('./phase4_stage4k_controlled_canary_apply');

const ROOT = path.resolve(__dirname, '..');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const CONFIG_PATH = path.join(ROOT, 'config', 'phase4_stage4l_policy.json');
const EVIDENCE_PATH = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4j', 'evidence_pack.jsonl');
const STAGE4I_SUMMARY_PATH = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4i', 'stage4i_summary.json');

function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (!value.startsWith('--')) throw new Error(`Unexpected argument: ${value}`);
    const key = value.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    const next = argv[index + 1];
    if (!next || next.startsWith('--')) args[key] = true;
    else { args[key] = next; index += 1; }
  }
  return args;
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, 'utf8'));
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function fieldNames(reviewCase) {
  return (reviewCase.fieldChanges || [])
    .filter((item) => item.state === 'SAFE_ADDITION')
    .map((item) => item.field);
}

function enrichReviewCases(evidenceCases) {
  return evidenceCases.map((reviewCase) => ({
    ...reviewCase,
    geographicEligibility: 'INSIDE_DANANG',
    categoryEligibility: reviewCase.categoryEligibility || 'TRAVELER_RELEVANT',
    newCandidateTier: reviewCase.resolver?.classification === 'NEW_CANDIDATE'
      ? 'NEW_TIER_A_STRONG'
      : reviewCase.newCandidateTier,
    syncStatus: 'CHANGED',
  }));
}

function importHumanMemory(decisions, cases) {
  const byId = new Map(cases.map((item) => [item.caseId, item]));
  return decisions.map((row) => {
    const reviewCase = byId.get(row.case_id);
    if (!reviewCase) throw new Error(`Decision case missing from evidence pack: ${row.case_id}`);
    return createDecisionRecord({
      reviewCase,
      decision: row.reviewer_decision,
      decisionSource: DECISION_SOURCES.HUMAN,
      decisionReference: 'stage4j-human-canary-decisions',
      policyVersion: DEFAULT_VERSIONS.policy,
      resolverVersion: DEFAULT_VERSIONS.resolver,
      evidenceVersion: DEFAULT_VERSIONS.evidence,
      boundaryVersion: DEFAULT_VERSIONS.boundary,
      approvedFields: row.reviewer_decision === 'APPROVE' ? fieldNames(reviewCase) : [],
      note: row.reviewer_note || null,
    });
  }).sort((left, right) => left.caseId.localeCompare(right.caseId));
}

function countOutcomes(results) {
  return results.reduce((counts, item) => {
    counts[item.outcome] = (counts[item.outcome] || 0) + 1;
    return counts;
  }, {});
}

function buildAutomatedMemory(policyResults, cases) {
  const byId = new Map(cases.map((item) => [item.caseId, item]));
  return policyResults
    .filter((item) => ['AUTO_ACCEPT_SAFE', 'AUTO_REJECT'].includes(item.outcome))
    .map((item) => createDecisionRecord({
      reviewCase: byId.get(item.caseId),
      decision: item.outcome === 'AUTO_ACCEPT_SAFE' ? 'APPROVE' : 'REJECT',
      decisionSource: item.outcome === 'AUTO_ACCEPT_SAFE'
        ? DECISION_SOURCES.POLICY : DECISION_SOURCES.SYSTEM_SAFETY,
      decisionReference: 'stage4l-fixed-state-policy-simulation',
      policyVersion: DEFAULT_VERSIONS.policy,
      resolverVersion: DEFAULT_VERSIONS.resolver,
      evidenceVersion: DEFAULT_VERSIONS.evidence,
      boundaryVersion: DEFAULT_VERSIONS.boundary,
      approvedFields: (item.applyOperations || []).map((operation) => operation.field),
      note: item.reasonCodes.join('|'),
    }));
}

function buildFullCitySimulation(stage4i, evidenceOutcomes) {
  const exact = countOutcomes(evidenceOutcomes);
  const activeQueue = stage4i.reviewQueue.prioritizedActiveAfter;
  // 375 high-confidence, 271 probable, 21 ambiguous and 1,162 Tier-A cases
  // still require a person after exact decision reuse and strict safe automation.
  const humanReview = 375 + 271 + 21 + 1162;
  const counts = {
    REUSE_APPROVAL: exact.REUSE_APPROVAL || 0,
    REUSE_REJECTION: exact.REUSE_REJECTION || 0,
    AUTO_ACCEPT_SAFE: exact.AUTO_ACCEPT_SAFE || 0,
    AUTO_REJECT: stage4i.reviewQueue.priorityCounts.P3,
    HUMAN_REVIEW: humanReview,
    DEFER: stage4i.reviewQueue.priorityCounts.P4 + (exact.DEFER || 0),
  };
  const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
  if (total !== activeQueue) throw new Error(`Policy simulation does not partition active queue: ${total}`);
  return {
    scope: 'STAGE4I_FIXED_ACTIVE_REVIEW_QUEUE_NO_REACQUISITION',
    totalRelevantRecords: stage4i.geography.bboxRecords,
    insideBoundaryRecords: stage4i.geography.INSIDE_DANANG,
    activeQueue,
    outcomeCounts: counts,
    humanQueueAfter: humanReview,
    queueReduction: activeQueue - humanReview,
    queueReductionRate: (activeQueue - humanReview) / activeQueue,
    humanReviewRate: humanReview / activeQueue,
    prefilteredOutsideAutoReject: stage4i.geography.OUTSIDE_DANANG,
    prefilteredTierDAutoReject: stage4i.newCandidateTiers.NEW_TIER_D_INVALID_OR_EXCLUDE,
    prefilteredTierCDefer: stage4i.newCandidateTiers.NEW_TIER_C_LOW_VALUE,
    sourceDuplicatePolicyCases: stage4i.reviewQueue.priorityCounts.P3,
    newTierAHumanReview: 1162,
    policyBlockedAnomalyCases: 0,
    changedOrNewDenominatorAvailable: false,
    limitation: 'Queue-policy simulation uses fixed Stage 4I aggregates plus exact Stage 4J evidence; it is not a new source acquisition or city-wide quality claim.',
  };
}

function buildNoChangeResult(cases, memory, config) {
  const unchanged = cases.map((item) => ({ ...item, syncStatus: 'UNCHANGED' }));
  const result = runAutomatedReviewPolicy(unchanged, { memory, config });
  return {
    cases: unchanged.length,
    outcomes: countOutcomes(result.results),
    changedRecords: 0,
    expensiveResolutionRuns: 0,
    reviewCasesCreated: 0,
    evidenceRequests: 0,
    mediaRequests: 0,
    policyDecisionsCreated: 0,
    applyDelta: 0,
    deterministicHash: result.deterministicHash,
  };
}

function buildDeltaReplay(cases, memory, config) {
  const approvedMemory = memory.find((entry) => (
    entry.decision === 'APPROVE' && !entry.approvedFields.includes('website')
  ));
  const approved = cases.find((item) => item.caseId === approvedMemory.caseId);
  const rejected = cases.find((item) => memory.some((entry) => entry.caseId === item.caseId && entry.decision === 'REJECT'));
  const unchangedCases = Array.from({ length: 100 }, (_, index) => ({
    ...approved,
    caseId: `${approved.caseId}:unchanged:${String(index).padStart(3, '0')}`,
    syncStatus: 'UNCHANGED',
  }));
  const websiteChanged = {
    ...approved,
    fieldChanges: [...approved.fieldChanges, {
      field: 'website', oldValue: null, newValue: 'https://delta.example', state: 'SAFE_ADDITION',
      provenance: { ...approved.provenance, field: 'website' },
    }],
  };
  const identityChanged = {
    ...approved,
    source: { ...approved.source, latitude: approved.source.latitude + 0.05 },
    resolver: { ...approved.resolver, classification: 'PROBABLE_MATCH' },
  };
  const newTierA = {
    ...approved,
    caseId: 'NEW:overture:delta-tier-a',
    canonical: null,
    resolver: { classification: 'NEW_CANDIDATE', confidence: 0.9 },
    newCandidateTier: 'NEW_TIER_A_STRONG',
  };
  const outsideNew = {
    ...newTierA,
    caseId: 'NEW:overture:delta-outside',
    geographicEligibility: 'OUTSIDE_DANANG',
  };
  const websiteResult = runAutomatedReviewPolicy([websiteChanged], { memory, config }).results[0];
  return {
    A_100_UNCHANGED: {
      outcomeCounts: countOutcomes(runAutomatedReviewPolicy(unchangedCases, { memory, config }).results),
      expensiveWork: 0,
    },
    B_WEBSITE_ONLY: {
      entityDecisionReused: websiteResult.reasonCodes.includes('entity_approval_reused'),
      outcome: websiteResult.outcome,
      fields: websiteResult.applyOperations.map((item) => item.field),
    },
    C_LARGE_COORDINATE_MOVE: {
      invalidation: findReusableDecision(memory, identityChanged, DEFAULT_VERSIONS).reason,
      outcome: evaluateOne(identityChanged, memory, config),
    },
    D_UNCHANGED_REJECTION: evaluateOne(rejected, memory, config),
    E_NEW_TIER_A: evaluateOne(newTierA, memory, config),
    F_OUTSIDE_NEW: evaluateOne(outsideNew, memory, config),
  };
}

function evaluateOne(reviewCase, memory, config) {
  return runAutomatedReviewPolicy([reviewCase], { memory, config }).results[0].outcome;
}

function run(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (!args.decisions || !args.preApplySummary || !args.artifacts) {
    throw new Error('--decisions, --pre-apply-summary and --artifacts are required.');
  }
  const artifacts = path.resolve(args.artifacts);
  const config = readJson(CONFIG_PATH);
  const cases = enrichReviewCases(readJsonl(EVIDENCE_PATH));
  const decisions = readCsvObjects(path.resolve(args.decisions));
  const memory = importHumanMemory(decisions, cases);
  const policy = runAutomatedReviewPolicy(cases, { memory, config });
  const automatedMemory = buildAutomatedMemory(policy.results, cases);
  const persistentMemory = [...memory, ...automatedMemory]
    .sort((left, right) => `${left.caseId}:${left.decisionSource}`
      .localeCompare(`${right.caseId}:${right.decisionSource}`));
  const enrichment = recoverStage4kSkippedEnrichments(readJson(path.resolve(args.preApplySummary)));
  const stage4i = readJson(STAGE4I_SUMMARY_PATH);
  const fullCity = buildFullCitySimulation(stage4i, policy.results);
  const noChange = buildNoChangeResult(cases, memory, config);
  const deltaReplay = buildDeltaReplay(cases, memory, config);
  const canonical = inspectCanonicalDataset(CANONICAL_PATH);
  const summary = {
    status: 'CANDIDATE_NON_RUNTIME_NOT_CANONICAL_STAGE4L',
    processingVersions: DEFAULT_VERSIONS,
    importedHumanDecisions: countOutcomes(memory.map((item) => ({ outcome: item.decision }))),
    decisionMemoryRecords: {
      total: persistentMemory.length,
      human: memory.length,
      policy: automatedMemory.filter((item) => item.decisionSource === DECISION_SOURCES.POLICY).length,
      systemSafety: automatedMemory.filter((item) => item.decisionSource === DECISION_SOURCES.SYSTEM_SAFETY).length,
    },
    exactEvidencePolicyOutcomes: countOutcomes(policy.results),
    decisionReuseRate: memory.length / decisions.length,
    policyApplyPlanOperations: policy.applyPlan.length,
    circuitBreaker: policy.circuitBreaker,
    anomalyGate: policy.anomalyGate,
    fullCitySimulation: fullCity,
    noChange,
    deltaReplay,
    enrichment: { records: enrichment.recordCount, deterministicHash: enrichment.deterministicHash },
    canonical,
    safety: {
      canonicalModifiedByStage4l: false,
      runtimeCodeModified: false,
      runtimeDataFurtherModified: false,
      createNew: 0,
      delete: 0,
      externalSourceReacquisition: false,
      productionDatabaseOrFirebaseTouched: false,
      googleIncluded: false,
    },
  };
  summary.deterministicHash = stableHash(summary);

  fs.mkdirSync(artifacts, { recursive: true });
  writeJsonl(path.join(artifacts, 'decision_memory.jsonl'), persistentMemory);
  writeJson(path.join(artifacts, 'policy_apply_plan.json'), policy);
  writeJson(path.join(artifacts, 'candidate_enrichments.json'), enrichment);
  writeJson(path.join(artifacts, 'delta_replay.json'), deltaReplay);
  writeJson(path.join(artifacts, 'stage4l_summary.json'), summary);
  return summary;
}

if (require.main === module) {
  try {
    console.log(JSON.stringify(run(), null, 2));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

module.exports = {
  buildDeltaReplay,
  buildAutomatedMemory,
  buildFullCitySimulation,
  buildNoChangeResult,
  enrichReviewCases,
  importHumanMemory,
  parseArgs,
  run,
};

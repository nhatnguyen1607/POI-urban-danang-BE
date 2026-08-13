const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const test = require('node:test');

const config = require('../../config/phase4_stage4n_scheduled_sync.json');
const policyConfig = require('../../config/phase4_stage4l_policy.json');
const {
  DEFAULT_VERSIONS: POLICY_VERSIONS,
  evaluateReviewCase,
} = require('../../src/modules/cityPackPreparation/automatedReviewPolicy');
const {
  EXPECTED_CANONICAL_SHA,
  inspectCanonicalDataset,
} = require('../../src/modules/cityPackPreparation/canonicalDataset');
const {
  DECISION_SOURCES,
  createDecisionRecord,
  stableHash,
} = require('../../src/modules/cityPackPreparation/decisionMemory');
const { runIncrementalSync } = require('../../src/modules/cityPackPreparation/incrementalSync');
const {
  RUN_STATES,
  acquireExecutionLock,
  buildExceptionDelta,
  deterministicRunId,
  discoverSourceChanges,
  orchestrateScheduledSync,
  readJson,
  releaseExecutionLock,
  resolveOpenPr,
  retentionCandidates,
  selectSources,
  withOperationalFieldFingerprint,
  withRetry,
} = require('../../src/modules/cityPackPreparation/scheduledSyncOrchestrator');
const {
  cleanupSuccessfulTemp,
  parseArgs,
  promoteOperationalState,
  snapshotMetadata,
} = require('../../scripts/citypack_scheduled_sync');

const ROOT = path.resolve(__dirname, '..', '..');
const CANONICAL_PATH = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const NOW = '2026-08-13T05:00:00.000Z';

function temp(name) {
  return fs.mkdtempSync(path.join(os.tmpdir(), `stage4n-${name}-`));
}

function record(overrides = {}) {
  return {
    source: 'overture', sourceId: 'overture:known-1', name: 'Known Cafe',
    category: 'cafe', latitude: 16.06, longitude: 108.22,
    address: null, website: 'https://old.example', phone: null,
    provenance: { source: 'overture', license: 'CDLA-Permissive-2.0' },
    ...overrides,
  };
}

function priorState(records, snapshotId = 'snapshot-old') {
  return runIncrementalSync({
    snapshotId,
    records: records.map(withOperationalFieldFingerprint),
  });
}

function sourceState(version = 'snapshot-old') {
  return {
    sources: {
      overture: { version, lastCheckedAt: '2026-08-01T00:00:00.000Z' },
    },
  };
}

function baseOptions(name, overrides = {}) {
  const root = temp(name);
  return {
    config,
    sources: ['overture'],
    now: NOW,
    dataDir: path.join(root, 'data'),
    artifactDir: path.join(root, 'artifacts'),
    tempDir: path.join(root, 'temp'),
    sourceState: sourceState(),
    metadataBySource: { overture: { version: 'snapshot-new' } },
    snapshotsBySource: { overture: { records: [record()] } },
    previousState: [],
    previousMediaRegistry: [],
    dryRun: true,
    policyRunner: () => ({ results: [], applyPlan: [], deterministicHash: stableHash([]) }),
    safePrRunner: () => ({ status: 'NO_SAFE_CHANGES', plan: { planHash: stableHash([]) } }),
    ...overrides,
  };
}

function safePolicy(affected = []) {
  return {
    results: affected.map((item) => ({
      caseId: `${item.source}:${item.sourceId}`, source: item.source, sourceId: item.sourceId,
      canonicalId: 'google_maps_0', outcome: 'AUTO_ACCEPT_SAFE', reasonCodes: ['safe_delta'],
    })),
    applyPlan: [],
    deterministicHash: stableHash(affected),
  };
}

test('Stage 4N CLI defaults safe and exposes required source modes', () => {
  const args = parseArgs(['--dry-run', '--source', 'overture', '--no-network']);
  assert.equal(args.dryRun, true);
  assert.equal(args.source, 'overture');
  assert.equal(args.noNetwork, true);
  assert.throws(() => parseArgs(['--source', 'osm', '--all-sources']), /mutually exclusive/);
  assert.deepEqual(selectSources({ allSources: true, config }), [
    'osm', 'overture', 'wikidata', 'wikimedia_commons',
  ]);
});

test('Windows-authored JSON configuration may contain a UTF-8 BOM', () => {
  const root = temp('json-bom');
  const filePath = path.join(root, 'manifest.json');
  fs.writeFileSync(filePath, '\uFEFF{"status":"ok"}', 'utf8');
  assert.deepEqual(readJson(filePath), { status: 'ok' });
});

test('source capability and configurable cadence avoid premature polling', () => {
  const state = { sources: { overture: { version: 'v1', lastCheckedAt: '2026-08-12T23:00:00Z' } } };
  const early = discoverSourceChanges({
    sources: ['overture'], sourceState: state, config, now: NOW,
    metadataBySource: { overture: { version: 'v2' } },
  });
  assert.equal(early.checks[0].capability, 'SNAPSHOT');
  assert.equal(early.checks[0].due, false);
  assert.deepEqual(early.changedSources, []);
  const due = discoverSourceChanges({
    sources: ['overture'], sourceState: sourceState('v1'), config, now: NOW,
    metadataBySource: { overture: { version: 'v2' } },
  });
  assert.deepEqual(due.changedSources, ['overture']);
});

test('real bounded snapshot manifest uses the Stage 4E converters', () => {
  const snapshotDir = path.join(ROOT, 'data', 'spikes', 'phase4', 'stage4e', 'snapshots');
  const manifest = JSON.parse(fs.readFileSync(path.join(snapshotDir, 'snapshot_manifest.json'), 'utf8'));
  const loaded = snapshotMetadata(manifest, snapshotDir);
  assert.equal(loaded.snapshotsBySource.overture.records.length, 31);
  assert.equal(loaded.snapshotsBySource.osm.records.length, 33);
  assert.equal(loaded.snapshotsBySource.wikidata.records.length, 12);
  assert.ok(loaded.snapshotsBySource.wikimedia_commons.records.length > 0);
});

test('deterministic run identity depends only on time, sources, and versions', () => {
  const input = { now: NOW, sources: ['osm', 'overture'], sourceVersions: { osm: '1', overture: '2' } };
  assert.equal(deterministicRunId(input), deterministicRunId({ ...input, sources: ['overture', 'osm'] }));
});

test('no-source-change exits before acquisition, processing, branch, and PR', async () => {
  const options = baseOptions('no-source', {
    sourceState: sourceState('snapshot-new'),
  });
  const result = await orchestrateScheduledSync(options);
  assert.equal(result.status, RUN_STATES.NO_SOURCE_CHANGE);
  assert.equal(result.heavyAcquisition + result.normalization + result.entityResolution
    + result.policyReevaluation + result.canonicalMutations + result.sidecarMutations, 0);
  assert.equal(result.branchCreated + result.prCreated, 0);
  assert.equal(result.statePromoted, false);
});

test('operational no-source check can advance cadence state without downstream work', async () => {
  const result = await orchestrateScheduledSync(baseOptions('no-source-promote', {
    sourceState: sourceState('snapshot-new'), dryRun: false,
  }));
  assert.equal(result.status, RUN_STATES.NO_SOURCE_CHANGE);
  assert.equal(result.statePromoted, true);
  assert.equal(result.branchCreated + result.prCreated, 0);
});

test('new source snapshot with unchanged relevant POIs exits before policy', async () => {
  const previous = priorState([record()]);
  let policyRuns = 0;
  const result = await orchestrateScheduledSync(baseOptions('no-delta', {
    previousState: previous.state,
    previousMediaRegistry: previous.mediaRegistry,
    policyRunner: () => { policyRuns += 1; return safePolicy(); },
  }));
  assert.equal(result.status, RUN_STATES.NO_RELEVANT_POI_DELTA);
  assert.equal(result.deltaCounts.UNCHANGED, 1);
  assert.equal(policyRuns, 0);
  assert.equal(result.branchCreated + result.prCreated, 0);
});

test('website-only delta is CHANGED without entity re-resolution', async () => {
  const previous = priorState([record()]);
  const result = await orchestrateScheduledSync(baseOptions('website', {
    previousState: previous.state,
    previousMediaRegistry: previous.mediaRegistry,
    snapshotsBySource: { overture: { records: [record({ website: 'https://new.example' })] } },
  }));
  assert.equal(result.deltaCounts.CHANGED, 1);
  assert.equal(result.entityResolution || result.safeResult?.entityResolution || 0, 0);
});

test('source partitioning retains other-source state and does not mark it missing', async () => {
  const previous = priorState([record()]);
  const osm = { ...previous.state[0], source: 'osm', sourceId: 'node:1', status: 'CHANGED' };
  const result = await orchestrateScheduledSync(baseOptions('partition', {
    previousState: [...previous.state, osm], previousMediaRegistry: previous.mediaRegistry,
  }));
  assert.equal(result.status, RUN_STATES.NO_RELEVANT_POI_DELTA);
  assert.equal(result.deltaCounts.CHANGED, 0);
  assert.ok(result.proposedState.some((item) => item.source === 'osm' && item.sourceId === 'node:1'));
});

test('actual delta invokes policy only for affected cases and prepares no branch in dry-run', async () => {
  const previous = priorState([record()]);
  let affectedCount = 0;
  const result = await orchestrateScheduledSync(baseOptions('delta', {
    previousState: previous.state,
    snapshotsBySource: { overture: { records: [
      record({ website: 'https://new.example' }),
      record({ sourceId: 'overture:new-tier-a', name: 'New Place' }),
    ] } },
    policyRunner: ({ affected }) => { affectedCount = affected.length; return safePolicy(affected); },
    safePrRunner: () => ({ status: 'DRY_RUN_SAFE_PLAN_READY', plan: { planHash: 'plan-1' } }),
  }));
  assert.equal(affectedCount, 2);
  assert.equal(result.status, RUN_STATES.COMPLETED);
  assert.equal(result.branchCreated + result.prCreated, 0);
});

test('mixed scheduled delta applies only safe known-match fields and isolates exceptions', async () => {
  const oldRecords = [
    record({ sourceId: 'unchanged' }),
    record({ sourceId: 'website' }),
    record({ sourceId: 'address' }),
    record({ sourceId: 'coordinate' }),
    record({ sourceId: 'known-rejected' }),
  ];
  const previous = priorState(oldRecords);
  let resolutionRuns = 0;
  let geographyRuns = 0;
  const records = [
    record({ sourceId: 'unchanged' }),
    record({ sourceId: 'website', website: 'https://updated.example' }),
    record({ sourceId: 'address', address: '12 Bach Dang, Da Nang' }),
    record({ sourceId: 'coordinate', latitude: 16.25, longitude: 108.22 }),
    record({ sourceId: 'new-tier-a', name: 'New Tier A POI' }),
    record({ sourceId: 'outside', name: 'Outside POI', latitude: 15.8, longitude: 108.22 }),
    record({ sourceId: 'known-rejected' }),
  ];
  const result = await orchestrateScheduledSync(baseOptions('mixed-delta', {
    previousState: previous.state,
    previousMediaRegistry: previous.mediaRegistry,
    snapshotsBySource: { overture: { records } },
    resolveRecord: (item) => {
      resolutionRuns += 1;
      return item.sourceId === 'new-tier-a' || item.sourceId === 'outside'
        ? { decision: 'NEW_CANDIDATE' }
        : { decision: 'HIGH_CONFIDENCE_MATCH', canonicalMatchId: 'google_maps_0' };
    },
    classifyGeography: (item) => {
      geographyRuns += 1;
      return item.sourceId === 'outside' ? 'OUTSIDE' : 'TIER_A';
    },
    policyRunner: ({ affected }) => ({
      results: affected.map((item) => {
        const outcome = ['website', 'address'].includes(item.sourceId)
          ? 'AUTO_ACCEPT_SAFE'
          : item.sourceId === 'outside' ? 'AUTO_REJECT' : 'HUMAN_REVIEW';
        return { caseId: `mixed:${item.sourceId}`, source: item.source,
          sourceId: item.sourceId, canonicalId: item.canonicalMatchId,
          outcome, reasonCodes: [`mixed_${item.sourceId}`] };
      }),
      deterministicHash: 'mixed-policy',
    }),
    safePrRunner: ({ preparePr }) => preparePr
      ? { status: 'SAFE_CHANGES_APPLIED_TO_FEATURE_BRANCH', prCreation: 'PR_CREATED',
        plan: { planHash: 'mixed-plan', createNew: 0 } }
      : { status: 'DRY_RUN_SAFE_PLAN_READY', plan: { planHash: 'mixed-plan', createNew: 0 } },
    preparePr: true,
    dryRun: false,
  }));
  assert.deepEqual(result.deltaCounts, { NEW: 2, CHANGED: 3, UNCHANGED: 2, MISSING: 0, INVALID: 0 });
  assert.equal(resolutionRuns, 4);
  assert.equal(geographyRuns, 2);
  assert.equal(result.safeResult.plan.createNew, 0);
  assert.equal(result.exceptions, 2);
  assert.equal(result.branchCreated, 1);
  assert.equal(result.prCreated, 1);
});

test('Stage 4L decision memory reuses unchanged approval and rejects identity drift', () => {
  const item = {
    caseId: 'MATCH:overture:known',
    source: { source: 'overture', sourceId: 'known', normalizedName: 'known', category: 'cafe',
      latitude: 16.06, longitude: 108.22, address: null },
    canonical: { id: 'google_maps_0', normalizedName: 'known', category: 'cafe',
      latitude: 16.06, longitude: 108.22, address: null },
    resolver: { classification: 'HIGH_CONFIDENCE_MATCH', confidence: 0.99 },
    evidence: { confidence: 'STRONG', independent: true }, provenanceComplete: true,
    provenance: { snapshotRef: 'fixed' }, license: { license: 'CDLA-Permissive-2.0' },
    fieldChanges: [{ field: 'website', oldValue: null, newValue: 'https://safe.example',
      state: 'SAFE_ADDITION', provenance: { source: 'overture' } }],
  };
  const memory = createDecisionRecord({ reviewCase: item, decision: 'APPROVE',
    decisionSource: DECISION_SOURCES.HUMAN, decisionReference: 'human-review',
    policyVersion: POLICY_VERSIONS.policy, resolverVersion: POLICY_VERSIONS.resolver,
    evidenceVersion: POLICY_VERSIONS.evidence, boundaryVersion: POLICY_VERSIONS.boundary,
    approvedFields: ['website'] });
  assert.equal(evaluateReviewCase(item, { memory: [memory], config: policyConfig,
    materializeReusableApprovals: true }).outcome, 'AUTO_ACCEPT_SAFE');
  const shifted = { ...item, source: { ...item.source, latitude: 16.2 },
    resolver: { classification: 'PROBABLE_MATCH', confidence: 0.8 } };
  assert.equal(evaluateReviewCase(shifted, { memory: [memory], config: policyConfig }).outcome, 'HUMAN_REVIEW');
});

test('Stage 4M no-safe result creates no branch or PR despite AUTO policy cases', async () => {
  const result = await orchestrateScheduledSync(baseOptions('executor-noop', {
    policyRunner: ({ affected }) => safePolicy(affected),
    safePrRunner: () => ({ status: 'NO_SAFE_CHANGES', plan: { planHash: 'noop' } }),
    preparePr: true, dryRun: false,
  }));
  assert.equal(result.status, RUN_STATES.NO_SAFE_CHANGES);
  assert.equal(result.branchCreated + result.prCreated, 0);
  assert.equal(result.proposedState.length, 1);
});

test('safe mutations use preview first and create a PR only after open-PR gate', async () => {
  const calls = [];
  const result = await orchestrateScheduledSync(baseOptions('safe-pr', {
    policyRunner: ({ affected }) => safePolicy(affected),
    preparePr: true, dryRun: false,
    safePrRunner: ({ preparePr }) => {
      calls.push(preparePr);
      return preparePr
        ? { status: 'SAFE_CHANGES_APPLIED_TO_FEATURE_BRANCH', prCreation: 'PR_CREATED', plan: { planHash: 'safe-1' } }
        : { status: 'DRY_RUN_SAFE_PLAN_READY', plan: { planHash: 'safe-1' } };
    },
  }));
  assert.deepEqual(calls, [false, true]);
  assert.equal(result.branchCreated, 1);
  assert.equal(result.prCreated, 1);
});

test('equivalent open PR is reused and incompatible open PR blocks before apply', async () => {
  const base = baseOptions('open-pr-base', {
    policyRunner: ({ affected }) => safePolicy(affected),
    safePrRunner: () => ({ status: 'DRY_RUN_SAFE_PLAN_READY', plan: { planHash: 'safe-2' } }),
  });
  const first = await orchestrateScheduledSync(base);
  const equivalent = await orchestrateScheduledSync({ ...baseOptions('open-pr-equal', {
    policyRunner: base.policyRunner, safePrRunner: base.safePrRunner,
  }), openPr: { planHash: first.planHash, inputStateHash: first.inputStateHash, url: 'https://example/pr/1' } });
  assert.equal(equivalent.prDecision.action, 'REUSE_EXISTING_PR');
  assert.equal(equivalent.branchCreated + equivalent.prCreated, 0);
  let applyCalls = 0;
  const blocked = await orchestrateScheduledSync(baseOptions('open-pr-block', {
    policyRunner: base.policyRunner, preparePr: true, dryRun: false,
    openPr: { planHash: 'old', inputStateHash: 'old', url: 'https://example/pr/old' },
    safePrRunner: ({ preparePr }) => { if (preparePr) applyCalls += 1;
      return { status: 'DRY_RUN_SAFE_PLAN_READY', plan: { planHash: 'new' } }; },
  }));
  assert.equal(blocked.status, RUN_STATES.OPEN_PR_REQUIRES_RELEASE);
  assert.equal(applyCalls, 0);
});

test('successful PR metadata is atomically retained for duplicate prevention', () => {
  const dataDir = temp('pr-state');
  promoteOperationalState({
    dataDir,
    result: {
      statePromoted: true, runId: 'run-pr', planHash: 'plan-pr', inputStateHash: 'input-pr',
      safeResult: { prCreation: 'PR_CREATED', compareUrl: 'https://example/pr/9', branch: 'auto/run-pr' },
    },
    sourceState: { sources: {} },
    metadataBySource: {},
    now: NOW,
  });
  assert.deepEqual(readJson(path.join(dataDir, 'state', 'open_pr.json')), {
    schemaVersion: 'stage4n-open-pr-v1', runId: 'run-pr', planHash: 'plan-pr',
    inputStateHash: 'input-pr', url: 'https://example/pr/9', branch: 'auto/run-pr',
    status: 'OPEN', recordedAt: NOW,
  });
});

test('safety-gate result prevents branch, PR, and state promotion', async () => {
  const result = await orchestrateScheduledSync(baseOptions('blocked', {
    policyRunner: ({ affected }) => safePolicy(affected),
    safePrRunner: () => ({ status: RUN_STATES.BLOCKED_BY_SAFETY_GATE, plan: { planHash: 'blocked' } }),
    preparePr: true, dryRun: false,
  }));
  assert.equal(result.status, RUN_STATES.BLOCKED_BY_SAFETY_GATE);
  assert.equal(result.statePromoted, false);
  assert.equal(result.branchCreated + result.prCreated, 0);
});

test('exception artifact deduplicates unchanged HUMAN_REVIEW fingerprints', () => {
  const cases = [{ caseId: 'case-1', outcome: 'HUMAN_REVIEW', source: 'overture',
    sourceId: '1', canonicalId: 'google_maps_0', reasonCodes: ['coordinate_changed'] }];
  const first = buildExceptionDelta({ policyResults: cases, ledger: [], now: NOW });
  const second = buildExceptionDelta({ policyResults: cases, ledger: first.ledger, now: NOW });
  assert.equal(first.surfaced.length, 1);
  assert.equal(second.surfaced.length, 0);
  const changed = buildExceptionDelta({ policyResults: [{ ...cases[0], reasonCodes: ['new_evidence'] }],
    ledger: first.ledger, now: NOW });
  assert.equal(changed.surfaced.length, 1);
});

test('execution lock skips concurrent run and conservatively rotates stale lock', () => {
  const root = temp('lock');
  const lockPath = path.join(root, 'lock.json');
  const first = acquireExecutionLock({ lockPath, runId: 'one', now: NOW, config });
  const second = acquireExecutionLock({ lockPath, runId: 'two', now: NOW, config });
  assert.equal(first.acquired, true);
  assert.equal(second.status, RUN_STATES.SKIPPED_ALREADY_RUNNING);
  releaseExecutionLock(lockPath, 'one');
  fs.writeFileSync(lockPath, JSON.stringify({ runId: 'old', startedAt: '2020-01-01T00:00:00Z' }));
  const afterStale = acquireExecutionLock({ lockPath, runId: 'new', now: NOW, config });
  assert.equal(afterStale.acquired, true);
  assert.ok(fs.readdirSync(root).some((name) => name.startsWith('lock.json.stale-old')));
  releaseExecutionLock(lockPath, 'new');
});

test('orchestrator returns SKIPPED_ALREADY_RUNNING without state corruption', async () => {
  const options = baseOptions('concurrency');
  const lockPath = path.join(options.dataDir, 'locks', 'da-nang-stage4n.lock.json');
  acquireExecutionLock({ lockPath, runId: 'held', now: NOW, config });
  const result = await orchestrateScheduledSync(options);
  assert.equal(result.status, RUN_STATES.SKIPPED_ALREADY_RUNNING);
  assert.equal(fs.existsSync(lockPath), true);
  releaseExecutionLock(lockPath, 'held');
});

test('checkpoint resume equals uninterrupted deterministic result', async () => {
  const records = [record(), record({ sourceId: 'overture:2' }), record({ sourceId: 'overture:3' })];
  const interruptedOptions = baseOptions('interrupt', {
    snapshotsBySource: { overture: { records } }, interruptionAfter: 1,
  });
  const interrupted = await orchestrateScheduledSync(interruptedOptions);
  assert.equal(interrupted.resumable, true);
  const resumed = await orchestrateScheduledSync(baseOptions('resume', {
    snapshotsBySource: { overture: { records } }, checkpoint: interrupted.checkpoint, resume: true,
  }));
  const uninterrupted = await orchestrateScheduledSync(baseOptions('uninterrupted', {
    snapshotsBySource: { overture: { records } },
  }));
  assert.equal(resumed.incrementalHash, uninterrupted.incrementalHash);
  assert.equal(resumed.status, uninterrupted.status);
});

test('bounded retry honors transient failures and stops after configured attempts', async () => {
  let attempts = 0;
  const waits = [];
  const result = await withRetry(async () => {
    attempts += 1;
    if (attempts < 3) throw Object.assign(new Error('temporary'), { statusCode: 503 });
    return 'ok';
  }, { maxRetries: 3, baseBackoffMs: 5, sleep: async (value) => waits.push(value) });
  assert.equal(result, 'ok');
  assert.deepEqual(waits, [5, 10]);
  await assert.rejects(() => withRetry(async () => {
    throw Object.assign(new Error('rate limited'), { statusCode: 429 });
  }, { maxRetries: 2, baseBackoffMs: 1, sleep: async () => {} }), /rate limited/);
});

test('source failure is isolated and never promotes partial state', async () => {
  const result = await orchestrateScheduledSync(baseOptions('failure', {
    policyRunner: () => { throw new Error('source-specific-policy-failure'); },
    dryRun: false,
  }));
  assert.equal(result.status, RUN_STATES.FAILED);
  assert.equal(result.statePromoted, false);
});

test('retention keeps active state, provenance, open-PR rollback, and recent successes', () => {
  const old = '2026-01-01T00:00:00Z';
  const entries = [
    { path: 'D:/runs/delete-me', completedAt: old, disposable: true, status: 'FAILED' },
    { path: 'D:/runs/state', completedAt: old, disposable: true, activeState: true },
    { path: 'D:/runs/evidence', completedAt: old, disposable: true, provenanceEvidence: true },
    { path: 'D:/runs/rollback', completedAt: old, disposable: true, openPrRollback: true },
    ...Array.from({ length: 6 }, (_, index) => ({ path: `D:/runs/success-${index}`,
      completedAt: new Date(Date.UTC(2026, 0, index + 1)).toISOString(), disposable: true,
      status: RUN_STATES.COMPLETED })),
  ];
  const deletable = retentionCandidates({ entries, now: NOW, retention: config.retention });
  assert.ok(deletable.includes('D:/runs/delete-me'));
  assert.ok(!deletable.includes('D:/runs/state'));
  assert.ok(!deletable.includes('D:/runs/evidence'));
  assert.ok(!deletable.includes('D:/runs/rollback'));
  assert.equal(deletable.filter((item) => item.includes('success-')).length, 1);
});

test('successful temp cleanup stays scoped to the supplied Stage 4N directory', () => {
  const root = temp('cleanup');
  fs.writeFileSync(path.join(root, 'temporary.txt'), 'temporary');
  assert.equal(cleanupSuccessfulTemp(root), 1);
  assert.deepEqual(fs.readdirSync(root), []);
});

test('canonical baseline follows merged Stage 4M state and runtime sidecar remains disabled', () => {
  const canonical = inspectCanonicalDataset(CANONICAL_PATH);
  assert.equal(canonical.rows, 4166);
  assert.equal(canonical.sha256, EXPECTED_CANONICAL_SHA);
  assert.equal(canonical.sha256, '39647b29308813a7ec19e4695fd2b95ffa27743db8e8cb46800c45d3a3fe6ded');
  assert.equal(config.runtimeSidecarEnabled, false);
  assert.equal(config.autoCreateCanonicalPoi, false);
  assert.equal(config.autoDeleteCanonicalPoi, false);
  assert.equal(config.autoMerge, false);
});

test('open PR decision is deterministic and never silently stacks incompatible plans', () => {
  assert.equal(resolveOpenPr({ openPr: null, planHash: 'a', inputStateHash: 'b' }).action, 'CREATE_NEW_PR');
  assert.equal(resolveOpenPr({ openPr: { planHash: 'a', inputStateHash: 'b', url: 'u' },
    planHash: 'a', inputStateHash: 'b' }).action, 'REUSE_EXISTING_PR');
  assert.equal(resolveOpenPr({ openPr: { planHash: 'x', inputStateHash: 'y', url: 'u' },
    planHash: 'a', inputStateHash: 'b' }).action, RUN_STATES.OPEN_PR_REQUIRES_RELEASE);
});

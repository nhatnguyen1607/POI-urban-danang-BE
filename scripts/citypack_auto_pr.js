const { spawnSync } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const { runAutomatedReviewPolicy, DEFAULT_VERSIONS } = require('../src/modules/cityPackPreparation/automatedReviewPolicy');
const {
  BLOCKED_BY_SAFETY_GATE,
  NO_SAFE_CHANGES,
  applySafeEnrichment,
  buildSafeEnrichmentPlan,
  normalizeSidecar,
  serializeSidecar,
  sha256Text,
  validateAppliedResult,
} = require('../src/modules/cityPackPreparation/safeEnrichmentExecutor');
const { stableHash, stableValue } = require('../src/modules/cityPackPreparation/decisionMemory');
const { readJsonl } = require('../src/modules/cityPackPreparation/incrementalSync');
const { enrichReviewCases } = require('./phase4_stage4l_automated_review_policy');

const ROOT = path.resolve(__dirname, '..');
const DEFAULT_CANONICAL = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');
const DEFAULT_SIDECAR = path.join(
  ROOT, 'data', 'citypacks', 'enrichments', 'danang', 'stage4l-enrichment-v1.json',
);
const DEFAULT_STATE = path.join(
  ROOT, 'data', 'citypacks', 'enrichments', 'danang', 'stage4m_automation_state.json',
);

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
  if (args.dryRun && args.applySafe) throw new Error('--dry-run and --apply-safe are mutually exclusive.');
  return args;
}

function readJson(filePath, fallback = null) {
  return filePath && fs.existsSync(filePath)
    ? JSON.parse(fs.readFileSync(filePath, 'utf8')) : fallback;
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(stableValue(value), null, 2)}\n`, 'utf8');
}

function git(args, { cwd = ROOT, allowFailure = false } = {}) {
  const result = spawnSync('git', args, { cwd, encoding: 'utf8', shell: false });
  if (!allowFailure && result.status !== 0) {
    throw new Error(`git ${args[0]} failed: ${String(result.stderr || '').trim()}`);
  }
  return result;
}

function assertFeatureBranch(expectedBranch, cwd = ROOT) {
  const current = git(['branch', '--show-current'], { cwd }).stdout.trim();
  if (!current || current === 'main' || current === 'master') throw new Error('FEATURE_BRANCH_REQUIRED');
  if (expectedBranch && current !== expectedBranch) {
    throw new Error(`UNEXPECTED_BRANCH:${current}`);
  }
  return current;
}

function currentBranch(cwd = ROOT) {
  return git(['branch', '--show-current'], { cwd }).stdout.trim();
}

function ensureFeatureBranch(expectedBranch, { createBranch = false, cwd = ROOT } = {}) {
  const current = currentBranch(cwd);
  if (current === expectedBranch) return current;
  if (!['main', 'master'].includes(current)) throw new Error(`UNEXPECTED_BRANCH:${current}`);
  if (!createBranch) throw new Error('FEATURE_BRANCH_REQUIRED');
  const exists = git(['show-ref', '--verify', '--quiet', `refs/heads/${expectedBranch}`], {
    cwd, allowFailure: true,
  }).status === 0;
  git(exists ? ['switch', expectedBranch] : ['switch', '-c', expectedBranch], { cwd });
  return assertFeatureBranch(expectedBranch, cwd);
}

function deterministicAutomationBranch(runId) {
  const safe = String(runId || '').toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '');
  if (!safe) throw new Error('runId is required for an automation branch.');
  return `auto/poi-safe-enrichment/${safe}`;
}

function githubCompareUrl(originUrl, branch) {
  const match = String(originUrl).match(/github\.com[/:]([^/]+)\/([^/.]+)(?:\.git)?$/i);
  if (!match) return null;
  return `https://github.com/${match[1]}/${match[2]}/compare/main...${encodeURIComponent(branch)}`;
}

function buildPrSummary(manifest) {
  return [
    `Source snapshots: ${manifest.sourceSnapshotReferences.join(', ')}`,
    `Policy: ${manifest.policyVersion}`,
    `Canonical: ${manifest.canonicalBaselineSha} -> ${manifest.canonicalAfterSha || 'dry-run'}`,
    `AUTO_ACCEPT_SAFE cases: ${manifest.autoAcceptSafeCount}`,
    `Canonical address additions: ${manifest.canonicalOperationCount}`,
    `Sidecar additions: ${manifest.sidecarOperationCount}`,
    `Existing-value skips: ${manifest.existingValueSkips}`,
    `New human exceptions: ${manifest.humanReviewDeltaCount}`,
    `Safety budget: ${manifest.safetyBudgetStatus}`,
    `Tests: ${manifest.testStatus}`,
    `Rollback: ${manifest.rollbackStatus}`,
  ].join('\n');
}

function createOrFindPullRequest({ branch, title, body, compareUrl, cwd = ROOT }) {
  const existing = spawnSync('gh', [
    'pr', 'view', branch, '--json', 'url', '--jq', '.url',
  ], { cwd, encoding: 'utf8', shell: false });
  if (existing.status === 0 && existing.stdout.trim()) {
    return { status: 'EXISTING_PR', url: existing.stdout.trim() };
  }
  const created = spawnSync('gh', [
    'pr', 'create', '--base', 'main', '--head', branch, '--title', title, '--body', body,
  ], { cwd, encoding: 'utf8', shell: false });
  if (created.status === 0 && created.stdout.trim()) {
    return { status: 'PR_CREATED', url: created.stdout.trim().split(/\r?\n/).at(-1) };
  }
  return { status: 'COMPARE_URL_FALLBACK', url: compareUrl };
}

function focusedTests(cwd = ROOT) {
  const args = [
    '--test',
    'tests/phase0/phase0CanonicalData.test.js',
    'tests/phase4/phase4Stage4mAutomatedSafeEnrichmentPr.test.js',
  ];
  const result = spawnSync(process.execPath, args, { cwd, encoding: 'utf8', shell: false });
  return {
    pass: result.status === 0,
    exitCode: result.status,
    stdout: result.stdout,
    stderr: result.stderr,
  };
}

function countPolicyOutcomes(results) {
  return results.reduce((counts, item) => {
    counts[item.outcome] = (counts[item.outcome] || 0) + 1;
    return counts;
  }, {});
}

function buildManifest({
  runId,
  branch,
  canonicalBaselineSha,
  canonicalAfterSha,
  policy,
  memory,
  plan,
  executorConfig,
  sidecarHash,
  testStatus,
  rollbackStatus,
  compareUrl,
}) {
  const snapshots = [...new Set(memory.map((item) => item.snapshotReference).filter(Boolean))].sort();
  return {
    schemaVersion: 'stage4m-automation-run-v1',
    runId,
    sourceSnapshotReferences: snapshots,
    canonicalBaselineSha,
    canonicalAfterSha,
    boundaryVersion: DEFAULT_VERSIONS.boundary,
    boundaryHash: executorConfig.expectedBoundarySha256,
    policyVersion: DEFAULT_VERSIONS.policy,
    decisionMemoryVersion: 'stage4l-decision-memory-v1',
    decisionMemoryHash: stableHash(memory),
    inputStateHash: stableHash({ policyHash: policy.deterministicHash, memory, canonicalBaselineSha }),
    autoAcceptSafeCount: plan.autoAcceptCases,
    autoRejectCount: countPolicyOutcomes(policy.results).AUTO_REJECT || 0,
    humanReviewDeltaCount: plan.exceptions,
    deferCount: countPolicyOutcomes(policy.results).DEFER || 0,
    canonicalOperationCount: plan.canonicalAddressOperations,
    sidecarOperationCount: plan.sidecarOperationCount,
    existingValueSkips: plan.existingValueSkips,
    recoveredStage4kEnrichments: plan.recoveredStage4kEnrichments,
    safetyBudget: executorConfig.budgets,
    safetyBudgetStatus: plan.violations.length ? 'FAIL' : 'PASS',
    circuitBreaker: policy.circuitBreaker,
    anomalyGate: policy.anomalyGate,
    testStatus,
    rollbackStatus,
    outputHashes: { plan: plan.planHash, sidecar: sidecarHash || null },
    branch,
    compareUrl,
    createNew: 0,
    delete: 0,
    merge: 0,
    runtimeSidecarEnabled: false,
  };
}

function run(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  const required = ['policyCases', 'decisionMemory', 'recoveredEnrichments', 'artifactDir'];
  const missing = required.filter((key) => !args[key]);
  if (!args.branch && !args.runId) missing.push('branch-or-run-id');
  if (missing.length) throw new Error(`Missing required arguments: ${missing.join(', ')}`);
  let branch = args.branch || deterministicAutomationBranch(args.runId);
  const applySafe = args.applySafe === true;
  const noNetwork = args.noNetwork === true;
  if (!applySafe && (args.commit || args.push || args.createPr)) {
    throw new Error('Git automation requires --apply-safe.');
  }
  if (noNetwork && (args.push || args.createPr)) throw new Error('--no-network blocks push and PR creation.');
  const initialBranch = currentBranch();
  if (!['main', 'master', branch].includes(initialBranch)) {
    throw new Error(`UNEXPECTED_BRANCH:${initialBranch}`);
  }

  const canonicalPath = path.resolve(args.canonical || DEFAULT_CANONICAL);
  const sidecarPath = path.resolve(args.sidecar || DEFAULT_SIDECAR);
  const statePath = path.resolve(args.state || DEFAULT_STATE);
  const artifactDir = path.resolve(args.artifactDir);
  const policyConfig = readJson(path.resolve(args.policyConfig
    || path.join(ROOT, 'config', 'phase4_stage4l_policy.json')));
  const executorConfig = readJson(path.resolve(args.executorConfig
    || path.join(ROOT, 'config', 'phase4_stage4m_auto_pr.json')));
  const cases = enrichReviewCases(readJsonl(path.resolve(args.policyCases)));
  const memory = readJsonl(path.resolve(args.decisionMemory));
  const recovered = readJson(path.resolve(args.recoveredEnrichments));
  const state = readJson(statePath, null);
  const canonicalText = fs.readFileSync(canonicalPath, 'utf8');
  const canonicalSha = sha256Text(canonicalText);
  const expectedBaseline = state?.canonicalShaAfter || executorConfig.initialCanonicalSha256;
  if (canonicalSha !== expectedBaseline) {
    throw new Error(`${BLOCKED_BY_SAFETY_GATE}:UNEXPECTED_CANONICAL_SHA:${canonicalSha}`);
  }
  policyConfig.anomalyThresholds.expectedCanonicalSha256 = expectedBaseline;
  const policy = runAutomatedReviewPolicy(cases, {
    memory,
    config: policyConfig,
    materializeReusableApprovals: true,
    runMetrics: { canonicalSha256: canonicalSha },
  });
  const existingSidecar = readJson(sidecarPath, null);
  const plan = buildSafeEnrichmentPlan({
    policyResult: policy,
    decisionMemory: memory,
    canonicalText,
    existingSidecar,
    executorConfig,
    recoveredEnrichments: recovered.records || [],
    previousExceptionCaseIds: state?.exceptionCaseIds || [],
    budgetOverrides: {
      maxAutoCases: args.maxAutoCases === undefined ? undefined : Number(args.maxAutoCases),
      maxFieldMutations: args.maxFieldMutations === undefined ? undefined : Number(args.maxFieldMutations),
    },
  });
  fs.mkdirSync(artifactDir, { recursive: true });
  writeJson(path.join(artifactDir, 'pre_apply_plan.json'), plan);
  writeJson(path.join(artifactDir, 'delta_exceptions.json'), plan.exceptionRecords);
  if (plan.status === BLOCKED_BY_SAFETY_GATE) {
    throw new Error(`${BLOCKED_BY_SAFETY_GATE}:${plan.violations.join('|')}`);
  }
  if (applySafe && plan.status !== NO_SAFE_CHANGES) {
    branch = ensureFeatureBranch(branch, { createBranch: args.createBranch === true });
  }

  const origin = git(['remote', 'get-url', 'origin']).stdout.trim();
  const compareUrl = githubCompareUrl(origin, branch);
  let result = null;
  let validation = null;
  let testStatus = 'NOT_RUN_DRY_RUN';
  let rollbackStatus = 'NOT_RUN_DRY_RUN';
  let sidecarHash = existingSidecar ? stableHash(normalizeSidecar(existingSidecar).records) : null;
  let canonicalAfterSha = canonicalSha;

  if (applySafe && plan.status !== NO_SAFE_CHANGES) {
    result = applySafeEnrichment({ plan, canonicalText, existingSidecar, executorConfig });
    validation = validateAppliedResult({
      plan, beforeCanonicalText: canonicalText, result, existingSidecar, executorConfig,
    });
    rollbackStatus = validation.rollbackCanonicalByteIdentical
      && validation.rollbackSidecarExact ? 'PASS' : 'FAIL';
    fs.writeFileSync(path.join(artifactDir, 'canonical.before.csv'), canonicalText, 'utf8');
    if (existingSidecar) {
      fs.writeFileSync(path.join(artifactDir, 'sidecar.before.json'), serializeSidecar(existingSidecar), 'utf8');
    }
    fs.mkdirSync(path.dirname(sidecarPath), { recursive: true });
    fs.writeFileSync(canonicalPath, result.canonicalText, 'utf8');
    fs.writeFileSync(sidecarPath, result.sidecarText, 'utf8');
    canonicalAfterSha = result.canonicalSha256;
    sidecarHash = result.sidecarHash;
    const nextState = {
      schemaVersion: 'stage4m-automation-state-v1',
      executorVersion: executorConfig.executorVersion,
      policyVersion: DEFAULT_VERSIONS.policy,
      boundaryVersion: DEFAULT_VERSIONS.boundary,
      canonicalShaBefore: canonicalSha,
      canonicalShaAfter: canonicalAfterSha,
      sidecarHash,
      planHash: plan.planHash,
      exceptionCaseIds: policy.results
        .filter((item) => item.outcome === 'HUMAN_REVIEW').map((item) => item.caseId).sort(),
      appliedOperationKeys: [
        ...plan.canonicalOperations,
        ...plan.sidecarOperations,
      ].map((item) => `${item.caseId}:${item.destination}:${item.field}`).sort(),
    };
    writeJson(statePath, nextState);
    writeJson(path.join(artifactDir, 'post_apply_validation.json'), validation);
    writeJson(path.join(artifactDir, 'rollback_verification.json'), {
      canonicalExpected: canonicalSha,
      canonicalRolledBack: validation.rollbackCanonicalSha256,
      canonicalByteIdentical: validation.rollbackCanonicalByteIdentical,
      sidecarExact: validation.rollbackSidecarExact,
    });

    if (args.runTests) {
      const tests = focusedTests();
      fs.writeFileSync(path.join(artifactDir, 'focused_tests.log'), `${tests.stdout}${tests.stderr}`, 'utf8');
      if (!tests.pass) {
        fs.writeFileSync(canonicalPath, result.rollback.canonicalText, 'utf8');
        if (result.rollback.sidecarText === null) fs.rmSync(sidecarPath, { force: true });
        else fs.writeFileSync(sidecarPath, result.rollback.sidecarText, 'utf8');
        fs.rmSync(statePath, { force: true });
        throw new Error('FOCUSED_TESTS_FAILED_CHANGES_ROLLED_BACK');
      }
      testStatus = 'PASS';
    }
  } else if (applySafe && plan.status === NO_SAFE_CHANGES) {
    testStatus = 'NOT_RERUN_NO_SAFE_CHANGES';
    rollbackStatus = 'NO_MUTATION';
  }

  const manifest = buildManifest({
    runId: args.runId || path.basename(artifactDir),
    branch,
    canonicalBaselineSha: canonicalSha,
    canonicalAfterSha,
    policy,
    memory,
    plan,
    executorConfig,
    sidecarHash,
    testStatus,
    rollbackStatus,
    compareUrl,
  });
  writeJson(path.join(artifactDir, 'run_manifest.json'), manifest);
  fs.writeFileSync(path.join(artifactDir, 'pr_summary.md'), `${buildPrSummary(manifest)}\n`, 'utf8');

  if (applySafe && args.commit && plan.status !== NO_SAFE_CHANGES) {
    git(['add', '--', path.relative(ROOT, canonicalPath), path.relative(ROOT, sidecarPath), path.relative(ROOT, statePath)]);
    git(['commit', '-m', `Automated POI safe enrichment: ${manifest.runId}`]);
  }
  if (args.push && plan.status !== NO_SAFE_CHANGES) git(['push', '-u', 'origin', branch]);
  let prCreation = args.createPr ? 'NO_SAFE_CHANGES' : 'NOT_REQUESTED';
  let prUrl = compareUrl;
  if (args.createPr && plan.status !== NO_SAFE_CHANGES) {
    const pr = createOrFindPullRequest({
      branch,
      title: `Automated POI safe enrichment: ${manifest.runId}`,
      body: buildPrSummary(manifest),
      compareUrl,
    });
    prCreation = pr.status;
    prUrl = pr.url;
    manifest.prUrl = prUrl;
    writeJson(path.join(artifactDir, 'run_manifest.json'), manifest);
  }

  return {
    status: plan.status === NO_SAFE_CHANGES ? NO_SAFE_CHANGES
      : applySafe ? 'SAFE_CHANGES_APPLIED_TO_FEATURE_BRANCH' : 'DRY_RUN_SAFE_PLAN_READY',
    plan: {
      autoAcceptCases: plan.autoAcceptCases,
      canonicalAddressOperations: plan.canonicalAddressOperations,
      sidecarOperations: plan.sidecarOperationCount,
      existingValueSkips: plan.existingValueSkips,
      recoveredStage4kEnrichments: plan.recoveredStage4kEnrichments,
      exceptions: plan.exceptions,
      planHash: plan.planHash,
    },
    canonicalBeforeSha: canonicalSha,
    canonicalAfterSha,
    sidecarHash,
    validation,
    testStatus,
    rollbackStatus,
    branch,
    compareUrl: prUrl,
    networkUsed: !noNetwork && Boolean(args.push || args.createPr),
    prCreation,
  };
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
  assertFeatureBranch,
  buildManifest,
  buildPrSummary,
  createOrFindPullRequest,
  deterministicAutomationBranch,
  ensureFeatureBranch,
  focusedTests,
  githubCompareUrl,
  parseArgs,
  run,
};

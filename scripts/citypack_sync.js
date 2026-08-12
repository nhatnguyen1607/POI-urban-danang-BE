const fs = require('node:fs');
const path = require('node:path');

const { readCanonicalPois } = require('../src/modules/cityPackPreparation/canonicalDataset');
const { normalizeWithAdapters } = require('../src/modules/cityPackPreparation/dryRun');
const { classifySourceRecordHardened } = require('../src/modules/cityPackPreparation/entityResolution');
const { runAutomatedReviewPolicy } = require('../src/modules/cityPackPreparation/automatedReviewPolicy');
const {
  DEFAULT_SOURCE_CAPABILITIES,
  SOURCE_CAPABILITIES,
  readJsonl,
  runIncrementalSync,
  writeJsonl,
} = require('../src/modules/cityPackPreparation/incrementalSync');

const ROOT = path.resolve(__dirname, '..');
const DEFAULT_CANONICAL = path.join(ROOT, 'data', 'canonical', 'urbanagent_poi_master_v1.csv');

function parseArgs(argv) {
  const args = { command: argv[0] || null };
  for (let index = 1; index < argv.length; index += 1) {
    const value = argv[index];
    if (!value.startsWith('--')) throw new Error(`Unexpected argument: ${value}`);
    const key = value.slice(2).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase());
    const next = argv[index + 1];
    if (!next || next.startsWith('--')) args[key] = true;
    else {
      args[key] = next;
      index += 1;
    }
  }
  return args;
}

function readJson(filePath, fallback = null) {
  return filePath && fs.existsSync(filePath)
    ? JSON.parse(fs.readFileSync(filePath, 'utf8'))
    : fallback;
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

function validateCapability(capability) {
  if (!Object.values(SOURCE_CAPABILITIES).includes(capability)) {
    throw new Error(`Unsupported source capability: ${capability}`);
  }
  return capability;
}

function runCli(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (args.command !== 'sync') {
    throw new Error('Usage: npm run citypack:sync -- sync [--policy auto | --snapshot <json> --snapshot-id <id>]');
  }
  if (args.policy === 'auto') {
    if (!args.policyCases || !args.policyConfig) {
      throw new Error('--policy auto requires --policy-cases and --policy-config.');
    }
    const result = runAutomatedReviewPolicy(
      readJson(path.resolve(args.policyCases), []),
      {
        config: readJson(path.resolve(args.policyConfig)),
        memory: readJsonl(args.policyMemory ? path.resolve(args.policyMemory) : null),
      },
    );
    if (args.summary) writeJson(path.resolve(args.summary), result);
    return result;
  }
  if (!args.snapshot || !args.snapshotId) throw new Error('--snapshot and --snapshot-id are required.');

  const input = readJson(path.resolve(args.snapshot));
  const rawRecords = input?.records || [];
  const normalizedRecords = normalizeWithAdapters(rawRecords);
  const canonicalPois = readCanonicalPois(path.resolve(args.canonical || DEFAULT_CANONICAL));
  const statePath = args.state ? path.resolve(args.state) : null;
  const mediaPath = args.mediaRegistry ? path.resolve(args.mediaRegistry) : null;
  const checkpointPath = args.checkpoint ? path.resolve(args.checkpoint) : null;
  const checkpoint = args.resume ? readJson(checkpointPath) : null;
  const capabilities = Object.fromEntries(
    Object.entries(DEFAULT_SOURCE_CAPABILITIES).map(([source, capability]) => [
      source,
      validateCapability(input?.sourceCapabilities?.[source] || capability),
    ]),
  );

  const result = runIncrementalSync({
    snapshotId: args.snapshotId,
    records: normalizedRecords,
    previousState: readJsonl(statePath),
    previousMediaRegistry: readJsonl(mediaPath),
    checkpoint,
    resolveRecord: (record) => classifySourceRecordHardened(record, canonicalPois, undefined, {
      hardened: true,
      maxCandidateDistanceMeters: 800,
      resolutionVersion: 'stage4h-v1',
    }),
    onCheckpoint: checkpointPath ? (value) => writeJson(checkpointPath, value) : null,
  });

  if (args.writeState) {
    if (!statePath || !mediaPath) {
      throw new Error('--write-state requires --state and --media-registry.');
    }
    if (!result.complete) throw new Error('Cannot write final state from an incomplete sync.');
    writeJsonl(statePath, result.state);
    writeJsonl(mediaPath, result.mediaRegistry);
    if (checkpointPath && fs.existsSync(checkpointPath)) fs.rmSync(checkpointPath);
  }

  const summary = {
    status: 'NON_RUNTIME_INCREMENTAL_CITYPACK_SYNC',
    snapshotId: args.snapshotId,
    complete: result.complete,
    sourceCapabilities: capabilities,
    metrics: result.metrics,
    stateWritten: Boolean(args.writeState),
    canonicalWriteAuthorized: false,
    runtimeChanged: false,
  };
  if (args.summary) writeJson(path.resolve(args.summary), summary);
  return summary;
}

if (require.main === module) {
  try {
    console.log(JSON.stringify(runCli(), null, 2));
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}

module.exports = { parseArgs, runCli, validateCapability };

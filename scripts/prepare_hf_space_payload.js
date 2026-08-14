const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const SOURCE_ROOT = path.resolve(__dirname, '..');
const DEFAULT_MANIFEST_PATH = path.join(
  SOURCE_ROOT,
  'deploy',
  'huggingface',
  'runtime-files.txt',
);

function sha256(filePath) {
  return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

function normalizeEntry(value) {
  const entry = String(value || '').trim().replaceAll('\\', '/').replace(/\/+$/, '');
  if (!entry || entry.startsWith('/') || entry.includes('..') || path.isAbsolute(entry)) {
    throw new Error(`Unsafe runtime manifest entry: ${value}`);
  }
  return entry;
}

function readRuntimeEntries(manifestPath = DEFAULT_MANIFEST_PATH) {
  return fs.readFileSync(manifestPath, 'utf8')
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('#'))
    .map(normalizeEntry);
}

function gitOutput(args) {
  const result = spawnSync('git', args, {
    cwd: SOURCE_ROOT,
    encoding: 'utf8',
    windowsHide: true,
  });
  if (result.status !== 0) {
    throw new Error(result.stderr.trim() || `git ${args.join(' ')} failed`);
  }
  return result.stdout;
}

function listTrackedRuntimeFiles(entries) {
  const tracked = gitOutput(['ls-files', '-z', '--', ...entries])
    .split('\0')
    .filter(Boolean)
    .sort((left, right) => left.localeCompare(right));

  for (const entry of entries) {
    const matched = tracked.some((file) => file === entry || file.startsWith(`${entry}/`));
    if (!matched) throw new Error(`Runtime manifest entry has no tracked files: ${entry}`);
  }
  return tracked;
}

function isGitLfsPointer(filePath) {
  const descriptor = fs.openSync(filePath, 'r');
  try {
    const buffer = Buffer.alloc(200);
    const bytesRead = fs.readSync(descriptor, buffer, 0, buffer.length, 0);
    return buffer.subarray(0, bytesRead).toString('utf8')
      .startsWith('version https://git-lfs.github.com/spec/v1');
  } finally {
    fs.closeSync(descriptor);
  }
}

function copyTrackedFile(relativePath, outputPath) {
  const sourcePath = path.join(SOURCE_ROOT, relativePath);
  const stat = fs.lstatSync(sourcePath);
  if (!stat.isFile() || stat.isSymbolicLink()) {
    throw new Error(`Runtime payload accepts regular tracked files only: ${relativePath}`);
  }
  if (isGitLfsPointer(sourcePath)) {
    throw new Error(`Runtime payload cannot contain a Git LFS pointer: ${relativePath}`);
  }

  const targetPath = path.join(outputPath, relativePath);
  fs.mkdirSync(path.dirname(targetPath), { recursive: true });
  fs.copyFileSync(sourcePath, targetPath);
  return {
    path: relativePath.replaceAll('\\', '/'),
    bytes: stat.size,
    sha256: sha256(sourcePath),
  };
}

function prepareRuntimePayload({ outputPath, manifestPath = DEFAULT_MANIFEST_PATH }) {
  if (!outputPath) throw new Error('An explicit --output directory is required.');
  const resolvedOutput = path.resolve(outputPath);
  if (resolvedOutput === SOURCE_ROOT || fs.existsSync(resolvedOutput)) {
    throw new Error(`Output directory must be new and must not be the source root: ${resolvedOutput}`);
  }

  const entries = readRuntimeEntries(manifestPath);
  const trackedFiles = listTrackedRuntimeFiles(entries);
  fs.mkdirSync(resolvedOutput, { recursive: false });

  const files = trackedFiles.map((file) => copyTrackedFile(file, resolvedOutput));
  const sourceCommit = gitOutput(['rev-parse', 'HEAD']).trim();
  const deploymentManifest = {
    schemaVersion: 1,
    sourceRepository: 'nhatnguyen1607/POI-urban-danang-BE',
    sourceCommit,
    targetSpace: 'nhttngy/back-end',
    files,
  };
  fs.writeFileSync(
    path.join(resolvedOutput, 'runtime_deployment_manifest.json'),
    `${JSON.stringify(deploymentManifest, null, 2)}\n`,
    'utf8',
  );

  return { outputPath: resolvedOutput, ...deploymentManifest };
}

function parseOutputArgument(argv) {
  const index = argv.indexOf('--output');
  return index >= 0 ? argv[index + 1] : null;
}

if (require.main === module) {
  try {
    const result = prepareRuntimePayload({ outputPath: parseOutputArgument(process.argv.slice(2)) });
    console.log(JSON.stringify({
      status: 'PASS',
      outputPath: result.outputPath,
      sourceCommit: result.sourceCommit,
      fileCount: result.files.length,
    }, null, 2));
  } catch (error) {
    console.error(`HF_PAYLOAD_PREPARATION_FAILED: ${error.message}`);
    process.exitCode = 1;
  }
}

module.exports = {
  DEFAULT_MANIFEST_PATH,
  SOURCE_ROOT,
  listTrackedRuntimeFiles,
  prepareRuntimePayload,
  readRuntimeEntries,
};

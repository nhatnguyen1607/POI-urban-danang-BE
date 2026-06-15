const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.resolve(__dirname, '..', '..');
const SUPPORTED_TEXT_VERSIONS = new Set(['v2', 'v4']);

function resolvePythonExecutable() {
  const candidates = [
    process.env.PYTHON_EXECUTABLE,
    path.join(ROOT_DIR, 'venv_new', 'Scripts', 'python.exe'),
    path.resolve(ROOT_DIR, '../poi_urban/venv/Scripts/python.exe'),
    path.resolve(ROOT_DIR, '../poi-urban-danang/.venv/Scripts/python.exe'),
  ].filter(Boolean);
  return candidates.find((candidate) => fs.existsSync(candidate)) || 'python';
}

function extractJson(stdout) {
  const match = String(stdout || '').match(/\[[\s\S]*\]/);
  if (!match) throw new Error('Semantic model did not return a JSON result list.');
  return JSON.parse(match[0]);
}

function runSemanticRetrieval({
  query,
  version = 'v4',
  topK = 200,
  candidateLimit = 200,
  timeoutMs = 45000,
  candidates = [],
} = {}) {
  const normalizedVersion = String(version || 'v4').toLowerCase();
  if (!SUPPORTED_TEXT_VERSIONS.has(normalizedVersion)) {
    return Promise.resolve({
      enabled: false,
      available: false,
      version: normalizedVersion,
      reason: `${normalizedVersion} does not provide a text encoder for semantic agent retrieval.`,
      scores: new Map(),
    });
  }
  if (!String(query || '').trim()) {
    return Promise.resolve({
      enabled: false,
      available: false,
      version: normalizedVersion,
      reason: 'Semantic retrieval requires a non-empty text query.',
      scores: new Map(),
    });
  }

  const scriptPath = path.join(ROOT_DIR, 'src', 'inference.py');
  return new Promise((resolve) => {
    const hasCandidates = Array.isArray(candidates) && candidates.length > 0;
    const child = spawn(resolvePythonExecutable(), [
      scriptPath,
      '--version', normalizedVersion,
      '--text', String(query),
      '--top-k', String(Math.max(1, Math.min(Number(topK) || 80, 200))),
      '--candidate-limit', String(Math.max(1, Math.min(Number(candidateLimit) || 200, 2000))),
      '--strict-errors',
      ...(hasCandidates ? ['--stdin-candidates'] : []),
    ], {
      env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
      windowsHide: true,
    });

    let stdout = '';
    let stderr = '';
    let settled = false;
    const finish = (result) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(result);
    };
    const timer = setTimeout(() => {
      child.kill();
      finish({
        enabled: true,
        available: false,
        version: normalizedVersion,
        reason: `Semantic model timed out after ${timeoutMs} ms.`,
        scores: new Map(),
      });
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    if (hasCandidates) {
      child.stdin.end(JSON.stringify(candidates.slice(0, Math.max(1, candidateLimit))));
    }
    child.on('error', (error) => {
      finish({
        enabled: true,
        available: false,
        version: normalizedVersion,
        reason: error.message,
        scores: new Map(),
      });
    });
    child.on('close', (code) => {
      if (settled) return;
      if (code !== 0) {
        finish({
          enabled: true,
          available: false,
          version: normalizedVersion,
          reason: stderr.trim().split(/\r?\n/).slice(-1)[0] || `Semantic model exited with code ${code}.`,
          scores: new Map(),
        });
        return;
      }
      try {
        const results = extractJson(stdout);
        const scores = new Map(
          results.map((item) => [
            String(item.id),
            Math.max(0, Math.min(1, Number(item.score) / 100)),
          ]),
        );
        finish({
          enabled: true,
          available: true,
          version: normalizedVersion,
          reason: null,
          resultCount: scores.size,
          scores,
        });
      } catch (error) {
        finish({
          enabled: true,
          available: false,
          version: normalizedVersion,
          reason: error.message,
          scores: new Map(),
        });
      }
    });
  });
}

module.exports = {
  SUPPORTED_TEXT_VERSIONS,
  resolvePythonExecutable,
  runSemanticRetrieval,
};

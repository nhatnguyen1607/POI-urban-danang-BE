const fs = require('fs');
const path = require('path');

const ROOT_DIR = path.resolve(__dirname, '..', '..');
const FEEDBACK_DIR = path.join(ROOT_DIR, 'storage', 'feedback');
const FEEDBACK_FILE = path.join(FEEDBACK_DIR, 'agent-feedback.jsonl');

function sanitizeFeedback(input = {}) {
  return {
    timestamp: new Date().toISOString(),
    role: String(input.role || 'traveler').slice(0, 40),
    eventType: String(input.eventType || 'unknown').slice(0, 80),
    query: String(input.query || '').slice(0, 1200),
    payload: input.payload || {},
  };
}

async function recordFeedback(input) {
  fs.mkdirSync(FEEDBACK_DIR, { recursive: true });
  const event = sanitizeFeedback(input);
  await fs.promises.appendFile(FEEDBACK_FILE, `${JSON.stringify(event)}\n`, 'utf8');
  return {
    saved: true,
    event,
    learningUse:
      'This event can be used later for reranking, preference memory, and supervised fine-tuning.',
  };
}

module.exports = {
  recordFeedback,
  FEEDBACK_FILE,
};

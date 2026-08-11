const BUSINESS_DESCRIPTORS = new Set([
  'bakery', 'bar', 'cafe', 'coffee', 'company', 'hotel', 'market',
  'restaurant', 'store', 'shop',
]);

const BUSINESS_DESCRIPTOR_PHRASES = [
  ['cua', 'hang'],
  ['nha', 'hang'],
  ['quan', 'an'],
  ['tiem', 'banh'],
];

const BRANCH_MARKERS = [
  ['chi', 'nhanh'],
  ['co', 'so'],
  ['branch'],
  ['store'],
];

const SPAM_PATTERNS = [
  /https?:\/\/\S+/gi,
  /\bwww\.\S+/gi,
  /\b(?:hotline|lien he|dat ban|dat phong)\s*[:\-]?\s*[+\d][\d\s.()-]{6,}/gi,
  /\b(?:khuyen mai|giam gia|sale)\s*[:\-]?\s*\d{1,3}\s*%/gi,
];

function removeObviousAdvertising(value) {
  return SPAM_PATTERNS.reduce(
    (text, pattern) => text.replace(pattern, ' '),
    String(value || ''),
  );
}

function normalizeVietnameseText(value) {
  const folded = String(value || '')
    .normalize('NFKC')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[\u0111\u0110]/g, 'd')
    .toLowerCase();
  return removeObviousAdvertising(folded)
    .replace(/&/g, ' and ')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function findMarker(tokens) {
  for (const marker of BRANCH_MARKERS) {
    for (let index = 0; index <= tokens.length - marker.length; index += 1) {
      if (marker.every((token, offset) => tokens[index + offset] === token)) {
        return { index, length: marker.length };
      }
    }
  }
  return null;
}

function removeEdgeDescriptors(tokens) {
  let start = 0;
  let end = tokens.length;
  let changed = true;
  while (changed && start < end) {
    changed = false;
    if (BUSINESS_DESCRIPTORS.has(tokens[start])) {
      start += 1;
      changed = true;
    }
    for (const phrase of BUSINESS_DESCRIPTOR_PHRASES) {
      if (phrase.every((token, offset) => tokens[start + offset] === token)) {
        start += phrase.length;
        changed = true;
      }
    }
  }
  changed = true;
  while (changed && end > start) {
    changed = false;
    if (BUSINESS_DESCRIPTORS.has(tokens[end - 1])) {
      end -= 1;
      changed = true;
    }
    for (const phrase of BUSINESS_DESCRIPTOR_PHRASES) {
      const phraseStart = end - phrase.length;
      if (phraseStart >= start && phrase.every((token, offset) => tokens[phraseStart + offset] === token)) {
        end -= phrase.length;
        changed = true;
      }
    }
  }
  return tokens.slice(start, end);
}

function findBusinessDescriptor(tokens) {
  for (let index = 0; index < tokens.length; index += 1) {
    if (BUSINESS_DESCRIPTORS.has(tokens[index])) return { index, length: 1 };
    for (const phrase of BUSINESS_DESCRIPTOR_PHRASES) {
      if (phrase.every((token, offset) => tokens[index + offset] === token)) {
        return { index, length: phrase.length };
      }
    }
  }
  return null;
}

function buildIdentityProfile(value) {
  const normalized = normalizeVietnameseText(value);
  const tokens = normalized.split(' ').filter(Boolean);
  const explicitMarker = findMarker(tokens);
  let baseTokens = tokens;
  let branchTokens = [];

  if (explicitMarker) {
    baseTokens = tokens.slice(0, explicitMarker.index);
    branchTokens = tokens.slice(explicitMarker.index + explicitMarker.length);
  } else {
    const descriptor = findBusinessDescriptor(tokens);
    if (descriptor && descriptor.index > 0 && descriptor.index + descriptor.length < tokens.length) {
      baseTokens = tokens.slice(0, descriptor.index);
      branchTokens = tokens.slice(descriptor.index + descriptor.length);
    }
  }

  baseTokens = removeEdgeDescriptors(baseTokens);
  if (baseTokens.length === 0) baseTokens = removeEdgeDescriptors(tokens);
  if (baseTokens.length === 0) baseTokens = tokens;

  return {
    original: String(value || ''),
    normalized,
    baseName: baseTokens.join(' '),
    branchName: branchTokens.join(' '),
    tokens,
    baseTokens,
    branchTokens,
  };
}

function tokenSet(value) {
  return new Set(normalizeVietnameseText(value).split(' ').filter(Boolean));
}

function tokenOverlap(leftValue, rightValue) {
  const left = tokenSet(leftValue);
  const right = tokenSet(rightValue);
  if (left.size === 0 || right.size === 0) return 0;
  const intersection = [...left].filter((token) => right.has(token)).length;
  return intersection / new Set([...left, ...right]).size;
}

function hasBranchConflict(leftValue, rightValue) {
  const left = buildIdentityProfile(leftValue);
  const right = buildIdentityProfile(rightValue);
  if (!left.baseName || left.baseName !== right.baseName) return false;
  if (!left.branchName || !right.branchName) return false;
  return tokenOverlap(left.branchName, right.branchName) < 0.5;
}

module.exports = {
  buildIdentityProfile,
  hasBranchConflict,
  normalizeVietnameseText,
  removeObviousAdvertising,
  tokenOverlap,
};

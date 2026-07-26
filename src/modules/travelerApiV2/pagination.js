const { DEFAULT_LIMIT, MAX_LIMIT } = require('./constants');

function decodeCursor(cursor) {
  if (!cursor) return 0;
  try {
    const json = Buffer.from(String(cursor), 'base64url').toString('utf8');
    const parsed = JSON.parse(json);
    const offset = Number(parsed.offset);
    if (!Number.isInteger(offset) || offset < 0) return null;
    return offset;
  } catch (_) {
    return null;
  }
}

function encodeCursor(offset) {
  return Buffer.from(JSON.stringify({ offset }), 'utf8').toString('base64url');
}

function parseLimit(value) {
  if (value === undefined || value === null || value === '') {
    return { limit: DEFAULT_LIMIT };
  }
  const limit = Number(value);
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_LIMIT) {
    return {
      error: {
        field: 'limit',
        rule: `integer_between_1_and_${MAX_LIMIT}`,
      },
    };
  }
  return { limit };
}

function parsePagination(query = {}) {
  const limitResult = parseLimit(query.limit);
  if (limitResult.error) return limitResult;

  const offset = decodeCursor(query.cursor);
  if (offset === null) {
    return {
      error: {
        field: 'cursor',
        rule: 'opaque_cursor',
      },
    };
  }

  return {
    limit: limitResult.limit,
    offset,
  };
}

function pageItems(items, { limit, offset }) {
  const page = items.slice(offset, offset + limit);
  const nextOffset = offset + page.length;
  return {
    items: page,
    page: {
      total: items.length,
      limit,
      nextCursor: nextOffset < items.length ? encodeCursor(nextOffset) : null,
    },
  };
}

module.exports = {
  decodeCursor,
  encodeCursor,
  pageItems,
  parseLimit,
  parsePagination,
};

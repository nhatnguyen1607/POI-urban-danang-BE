const canonical = require('./canonicalCsvPoiRepository');

class LegacyCsvPoiRepository extends canonical.CanonicalCsvPoiRepository {}

module.exports = {
  ...canonical,
  LegacyCsvPoiRepository,
  normalizeLegacyPoiRow: canonical.normalizeCanonicalPoiRow,
  SOURCE_FILES: [
    {
      declaredSource: 'canonical',
      file: canonical.CANONICAL_POI_CSV_PATH,
      fallbackFile: null,
    },
  ],
};

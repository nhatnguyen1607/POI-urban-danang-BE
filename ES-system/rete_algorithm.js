const fs = require('fs');
const csv = require('csv-parser');
const path = require('path');

// ============================================================================
//  RETE ALGORITHM - Nhanh chóng khớp mẫu thông qua Mạng lưới bộ nhớ
// ============================================================================

class ReteNetwork {
  constructor() {
    // Mạng lưới node Rete
    this.alphaMemory = {
      turnRestrictions: new Map(), // fromRoad -> Map(toRoad -> rule)
      timeRestrictions: new Map(), // roadName -> rule[]
      oneWayStreets: new Map(),    // roadName -> rule[]
    };
    this.loaded = false;
  }

  async load() {
    const ruleDir = path.join(__dirname, 'rule');

    const [turnRules, timeRules, oneWayRules] = await Promise.all([
      this._readCSV(path.join(ruleDir, 'cam_queo.csv')),
      this._readCSV(path.join(ruleDir, 'cam_theo_gio.csv')),
      this._readCSV(path.join(ruleDir, 'duong_1_chieu.csv')),
    ]);

    this._buildTurnNetwork(turnRules);
    this._buildTimeNetwork(timeRules);
    this._buildOneWayNetwork(oneWayRules);

    this.loaded = true;
    console.log(`[ES] Rete Network compiled! Rules indexed for fast matching.`);
  }

  _buildTurnNetwork(turnRules) {
    turnRules.filter(row => row['Loại cấm'] && row['Loại cấm'] !== 'N/A').forEach(row => {
      const fromRoad = this._normalizeRoadName(row['Đang đi trên đường']);
      const toRoad = this._normalizeRoadName(row['Cấm rẽ vào đường']);
      
      if (!this.alphaMemory.turnRestrictions.has(fromRoad)) {
        this.alphaMemory.turnRestrictions.set(fromRoad, new Map());
      }
      this.alphaMemory.turnRestrictions.get(fromRoad).set(toRoad, {
        type: this._normalizeTurnType(row['Loại cấm']),
        rawType: row['Loại cấm'],
      });
    });
  }

  _buildTimeNetwork(timeRules) {
    timeRules.forEach(row => {
      const roadName = this._normalizeRoadName(row['Tên đường']);
      if (!this.alphaMemory.timeRestrictions.has(roadName)) {
        this.alphaMemory.timeRestrictions.set(roadName, []);
      }
      this.alphaMemory.timeRestrictions.get(roadName).push({
        timeRanges: this._parseTimeRanges(row['Khung giờ']),
        restrictionType: row['Loại cấm'],
        rawTimeStr: row['Khung giờ'],
      });
    });
  }

  _buildOneWayNetwork(oneWayRules) {
    oneWayRules.forEach(row => {
      const roadName = this._normalizeRoadName(row['Tên đường']);
      if (!this.alphaMemory.oneWayStreets.has(roadName)) {
        this.alphaMemory.oneWayStreets.set(roadName, []);
      }
      const startCoord = this._parseCoord(row['Tọa độ bắt đầu']);
      const endCoord = this._parseCoord(row['Tọa độ kết thúc']);
      this.alphaMemory.oneWayStreets.get(roadName).push({
        startLat: startCoord.lat,
        startLng: startCoord.lng,
        endLat: endCoord.lat,
        endLng: endCoord.lng,
      });
    });
  }

  // --- Matching Methods (Truy vấn mạng Rete O(1) hoặc O(K)) ---

  matchTurnRestriction(fromRoad, toRoad) {
    const fromKeys = this._findFuzzyKeys(fromRoad, this.alphaMemory.turnRestrictions.keys());
    for (const fKey of fromKeys) {
      const toMap = this.alphaMemory.turnRestrictions.get(fKey);
      const toKeys = this._findFuzzyKeys(toRoad, toMap.keys());
      for (const tKey of toKeys) {
        return toMap.get(tKey); // Return first matching rule
      }
    }
    return null;
  }

  matchTimeRestriction(roadName) {
    const keys = this._findFuzzyKeys(roadName, this.alphaMemory.timeRestrictions.keys());
    const results = [];
    for (const k of keys) {
      results.push(...this.alphaMemory.timeRestrictions.get(k));
    }
    return results;
  }

  matchOneWayStreet(roadName) {
    const keys = this._findFuzzyKeys(roadName, this.alphaMemory.oneWayStreets.keys());
    const results = [];
    for (const k of keys) {
      results.push(...this.alphaMemory.oneWayStreets.get(k));
    }
    return results;
  }

  // --- Helper Methods ---

  _findFuzzyKeys(targetName, keysIterable) {
    const target = this._normalizeRoadName(targetName);
    const matched = [];
    for (const key of keysIterable) {
      if (this._fuzzyMatch(target, key)) {
        matched.push(key);
      }
    }
    return matched;
  }

  _fuzzyMatch(name1, name2) {
    if (!name1 || !name2) return false;
    if (name1 === name2) return true;
    const strip = (s) => s.replace(/^(đường|cầu|hầm chui|đại lộ|ga đi|ga đến)\s+/i, '').trim();
    const s1 = strip(name1);
    const s2 = strip(name2);
    if (s1 === s2) return true;
    if (s1.includes(s2) || s2.includes(s1)) return true;
    return false;
  }

  _readCSV(filePath) {
    return new Promise((resolve, reject) => {
      const results = [];
      fs.createReadStream(filePath, { encoding: 'utf-8' })
        .pipe(csv())
        .on('data', (data) => results.push(data))
        .on('end', () => resolve(results))
        .on('error', (err) => reject(err));
    });
  }

  _normalizeRoadName(name) {
    if (!name) return '';
    return name.trim()
      .replace(/^Đường\s+/i, '')
      .replace(/^Cầu\s+/i, 'Cầu ')
      .replace(/^Hầm chui\s+/i, 'Hầm chui ')
      .toLowerCase()
      .normalize('NFC');
  }

  _normalizeTurnType(type) {
    if (!type) return 'unknown';
    const t = type.toLowerCase().trim();
    if (t.includes('quẹo phải') || t.includes('rẽ phải')) return 'no_right_turn';
    if (t.includes('quẹo trái') || t.includes('rẽ trái')) return 'no_left_turn';
    if (t.includes('quay đầu')) return 'no_u_turn';
    if (t.includes('đi thẳng')) return 'no_straight';
    if (t.includes('chỉ được quẹo phải') || t.includes('chỉ được rẽ phải')) return 'only_right_turn';
    return 'unknown';
  }

  _parseTimeRanges(timeStr) {
    if (!timeStr) return [];
    const ranges = [];
    const parts = timeStr.replace(/AND/gi, ',').split(',');
    for (const part of parts) {
      const match = part.trim().match(/(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})/);
      if (match) {
        ranges.push({
          startHour: parseInt(match[1]),
          startMin: parseInt(match[2]),
          endHour: parseInt(match[3]),
          endMin: parseInt(match[4]),
        });
      }
    }
    return ranges;
  }

  _parseCoord(coordStr) {
    if (!coordStr) return { lat: 0, lng: 0 };
    const clean = coordStr.replace(/"/g, '').trim();
    const parts = clean.split(',');
    return {
      lat: parseFloat(parts[0]) || 0,
      lng: parseFloat(parts[1]) || 0,
    };
  }
}

module.exports = ReteNetwork;

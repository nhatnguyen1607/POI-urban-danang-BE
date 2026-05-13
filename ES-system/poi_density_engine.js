const fs = require('fs');
const csv = require('csv-parser');
const path = require('path');

// ============================================================================
//  POI DENSITY ENGINE - Suy luận mật độ giao thông từ dữ liệu POI
//  Thay thế road_profiles.csv tĩnh bằng bằng chứng thực tế (Evidence-based)
// ============================================================================

class POIDensityEngine {
  constructor() {
    this.roadPoiCount = new Map(); // roadName (normalized) -> count
    this.allPois = [];             // [{lat, lng}] cho proximity search
    this.loaded = false;
  }

  async load() {
    const dataDir = path.join(__dirname, '..', 'data');

    const [foodyPois, ggmapPois] = await Promise.all([
      this._readCSV(path.join(dataDir, 'poi_data_foody.csv')),
      this._readCSV(path.join(dataDir, 'poi_data_ggmap.csv')),
    ]);

    // Foody: có trường Address → trích xuất tên đường
    for (const poi of foodyPois) {
      const roadName = this._extractRoadName(poi['Address']);
      if (roadName) {
        const key = this._normalize(roadName);
        this.roadPoiCount.set(key, (this.roadPoiCount.get(key) || 0) + 1);
      }
      const lat = parseFloat(poi['Lat']);
      const lng = parseFloat(poi['Lon']);
      if (!isNaN(lat) && !isNaN(lng)) {
        this.allPois.push({ lat, lng });
      }
    }

    // GGMap: chỉ có tọa độ → dùng proximity search
    for (const poi of ggmapPois) {
      const lat = parseFloat(poi['Lat']);
      const lng = parseFloat(poi['Lon']);
      if (!isNaN(lat) && !isNaN(lng)) {
        this.allPois.push({ lat, lng });
      }
    }

    this.loaded = true;
    console.log(`[POI Engine] ${this.roadPoiCount.size} đường từ Foody | ${this.allPois.length} tổng POI`);

    // Log top 10 đường đông nhất
    const sorted = [...this.roadPoiCount.entries()].sort((a, b) => b[1] - a[1]).slice(0, 10);
    console.log(`[POI Engine] Top 10 đường đông:`, sorted.map(([r, c]) => `${r}(${c})`).join(', '));
  }

  /**
   * Lấy mật độ POI cho một đoạn đường (kết hợp cả 2 phương pháp)
   * @returns {number} Số lượng POI (giá trị liên tục cho fuzzy)
   */
  getDensity(roadName, lat, lng) {
    const nameCount = this._getCountByName(roadName);
    const nearbyCount = (lat && lng) ? this._countNearby(lat, lng, 0.2) : 0;
    return Math.max(nameCount, nearbyCount);
  }

  /** Đếm POI theo tên đường (từ Foody) */
  _getCountByName(roadName) {
    if (!roadName) return 0;
    const target = this._normalize(roadName);

    // Exact match
    if (this.roadPoiCount.has(target)) return this.roadPoiCount.get(target);

    // Fuzzy match
    for (const [key, count] of this.roadPoiCount) {
      if (key.includes(target) || target.includes(key)) return count;
    }
    return 0;
  }

  /** Đếm POI trong bán kính (km) của một tọa độ */
  _countNearby(lat, lng, radiusKm) {
    let count = 0;
    for (const poi of this.allPois) {
      if (this._fastDistance(lat, lng, poi.lat, poi.lng) <= radiusKm) count++;
    }
    return count;
  }

  /** Trích xuất tên đường từ Address foody: "100 Lê Đại Hành, P. Khuê Trung," → "Lê Đại Hành" */
  _extractRoadName(address) {
    if (!address) return null;
    const clean = address.replace(/"/g, '').trim();
    const firstPart = clean.split(',')[0].trim();
    // Bỏ số nhà: "100A", "Số 47", "K12/5"
    const name = firstPart
      .replace(/^(Số\s+)?(\d+[A-Za-z\/]*\s+)?/, '')
      .replace(/^(K\d+[\/\d]*\s+)?/, '')
      .trim();
    return name.length > 2 ? name : null;
  }

  _normalize(name) {
    if (!name) return '';
    return name.trim()
      .replace(/^(Đường|đường|Duong|duong)\s+/i, '')
      .toLowerCase()
      .normalize('NFC');
  }

  /** Haversine nhanh (đủ chính xác cho khoảng cách ngắn) */
  _fastDistance(lat1, lon1, lat2, lon2) {
    const R = 6371;
    const dLat = (lat2 - lat1) * 0.01745329;
    const dLon = (lon2 - lon1) * 0.01745329;
    const a = Math.sin(dLat / 2) ** 2 +
      Math.cos(lat1 * 0.01745329) * Math.cos(lat2 * 0.01745329) *
      Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
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
}

module.exports = POIDensityEngine;

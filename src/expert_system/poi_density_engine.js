const fs = require('fs');
const csv = require('csv-parser');
const path = require('path');
const { loadPOIs } = require('../services/poiDataService');

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
    const pois = await loadPOIs();

    // Canonical POIs keep current/raw address separately; never infer address from district.
    for (const poi of pois) {
      const roadName = this._extractRoadName(poi.addressCurrent || poi.addressRaw || poi.address);
      if (roadName) {
        const key = this._normalize(roadName);
        this.roadPoiCount.set(key, (this.roadPoiCount.get(key) || 0) + 1);
      }
      this.allPois.push({ lat: poi.lat, lng: poi.lon });
    }

    this.loaded = true;
    console.log(`[POI Engine] ${this.roadPoiCount.size} roads from canonical address data | ${this.allPois.length} POIs`);

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

  /** Count POIs by road name when canonical address data is available. */
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

# Data Audit

Updated: 2026-07-26 01:49:31 +07:00.

Canonical runtime dataset:

`data/canonical/urbanagent_poi_master_v1.csv`

Approved dataset decision:

`docs/rebuild/URBANAGENT_DATASET_DECISION.md`

## Canonical CSV Verification

- SHA-256: `5cc6ba843e6c93cb0b5403a03c5557f06a2e5d34a74340b4d0b4d6262035f7ae`
- Total rows: `4166`
- Column count: `34`
- Unique `Global_ID`: `4166`
- Duplicate `Global_ID` extra rows: `0`
- Missing `Global_ID`: `0`
- Missing `City_ID`: `0`
- City distribution: `da-nang: 4166`
- Non-POI entity rows loaded into product runtime: `0`
- Missing coordinates: `0`
- Invalid coordinates: `0`
- Coordinates outside Da Nang bbox: `0`
- Apparent Da Nang-center fallback coordinates: `0`
- Missing address: `4166`
- Missing primary rating: `312`
- Missing review count: `220`
- Alias `Global_ID` values preserved: `985`

## Columns

```text
Global_ID
Alias_Global_IDs
City_ID
Entity_Type
RestaurantID
Source_IDs
Restaurant Name
District
District_Raw
Admin_Normalization_Status
Address_Raw
Address_Current
Category
Category_Normalized
Lat
Lon
Overall Rating
Rating_Scale
Rating_Count
Review_Sample_Count
Google_Rating
Google_Rating_Count
Foody_Rating_10
Foody_Review_Sample_Count
Primary_Rating_Source
Price
Price_Min_VND
Price_Max_VND
Opening_Hours_Raw
Image_URL
LLM_Input_Text
Source
Merge_Status
Data_Quality_Flags
```

## Runtime Loader Results After Fix

- `readCSV()` first header: `Global_ID`
- Header matches expected schema: `true`
- `loadPOIs({ cityId: "da-nang" })`: `4166`
- `loadPOIs({ cityId: "hue" })`: `0`
- `getPoiDataQualityReport().totals.applicationPois`: `4166`
- `getPoiDataQualityReport().totals.invalidRows`: `0`

## Category Distribution

```text
restaurant: 1853
cafe: 1201
bakery: 433
shopping: 370
nightlife: 298
other: 4
park: 2
lodging: 2
amenity: 1
wellness: 1
service: 1
```

## Source And Merge Semantics

Source distribution from the approved decision:

```text
google_maps: 3941
foody: 220
google_maps+foody: 5
```

Merge status from the approved decision:

```text
single_source: 3363
same_source_deduplicated: 798
cross_source_high_confidence_merged: 5
```

No additional duplicate merge was performed in this fix batch.

## Backend Fields Previously Wrong Or Unsafe

Fixed in this batch:

- `csv-parser` BOM on first header caused `Global_ID` to be read as a different key. The loader now strips UTF-8 BOM from header index 0 and trims all headers.
- Runtime repository now loads 4,166 POIs instead of 0.
- `Image_URL` now maps to `imageUrls` by splitting comma-separated values, trimming, filtering invalid/non-http URLs, and deduplicating while preserving order.
- `ES-system/poi_density_engine.js` no longer reads deleted/legacy root `data/poi_data_foody.csv` or `data/poi_data_ggmap.csv`.
- Firestore POI normalization no longer turns unknown rating/review counts into `0`.

Still intentionally scoped:

- `RestaurantID` remains `sourceId`; it is not a verified Google `place_id`.
- `Address_Current` is empty for all 4,166 rows, so code must not fill address from `District`.
- `Foody_Review_Sample_Count` remains separate from `Rating_Count`.

## Values That Must Never Be Auto-Filled

- Missing lat/lon must not become `16.0544,108.2022`.
- Missing address must not become `District`, `Da Nang`, or any inferred label.
- Missing admin unit must not be inferred without boundary validation.
- Missing `place_id` must not be invented from `RestaurantID`.
- Missing rating/review count must remain null/unknown, not `0`.
- Missing opening hours, phone, website, operating status, source freshness, and price must not be fabricated.

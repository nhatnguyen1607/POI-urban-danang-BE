import argparse
import csv
import hashlib
import json
import math
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

from overturemaps.core import record_batch_reader
from shapely import wkb


BBOX = (108.2200, 16.0580, 108.2300, 16.0680)
OVERTURE_RELEASE = "2026-07-22.0"
QUOTAS = {
    "attraction": 6,
    "food": 8,
    "accommodation": 6,
    "public_cultural": 4,
}


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    return sha256_bytes(Path(path).read_bytes())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(value) + "\n", encoding="utf-8", newline="\n")


def fold(value):
    text = unicodedata.normalize("NFD", str(value or ""))
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", text.lower().replace("đ", "d")).strip()


def normalized_entity_name(value):
    text = fold(value).replace(" and ", " ")
    text = re.sub(r"\b(da nang|danang|dn|quan|tiem|cua hang|nha hang|coffee|cafe|bakery)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def category_family(value):
    raw = fold(value).replace("_", " ")
    if re.search(r"bakery|banh|bread", raw):
        return "bakery"
    if re.search(r"cafe|coffee|ca phe|tea room|tra sua|bubble|milk tea", raw):
        return "cafe"
    if re.search(r"bar|pub|beer|bia|cocktail|karaoke", raw):
        return "bar"
    if re.search(r"restaurant|food|diner|fast food|mon|mi quang|bo ne|quan an|nha hang|nhau|seafood", raw):
        return "restaurant"
    if re.search(r"hotel|hostel|lodging|accommodation|guest house|khach san|nha nghi", raw):
        return "accommodation"
    return raw.split(" ")[0] if raw else "unknown"


def levenshtein(left, right):
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_char != right_char),
            ))
        previous = current
    return previous[-1]


def name_similarity(left, right):
    left_name = normalized_entity_name(left)
    right_name = normalized_entity_name(right)
    if not left_name or not right_name:
        return 0
    if left_name == right_name:
        return 1
    left_tokens = set(left_name.split())
    right_tokens = set(right_name.split())
    token_score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
    edit_score = 1 - levenshtein(left_name, right_name) / max(len(left_name), len(right_name))
    return max(token_score, edit_score)


def read_canonical(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        records = []
        for row in csv.DictReader(handle):
            try:
                latitude = float(row["Lat"])
                longitude = float(row["Lon"])
            except (TypeError, ValueError):
                continue
            records.append({
                "id": row["Global_ID"],
                "name": row["Restaurant Name"],
                "category": category_family(row["Category"]),
                "coordinates": (latitude, longitude),
            })
        return records


def select_canonical_overlap(records, canonical, name_fn, category_fn, coordinate_fn, id_fn):
    stress_cases = {"tightExact": [], "probableFuzzy": [], "chainAmbiguous": []}
    for record in records:
        name = name_fn(record)
        coordinates = coordinate_fn(record)
        source_category = category_family(category_fn(record))
        if not name or not coordinates:
            continue
        candidates = []
        for canonical_record in canonical:
            if canonical_record["category"] != source_category:
                continue
            distance = haversine_meters(coordinates, canonical_record["coordinates"])
            if distance > 750:
                continue
            similarity = name_similarity(name, canonical_record["name"])
            if similarity >= 0.6 and distance <= 750:
                candidates.append((similarity, distance, canonical_record["id"]))
        tight = [candidate for candidate in candidates if candidate[0] >= 0.9 and candidate[1] <= 50]
        probable = [candidate for candidate in candidates if 0.6 <= candidate[0] < 0.9 and candidate[1] <= 150]
        chain = [candidate for candidate in candidates if candidate[0] >= 0.92 and candidate[1] <= 750]
        if len(chain) > 1:
            stress_cases["chainAmbiguous"].append(record)
        elif len(tight) == 1:
            stress_cases["tightExact"].append(record)
        elif len(probable) == 1:
            stress_cases["probableFuzzy"].append(record)

    selected = []
    for key, quota in (("tightExact", 4), ("probableFuzzy", 3), ("chainAmbiguous", 2)):
        ordered = sorted(stress_cases[key], key=id_fn)
        selected.extend(ordered[:quota])
        stress_cases[key] = [id_fn(record) for record in ordered[:quota]]
    return selected, stress_cases


def haversine_meters(left, right):
    radius = 6371000
    lat1, lon1 = math.radians(left[0]), math.radians(left[1])
    lat2, lon2 = math.radians(right[0]), math.radians(right[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def inside_bbox(latitude, longitude):
    west, south, east, north = BBOX
    return south <= latitude <= north and west <= longitude <= east


def select_records(records, group_fn, name_fn, coordinate_fn, id_fn):
    grouped = defaultdict(list)
    eligible = []
    for record in records:
        group = group_fn(record)
        name = name_fn(record)
        coordinates = coordinate_fn(record)
        if group in QUOTAS and name and coordinates and inside_bbox(*coordinates):
            grouped[group].append(record)
            eligible.append(record)

    selected = []
    for group, quota in QUOTAS.items():
        ordered = sorted(grouped[group], key=lambda item: (fold(name_fn(item)), id_fn(item)))
        selected.extend(ordered[:quota])

    duplicate_pairs = []
    ordered_records = sorted(eligible, key=id_fn)
    for index, left in enumerate(ordered_records):
        left_name = fold(name_fn(left))
        left_coordinates = coordinate_fn(left)
        if not left_name or not left_coordinates:
            continue
        for right in ordered_records[index + 1 :]:
            if fold(name_fn(right)) != left_name:
                continue
            right_coordinates = coordinate_fn(right)
            if right_coordinates and haversine_meters(left_coordinates, right_coordinates) <= 60:
                duplicate_pairs.append((left, right))
                break
        if duplicate_pairs:
            break

    if duplicate_pairs:
        selected.extend(duplicate_pairs[0])

    unique = {id_fn(record): record for record in selected}
    return [unique[key] for key in sorted(unique)], [id_fn(item) for item in duplicate_pairs[0]] if duplicate_pairs else []


def overture_group(record):
    category = fold((record.get("properties", {}).get("categories") or {}).get("primary"))
    if re.search(r"restaurant|cafe|coffee|bakery|bar|diner|food|tea", category):
        return "food"
    if re.search(r"hotel|hostel|lodging|guest house|accommodation", category):
        return "accommodation"
    if re.search(r"landmark|histor|museum|tour|attraction|art|park|bridge", category):
        return "attraction"
    if re.search(r"library|school|college|university|worship|church|temple|market|government", category):
        return "public_cultural"
    return None


def load_overture(canonical):
    reader = record_batch_reader(
        "place",
        bbox=BBOX,
        release=OVERTURE_RELEASE,
        connect_timeout=30,
        request_timeout=60,
        stac=True,
    )
    features = []
    for batch in reader:
        for row in batch.to_pylist():
            point = wkb.loads(row.pop("geometry"))
            features.append(
                {
                    "id": row.pop("id"),
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": list(point.coords)[0]},
                    "properties": row,
                }
            )
    features.sort(key=lambda item: item["id"])
    selected, duplicate_ids = select_records(
        features,
        overture_group,
        lambda item: item["properties"].get("names", {}).get("primary"),
        lambda item: (item["geometry"]["coordinates"][1], item["geometry"]["coordinates"][0]),
        lambda item: item["id"],
    )
    overlap, overlap_ids = select_canonical_overlap(
        features,
        canonical,
        lambda item: item["properties"].get("names", {}).get("primary"),
        lambda item: (item["properties"].get("categories") or {}).get("primary"),
        lambda item: (item["geometry"]["coordinates"][1], item["geometry"]["coordinates"][0]),
        lambda item: item["id"],
    )
    combined = {record["id"]: record for record in selected + overlap}
    return features, [combined[key] for key in sorted(combined)], duplicate_ids, overlap_ids


def osm_tags(element):
    return {tag.attrib["k"]: tag.attrib["v"] for tag in element.findall("tag")}


def osm_group(record):
    tags = record["tags"]
    tourism = fold(tags.get("tourism"))
    amenity = fold(tags.get("amenity"))
    historic = fold(tags.get("historic"))
    leisure = fold(tags.get("leisure"))
    shop = fold(tags.get("shop"))
    if amenity in {"restaurant", "cafe", "fast food"} or shop == "bakery":
        return "food"
    if tourism in {"hotel", "hostel", "guest house", "apartment", "motel"}:
        return "accommodation"
    if tourism or historic or leisure in {"park", "garden"}:
        return "attraction"
    if amenity in {"museum", "arts centre", "place of worship", "library", "theatre", "marketplace"}:
        return "public_cultural"
    return None


def load_osm(path, canonical):
    root = ET.parse(path).getroot()
    nodes = {
        node.attrib["id"]: (float(node.attrib["lat"]), float(node.attrib["lon"]))
        for node in root.findall("node")
    }
    records = []
    for element_type in ("node", "way", "relation"):
        for element in root.findall(element_type):
            tags = osm_tags(element)
            if not tags:
                continue
            if element_type == "node":
                coordinates = nodes.get(element.attrib["id"])
            elif element_type == "way":
                points = [nodes[ref.attrib["ref"]] for ref in element.findall("nd") if ref.attrib["ref"] in nodes]
                coordinates = (
                    sum(point[0] for point in points) / len(points),
                    sum(point[1] for point in points) / len(points),
                ) if points else None
            else:
                coordinates = None
            if not coordinates or not inside_bbox(*coordinates):
                continue
            records.append(
                {
                    "type": element_type,
                    "id": int(element.attrib["id"]),
                    "version": int(element.attrib.get("version", 0)),
                    "timestamp": element.attrib.get("timestamp"),
                    "latitude": coordinates[0],
                    "longitude": coordinates[1],
                    "tags": tags,
                }
            )
    records.sort(key=lambda item: (item["type"], item["id"]))
    selected, duplicate_ids = select_records(
        records,
        osm_group,
        lambda item: item["tags"].get("name") or item["tags"].get("name:en") or item["tags"].get("name:vi"),
        lambda item: (item["latitude"], item["longitude"]),
        lambda item: f'{item["type"]}:{item["id"]}',
    )
    overlap, overlap_ids = select_canonical_overlap(
        records,
        canonical,
        lambda item: item["tags"].get("name") or item["tags"].get("name:en") or item["tags"].get("name:vi"),
        lambda item: item["tags"].get("tourism") or item["tags"].get("amenity") or item["tags"].get("historic") or item["tags"].get("leisure") or item["tags"].get("shop"),
        lambda item: (item["latitude"], item["longitude"]),
        lambda item: f'{item["type"]}:{item["id"]}',
    )
    combined = {f'{record["type"]}:{record["id"]}': record for record in selected + overlap}
    return records, [combined[key] for key in sorted(combined)], duplicate_ids, overlap_ids, root.attrib


def commons_title(image_url):
    return unquote(urlparse(image_url).path.split("/")[-1]).replace("_", " ") if image_url else None


def metadata_value(metadata, key):
    return (metadata.get(key) or {}).get("value")


def load_commons(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    records = []
    for page in document.get("query", {}).get("pages", []):
        image_info = (page.get("imageinfo") or [{}])[0]
        metadata = image_info.get("extmetadata") or {}
        records.append(
            {
                "title": page.get("title", "").removeprefix("File:"),
                "pageId": page.get("pageid"),
                "descriptionUrl": image_info.get("descriptionurl"),
                "licenseShortName": metadata_value(metadata, "LicenseShortName"),
                "licenseUrl": metadata_value(metadata, "LicenseUrl"),
                "usageTerms": metadata_value(metadata, "UsageTerms"),
                "artist": metadata_value(metadata, "Artist"),
                "credit": metadata_value(metadata, "Credit"),
                "attributionRequired": metadata_value(metadata, "AttributionRequired"),
            }
        )
    return sorted(records, key=lambda item: item["title"])


def wikidata_group(record):
    categories = " ".join(record["instanceLabels"] + record["instanceIds"])
    value = fold(categories)
    if re.search(r"hotel|hostel|guest|khach san|nha nghi", value):
        return "accommodation"
    if re.search(r"museum|bridge|church|cathedral|bao tang|cau|nha tho", value):
        return "attraction"
    if re.search(r"ward|district|diocese|phuong|quan|giao phan", value):
        return "public_cultural"
    return None


def load_wikidata(path, commons_records):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    by_item = defaultdict(list)
    for binding in document.get("results", {}).get("bindings", []):
        by_item[binding["item"]["value"]].append(binding)
    commons_by_title = {record["title"]: record for record in commons_records}
    records = []
    for item_url, bindings in sorted(by_item.items()):
        first_binding = bindings[0]
        image_url = next((item["image"]["value"] for item in bindings if item.get("image")), None)
        image_title = commons_title(image_url)
        media = commons_by_title.get(image_title)
        records.append(
            {
                "qid": item_url.rsplit("/", 1)[-1],
                "label": first_binding.get("itemLabel", {}).get("value"),
                "coordinate": first_binding.get("coord", {}).get("value"),
                "instanceIds": sorted({item["instance"]["value"].rsplit("/", 1)[-1] for item in bindings if item.get("instance")}),
                "instanceLabels": sorted({item["instanceLabel"]["value"] for item in bindings if item.get("instanceLabel")}),
                "website": next((item["website"]["value"] for item in bindings if item.get("website")), None),
                "imageTitle": image_title,
                "mediaLicense": media.get("licenseShortName") if media else None,
                "mediaAttribution": media.get("artist") if media else None,
            }
        )
    grouped = defaultdict(list)
    for record in records:
        group = wikidata_group(record)
        if group:
            grouped[group].append(record)
    selected = []
    for group, quota in (("attraction", 6), ("accommodation", 8), ("public_cultural", 1)):
        selected.extend(sorted(grouped[group], key=lambda item: item["qid"])[:quota])
    return records, sorted({item["qid"]: item for item in selected}.values(), key=lambda item: item["qid"])


def snapshot_document(source, snapshot_ref, retrieved_at, access, license_info, records, **metadata):
    return {
        "status": "NON_CANONICAL_BOUNDED_REAL_SOURCE_SNAPSHOT",
        "source": source,
        "snapshotRef": snapshot_ref,
        "retrievedAt": retrieved_at,
        "bbox": {"west": BBOX[0], "south": BBOX[1], "east": BBOX[2], "north": BBOX[3], "crs": "EPSG:4326"},
        "access": access,
        "license": license_info,
        **metadata,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--osm-input", required=True)
    parser.add_argument("--wikidata-input", required=True)
    parser.add_argument("--commons-input", required=True)
    parser.add_argument("--canonical-input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--retrieved-at")
    args = parser.parse_args()
    retrieved_at = args.retrieved_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    output_dir = Path(args.output_dir)

    canonical = read_canonical(args.canonical_input)
    overture_all, overture_selected, overture_duplicate_ids, overture_overlap_ids = load_overture(canonical)
    osm_all, osm_selected, osm_duplicate_ids, osm_overlap_ids, osm_header = load_osm(args.osm_input, canonical)
    commons_records = load_commons(args.commons_input)
    wikidata_all, wikidata_selected = load_wikidata(args.wikidata_input, commons_records)

    documents = {
        "overture_places_2026-07-22.0_bounded.json": snapshot_document(
            "overture",
            "overture-2026-07-22.0-da-nang-core-bounded",
            retrieved_at,
            {"method": "overturemaps-python-client", "release": OVERTURE_RELEASE, "type": "place", "stac": True},
            {"license": "CDLA-Permissive-2.0", "policyClass": "OPEN_PERMISSIVE_CANDIDATE", "attribution": "Overture Maps Foundation"},
            overture_selected,
            boundedRecordCount=len(overture_all),
            selectedRecordCount=len(overture_selected),
            deterministicSelection="quota-by-poi-group-plus-disclosed-canonical-overlap-stress-cases-plus-first-near-duplicate-pair",
            boundedExtractSha256=sha256_bytes(stable_json(overture_all).encode("utf-8")),
            selectedDuplicatePairIds=overture_duplicate_ids,
            canonicalOverlapStressCaseIds=overture_overlap_ids,
        ),
        "osm_2026-08-10_bounded.json": snapshot_document(
            "osm",
            "osm-map-api-2026-08-10-da-nang-core-bounded",
            retrieved_at,
            {"method": "OSM Map API fallback after bounded Overpass dispatcher timeouts", "endpoint": "https://api.openstreetmap.org/api/0.6/map"},
            {"license": "ODbL-1.0", "policyClass": "OPEN_SHAREALIKE_ISOLATED", "attribution": "© OpenStreetMap contributors"},
            osm_selected,
            boundedRelevantRecordCount=len([record for record in osm_all if osm_group(record)]),
            selectedRecordCount=len(osm_selected),
            deterministicSelection="quota-by-poi-group-plus-disclosed-canonical-overlap-stress-cases-plus-first-near-duplicate-pair",
            rawInputSha256=sha256_file(args.osm_input),
            generator=osm_header.get("generator"),
            selectedDuplicatePairIds=osm_duplicate_ids,
            canonicalOverlapStressCaseIds=osm_overlap_ids,
        ),
        "wikidata_2026-08-10_bounded.json": snapshot_document(
            "wikidata",
            "wikidata-wdqs-2026-08-10-da-nang-core-bounded",
            retrieved_at,
            {"method": "Wikidata Query Service wikibase:box", "endpoint": "https://query.wikidata.org/sparql"},
            {"license": "CC0-1.0", "policyClass": "OPEN_KNOWLEDGE_AND_MEDIA_ATTRIBUTION_REQUIRED", "attribution": "Wikidata contributors"},
            wikidata_selected,
            boundedRecordCount=len(wikidata_all),
            selectedRecordCount=len(wikidata_selected),
            deterministicSelection="poi-type-allowlist-with-fixed-group-quotas-then-qid",
            rawInputSha256=sha256_file(args.wikidata_input),
        ),
        "wikimedia_commons_2026-08-10_metadata.json": snapshot_document(
            "wikimedia_commons",
            "wikimedia-commons-api-2026-08-10-linked-media-metadata",
            retrieved_at,
            {"method": "MediaWiki Action API imageinfo extmetadata", "endpoint": "https://commons.wikimedia.org/w/api.php", "binaryMediaDownloaded": False},
            {"license": "PER_FILE", "policyClass": "OPEN_KNOWLEDGE_AND_MEDIA_ATTRIBUTION_REQUIRED", "attribution": "Per-file artist, credit, and license metadata required"},
            commons_records,
            linkedMetadataRecordCount=len(commons_records),
            rawInputSha256=sha256_file(args.commons_input),
        ),
    }

    for file_name, document in documents.items():
        write_json(output_dir / file_name, document)

    manifest = {
        "status": "NON_CANONICAL_BOUNDED_REAL_SOURCE_SNAPSHOT_MANIFEST",
        "retrievedAt": retrieved_at,
        "bbox": {"west": BBOX[0], "south": BBOX[1], "east": BBOX[2], "north": BBOX[3], "crs": "EPSG:4326"},
        "googlePlacesIncluded": False,
        "canonicalRuntimeAffected": False,
        "snapshots": [
            {
                "file": file_name,
                "source": document["source"],
                "snapshotRef": document["snapshotRef"],
                "records": len(document["records"]),
                "sha256": sha256_file(output_dir / file_name),
            }
            for file_name, document in sorted(documents.items())
        ],
    }
    write_json(output_dir / "snapshot_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

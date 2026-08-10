import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from overturemaps.core import record_batch_reader
from shapely import wkb


USER_AGENT = "UrbanAgent-Phase4G/1.0 (candidate-only source validation)"
STATUS = "CANDIDATE_NON_RUNTIME_NOT_CANONICAL"


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(value) + "\n", encoding="utf-8", newline="\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fold(value):
    text = str(value or "").lower()
    translations = str.maketrans("àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ", "aaaaaaaaaaaaaaaaaeeeeeeeeeeeiiiiiooooooooooooooooouuuuuuuuuuuyyyyyd")
    return re.sub(r"[^a-z0-9]+", " ", text.translate(translations)).strip()


def request_json(url, *, data=None, timeout=300, retries=3):
    payload = data.encode("utf-8") if isinstance(data, str) else data
    headers = {
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    if payload is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    last_error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, data=payload, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Source request failed after {retries} attempts: {last_error}")


def boundary_dict(config):
    boundary = config["boundary"]
    return {
        "source": boundary["source"],
        "version": boundary["version"],
        "crs": boundary["crs"],
        "west": boundary["west"],
        "south": boundary["south"],
        "east": boundary["east"],
        "north": boundary["north"],
    }


def inside_boundary(latitude, longitude, boundary):
    return (
        boundary["south"] <= latitude <= boundary["north"]
        and boundary["west"] <= longitude <= boundary["east"]
    )


def clearly_out_of_city(values, config):
    folded = " ".join(fold(value) for value in values if value)
    terms = [fold(term) for term in config["boundary"]["clearOutOfCityTerms"]]
    return any(term and term in folded for term in terms) and "da nang" not in folded


def overture_address_values(properties):
    values = []
    for address in properties.get("addresses") or []:
        for key in ("freeform", "locality", "postcode", "region", "country"):
            if address.get(key):
                values.append(address[key])
    return values


def acquire_overture(config, cache_dir, retrieved_at):
    source_config = config["overture"]
    boundary = config["boundary"]
    bbox = (boundary["west"], boundary["south"], boundary["east"], boundary["north"])
    reader = record_batch_reader(
        source_config["type"],
        bbox=bbox,
        release=source_config["release"],
        connect_timeout=30,
        request_timeout=120,
        stac=True,
    )
    records = []
    raw_count = 0
    invalid_count = 0
    out_of_city_count = 0
    for batch in reader:
        for row in batch.to_pylist():
            raw_count += 1
            geometry = row.pop("geometry", None)
            if not geometry:
                invalid_count += 1
                continue
            point = wkb.loads(geometry)
            longitude, latitude = list(point.coords)[0]
            properties = row
            name = (properties.get("names") or {}).get("primary")
            if not name or not inside_boundary(latitude, longitude, boundary):
                invalid_count += 1
                continue
            if clearly_out_of_city(overture_address_values(properties), config):
                out_of_city_count += 1
                continue
            records.append({
                "id": properties.pop("id"),
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
                "properties": properties,
            })
    records.sort(key=lambda item: item["id"])
    document = {
        "status": STATUS,
        "source": "overture",
        "snapshotRef": f'overture-{source_config["release"]}-da-nang-full-supported-bbox',
        "retrievedAt": retrieved_at,
        "boundary": boundary_dict(config),
        "access": {
            "method": source_config["accessMethod"],
            "release": source_config["release"],
            "type": source_config["type"],
            "stac": True,
        },
        "license": source_config["license"],
        "rawRecordCount": raw_count,
        "invalidRecordCount": invalid_count,
        "clearlyOutOfCityCount": out_of_city_count,
        "records": records,
    }
    path = Path(cache_dir) / "overture_full_danang.json"
    write_json(path, document)
    return path, document


def overpass_query(config):
    boundary = config["boundary"]
    bbox = f'{boundary["south"]},{boundary["west"]},{boundary["north"]},{boundary["east"]}'
    return f'''[out:json][timeout:300];
(
  nwr["name"]["amenity"]({bbox});
  nwr["name"]["tourism"]({bbox});
  nwr["name"]["historic"]({bbox});
  nwr["name"]["leisure"]({bbox});
  nwr["name"]["shop"]({bbox});
);
out center tags meta;'''


def acquire_osm(config, cache_dir, retrieved_at):
    query = overpass_query(config)
    response = None
    endpoint_used = None
    errors = []
    for endpoint in config["osm"]["endpoints"]:
        try:
            response = request_json(
                endpoint,
                data=urllib.parse.urlencode({"data": query}),
                timeout=360,
                retries=2,
            )
            endpoint_used = endpoint
            break
        except RuntimeError as error:
            errors.append(str(error))
    if response is None:
        raise RuntimeError(f"All configured Overpass endpoints failed: {errors}")

    records = []
    invalid_count = 0
    out_of_city_count = 0
    for element in response.get("elements", []):
        tags = element.get("tags") or {}
        latitude = element.get("lat", (element.get("center") or {}).get("lat"))
        longitude = element.get("lon", (element.get("center") or {}).get("lon"))
        if not tags.get("name") or not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
            invalid_count += 1
            continue
        if clearly_out_of_city([
            tags.get("addr:city"),
            tags.get("addr:province"),
            tags.get("is_in"),
            tags.get("is_in:city"),
            tags.get("is_in:province"),
        ], config):
            out_of_city_count += 1
            continue
        records.append({
            "type": element["type"],
            "id": element["id"],
            "version": element.get("version", 0),
            "timestamp": element.get("timestamp"),
            "latitude": latitude,
            "longitude": longitude,
            "tags": tags,
        })
    records.sort(key=lambda item: (item["type"], item["id"]))
    document = {
        "status": STATUS,
        "source": "osm",
        "snapshotRef": "osm-overpass-da-nang-full-supported-bbox",
        "retrievedAt": retrieved_at,
        "boundary": boundary_dict(config),
        "access": {"method": config["osm"]["accessMethod"], "endpoint": endpoint_used},
        "license": config["osm"]["license"],
        "rawRecordCount": len(response.get("elements", [])),
        "invalidRecordCount": invalid_count,
        "clearlyOutOfCityCount": out_of_city_count,
        "records": records,
    }
    path = Path(cache_dir) / "osm_full_danang.json"
    write_json(path, document)
    return path, document


def wikidata_query(config):
    boundary = config["boundary"]
    southwest = f'Point({boundary["west"]} {boundary["south"]})'
    northeast = f'Point({boundary["east"]} {boundary["north"]})'
    return f'''SELECT ?item ?itemLabel ?coord ?instance ?instanceLabel ?website ?image WHERE {{
  SERVICE wikibase:box {{
    ?item wdt:P625 ?coord .
    bd:serviceParam wikibase:cornerWest "{southwest}"^^geo:wktLiteral .
    bd:serviceParam wikibase:cornerEast "{northeast}"^^geo:wktLiteral .
  }}
  ?item wdt:P31 ?instance .
  OPTIONAL {{ ?item wdt:P856 ?website . }}
  OPTIONAL {{ ?item wdt:P18 ?image . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "vi,en" . }}
}}
ORDER BY ?item ?instance
LIMIT 10000'''


def commons_title(image_url):
    return urllib.parse.unquote(urllib.parse.urlparse(image_url).path.split("/")[-1]).replace("_", " ") if image_url else None


def metadata_value(metadata, key):
    return (metadata.get(key) or {}).get("value")


def acquire_commons(config, titles, retrieved_at):
    records = []
    for offset in range(0, len(titles), 50):
        batch = titles[offset:offset + 50]
        params = urllib.parse.urlencode({
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "titles": "|".join(f"File:{title}" for title in batch),
        })
        response = request_json(f'{config["wikimediaCommons"]["endpoint"]}?{params}', timeout=120)
        for page in response.get("query", {}).get("pages", []):
            image_info = (page.get("imageinfo") or [{}])[0]
            metadata = image_info.get("extmetadata") or {}
            record = {
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
            records.append(record)
    records.sort(key=lambda item: item["title"])
    complete = []
    conflicts = []
    for record in records:
        missing = [field for field in ("pageId", "descriptionUrl", "licenseShortName", "licenseUrl", "artist") if not record.get(field)]
        if missing:
            conflicts.append({"title": record["title"], "missing": missing})
        else:
            complete.append(record)
    return complete, conflicts


def acquire_wikidata(config, cache_dir, retrieved_at):
    query = wikidata_query(config)
    response = None
    endpoint_used = None
    errors = []
    for endpoint in config["wikidata"]["endpoints"]:
        try:
            payload = urllib.parse.urlencode({"query": query, "format": "json"})
            response = request_json(endpoint, data=payload, timeout=300, retries=2)
            endpoint_used = endpoint
            break
        except RuntimeError as error:
            errors.append(str(error))
    if response is None:
        raise RuntimeError(f"All configured Wikidata endpoints failed: {errors}")

    by_item = {}
    for binding in response.get("results", {}).get("bindings", []):
        item_url = binding["item"]["value"]
        record = by_item.setdefault(item_url, {
            "qid": item_url.rsplit("/", 1)[-1],
            "label": binding.get("itemLabel", {}).get("value"),
            "coordinate": binding.get("coord", {}).get("value"),
            "instanceIds": set(),
            "instanceLabels": set(),
            "website": None,
            "imageTitle": None,
            "mediaLicense": None,
            "mediaAttribution": None,
        })
        if binding.get("instance"):
            record["instanceIds"].add(binding["instance"]["value"].rsplit("/", 1)[-1])
        if binding.get("instanceLabel"):
            record["instanceLabels"].add(binding["instanceLabel"]["value"])
        if binding.get("website") and not record["website"]:
            record["website"] = binding["website"]["value"]
        if binding.get("image") and not record["imageTitle"]:
            record["imageTitle"] = commons_title(binding["image"]["value"])

    records = []
    out_of_city_count = 0
    for item_url in sorted(by_item):
        record = by_item[item_url]
        record["instanceIds"] = sorted(record["instanceIds"])
        record["instanceLabels"] = sorted(record["instanceLabels"])
        if clearly_out_of_city([record["label"]], config):
            out_of_city_count += 1
            continue
        records.append(record)

    titles = sorted({record["imageTitle"] for record in records if record["imageTitle"]})
    commons_records, license_conflicts = acquire_commons(config, titles, retrieved_at)
    commons_by_title = {record["title"]: record for record in commons_records}
    for record in records:
        media = commons_by_title.get(record["imageTitle"])
        if media:
            record["mediaLicense"] = media["licenseShortName"]
            record["mediaAttribution"] = media["artist"]
        elif record["imageTitle"]:
            record["imageTitle"] = None

    wikidata_document = {
        "status": STATUS,
        "source": "wikidata",
        "snapshotRef": "wikidata-wdqs-da-nang-full-supported-bbox",
        "retrievedAt": retrieved_at,
        "boundary": boundary_dict(config),
        "access": {"method": config["wikidata"]["accessMethod"], "endpoint": endpoint_used},
        "license": config["wikidata"]["license"],
        "rawBindingCount": len(response.get("results", {}).get("bindings", [])),
        "rawRecordCount": len(by_item),
        "clearlyOutOfCityCount": out_of_city_count,
        "mediaLicenseConflictCount": len(license_conflicts),
        "mediaLicenseConflicts": license_conflicts,
        "records": records,
    }
    commons_document = {
        "status": STATUS,
        "source": "wikimedia_commons",
        "snapshotRef": "wikimedia-commons-da-nang-full-linked-media-metadata",
        "retrievedAt": retrieved_at,
        "boundary": boundary_dict(config),
        "access": {
            "method": config["wikimediaCommons"]["accessMethod"],
            "endpoint": config["wikimediaCommons"]["endpoint"],
            "binaryMediaDownloaded": False,
        },
        "license": config["wikimediaCommons"]["license"],
        "requestedTitleCount": len(titles),
        "completeRecordCount": len(commons_records),
        "licenseConflictCount": len(license_conflicts),
        "records": commons_records,
    }
    wikidata_path = Path(cache_dir) / "wikidata_full_danang.json"
    commons_path = Path(cache_dir) / "wikimedia_commons_full_danang_metadata.json"
    write_json(wikidata_path, wikidata_document)
    write_json(commons_path, commons_document)
    return wikidata_path, wikidata_document, commons_path, commons_document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--retrieved-at")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    retrieved_at = args.retrieved_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    overture_path, overture = acquire_overture(config, cache_dir, retrieved_at)
    osm_path, osm = acquire_osm(config, cache_dir, retrieved_at)
    wikidata_path, wikidata, commons_path, commons = acquire_wikidata(config, cache_dir, retrieved_at)

    snapshots = []
    for path, document in (
        (overture_path, overture),
        (osm_path, osm),
        (wikidata_path, wikidata),
        (commons_path, commons),
    ):
        snapshots.append({
            "source": document["source"],
            "snapshotRef": document["snapshotRef"],
            "retrievedAt": document["retrievedAt"],
            "access": document["access"],
            "license": document["license"],
            "recordCount": len(document["records"]),
            "rawRecordCount": document.get("rawRecordCount", document.get("requestedTitleCount", len(document["records"]))),
            "invalidRecordCount": document.get("invalidRecordCount", 0),
            "clearlyOutOfCityCount": document.get("clearlyOutOfCityCount", 0),
            "licenseConflictCount": document.get("mediaLicenseConflictCount", document.get("licenseConflictCount", 0)),
            "cacheFile": path.name,
            "cacheBytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })

    manifest = {
        "status": STATUS,
        "cityId": config["cityId"],
        "boundary": boundary_dict(config),
        "retrievedAt": retrieved_at,
        "googlePlacesIncluded": False,
        "canonicalRuntimeAffected": False,
        "acquisitionSeconds": round(time.perf_counter() - started, 3),
        "snapshots": sorted(snapshots, key=lambda item: item["source"]),
    }
    write_json(args.metadata_output, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""管理正式基准外部数据源的下载和校验。"""

from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from Env.benchmark.formal_spec import load_formal_spec


_COLLECTORS = {
    "existing_manifest",
    "tnm_1m",
    "overpass",
    "arcgis_query",
    "hrrr_ranges",
    "nasa_power",
    "direct",
}
_USER_AGENT = "paper10-formal-benchmark/1.0 (academic data collection)"


class SourceCatalogError(ValueError):
    """Raised when a declared public source cannot be retained exactly."""


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    evidence_class: str
    url: str
    licence: str
    local_path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceSpec:
    source_id: str
    evidence_class: str
    url: str
    licence: str
    local_path: Path
    collector: str
    expected_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id or not self.evidence_class or not self.licence:
            raise SourceCatalogError("source identity, evidence class, and licence are required")
        if not self.url.startswith("https://"):
            raise SourceCatalogError(f"source {self.source_id} must use HTTPS")
        if self.collector not in _COLLECTORS:
            raise SourceCatalogError(f"unknown collector {self.collector}")
        if self.expected_sha256 is not None and not _valid_sha256(
            self.expected_sha256
        ):
            raise SourceCatalogError(
                f"invalid expected SHA-256 for {self.source_id}"
            )


@dataclass(frozen=True, slots=True)
class HRRRMessageRange:
    selector: str
    start: int
    end: int


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_retained_file(record: SourceRecord) -> str:
    if not record.local_path.is_file():
        raise FileNotFoundError(record.local_path)
    if not _valid_sha256(record.sha256):
        raise SourceCatalogError(f"invalid SHA-256 for {record.source_id}")
    digest = sha256_file(record.local_path)
    if digest != record.sha256:
        raise SourceCatalogError(f"SHA-256 mismatch for {record.source_id}")
    return digest


def _request_bytes(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 120.0,
    attempts: int = 3,
) -> bytes:
    if attempts <= 0:
        raise SourceCatalogError("network attempts must be positive")
    request_headers = {"User-Agent": _USER_AGENT}
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url, data=data, headers=request_headers, method=method
    )
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if status < 200 or status >= 300:
                    raise SourceCatalogError(f"HTTP {status} for {url}")
                return response.read()
        except SourceCatalogError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise SourceCatalogError(
                    f"network acquisition failed for {url}: {exc}"
                ) from exc
            last_error = exc
        except OSError as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(min(2**attempt, 8))
    assert last_error is not None
    raise SourceCatalogError(
        f"network acquisition failed for {url}: {last_error}"
    ) from last_error


def _default_fetcher(spec: SourceSpec) -> bytes:
    return _request_bytes(spec.url)


def _record_for(spec: SourceSpec, digest: str) -> SourceRecord:
    return SourceRecord(
        spec.source_id,
        spec.evidence_class,
        spec.url,
        spec.licence,
        spec.local_path,
        digest,
    )


def acquire_source(
    spec: SourceSpec,
    *,
    recorded_sha256: str | None = None,
    fetcher: Callable[[SourceSpec], bytes] = _default_fetcher,
) -> SourceRecord:
    target = spec.local_path
    if target.exists():
        if recorded_sha256 is None:
            raise SourceCatalogError(
                f"existing source {spec.source_id} requires a manifest hash"
            )
        record = _record_for(spec, recorded_sha256)
        digest = validate_retained_file(record)
        if spec.expected_sha256 is not None and digest != spec.expected_sha256:
            raise SourceCatalogError(f"SHA-256 mismatch for {spec.source_id}")
        return record

    payload = fetcher(spec)
    if not payload:
        raise SourceCatalogError(f"empty response for {spec.source_id}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.partial")
    if partial.exists():
        raise SourceCatalogError(f"stale partial file for {spec.source_id}: {partial}")
    try:
        partial.write_bytes(payload)
        digest = sha256_file(partial)
        if spec.expected_sha256 is not None and digest != spec.expected_sha256:
            raise SourceCatalogError(f"SHA-256 mismatch for {spec.source_id}")
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return _record_for(spec, digest)


def parse_hrrr_message_ranges(
    index_text: str,
    selectors: tuple[str, ...],
    *,
    object_size: int,
) -> tuple[HRRRMessageRange, ...]:
    if object_size <= 0:
        raise SourceCatalogError("HRRR object size must be positive")
    entries: list[tuple[int, str]] = []
    for line_number, line in enumerate(index_text.splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split(":", 2)
        if len(fields) != 3:
            raise SourceCatalogError(f"invalid HRRR index line {line_number}")
        try:
            start = int(fields[1])
        except ValueError as exc:
            raise SourceCatalogError(
                f"invalid HRRR byte offset on line {line_number}"
            ) from exc
        entries.append((start, fields[2]))
    if not entries or any(
        entries[index][0] >= entries[index + 1][0]
        for index in range(len(entries) - 1)
    ):
        raise SourceCatalogError("HRRR index offsets must be strictly increasing")

    selected: list[HRRRMessageRange] = []
    for selector in selectors:
        matches = [index for index, (_, text) in enumerate(entries) if selector in text]
        if len(matches) != 1:
            raise SourceCatalogError(
                f"HRRR selector {selector!r} matched {len(matches)} messages"
            )
        index = matches[0]
        start = entries[index][0]
        end = entries[index + 1][0] - 1 if index + 1 < len(entries) else object_size - 1
        if end < start or end >= object_size:
            raise SourceCatalogError(f"invalid HRRR byte range for {selector}")
        selected.append(HRRRMessageRange(selector, start, end))
    return tuple(selected)


def _registered_hash(registry_path: Path, data_id: str) -> str | None:
    if not registry_path.is_file():
        return None
    with registry_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row.get("data_id") == data_id:
                value = row.get("sha256", "").strip().lower()
                return value if _valid_sha256(value) else None
    return None


def build_source_catalog(
    config_path: Path, output_root: Path
) -> tuple[SourceSpec, ...]:
    load_formal_spec(config_path)
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    required_ids = tuple(raw["required_source_ids"])
    registry = Path(config_path).parents[2] / "provenance" / "source_registry.csv"
    root = Path(output_root)
    definitions = {
        "usgs_3dep_10m": SourceSpec(
            "usgs_3dep_10m",
            "public_derived",
            "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage",
            "U.S. government public domain",
            root / "usgs_3dep_formal" / "dem_manifest.csv",
            "existing_manifest",
            _registered_hash(registry, "USGS-3DEP-FORMAL"),
        ),
        "usgs_1m": SourceSpec(
            "usgs_1m",
            "public_metadata",
            "https://tnmaccess.nationalmap.gov/api/v1/products",
            "U.S. government public domain",
            root / "usgs_1m_formal" / "tnm_queries.json",
            "tnm_1m",
        ),
        "tiger_2025_roads": SourceSpec(
            "tiger_2025_roads",
            "public_observation",
            "https://www2.census.gov/geo/tiger/TIGER2025/ROADS/",
            "U.S. government public domain",
            root / "census_tiger_roads_formal" / "roads_manifest.csv",
            "existing_manifest",
            _registered_hash(registry, "TIGER-ROADS-FORMAL"),
        ),
        "osm_power_towers": SourceSpec(
            "osm_power_towers",
            "public_observation",
            "https://overpass-api.de/api/interpreter",
            "OpenStreetMap ODbL 1.0",
            root / "osm_power_towers_formal" / "overpass_responses.json",
            "overpass",
        ),
        "hifld_corridors": SourceSpec(
            "hifld_corridors",
            "public_observation",
            "https://services2.arcgis.com/LYMgRMwHfrWWEg3s/ArcGIS/rest/services/HIFLD_US_Electric_Power_Transmission_Lines/FeatureServer/0/query",
            "HIFLD public feature-service terms; redistribution review required",
            root / "hifld_corridors_formal" / "arcgis_responses.json",
            "arcgis_query",
        ),
        "osm_operational_features": SourceSpec(
            "osm_operational_features",
            "public_observation",
            "https://overpass-api.de/api/interpreter",
            "OpenStreetMap ODbL 1.0",
            root / "osm_operational_features" / "overpass_responses.json",
            "overpass",
        ),
        "noaa_hrrr": SourceSpec(
            "noaa_hrrr",
            "public_derived",
            "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/",
            "NOAA public data; attribution requested",
            root / "noaa_hrrr_formal" / "hrrr_selection_manifest.json",
            "hrrr_ranges",
        ),
        "nasa_power": SourceSpec(
            "nasa_power",
            "public_observation",
            "https://power.larc.nasa.gov/api/temporal/hourly/point",
            "NASA POWER data access policy",
            root / "nasa_power_formal" / "power_responses.json",
            "nasa_power",
        ),
        "insplad": SourceSpec(
            "insplad",
            "literature_parameter",
            "https://arxiv.org/pdf/2311.01619",
            "Author preprint retained for scholarly verification",
            root / "inspection_taxonomy" / "insplad_2311.01619.pdf",
            "direct",
        ),
        "ttpla": SourceSpec(
            "ttpla",
            "literature_parameter",
            "https://arxiv.org/pdf/2010.10032",
            "Author preprint retained for scholarly verification",
            root / "inspection_taxonomy" / "ttpla_2010.10032.pdf",
            "direct",
        ),
    }
    missing = [source_id for source_id in required_ids if source_id not in definitions]
    if missing:
        raise SourceCatalogError(f"no collector definitions for {missing}")
    return tuple(definitions[source_id] for source_id in required_ids)


def _reuse_record(spec: SourceSpec, recorded_sha256: str | None) -> SourceRecord | None:
    if not spec.local_path.exists():
        return None
    if recorded_sha256 is None and spec.expected_sha256 is None:
        raise SourceCatalogError(
            f"existing source {spec.source_id} requires a manifest hash"
        )
    digest = recorded_sha256 or spec.expected_sha256
    assert digest is not None
    record = _record_for(spec, digest)
    validate_retained_file(record)
    if spec.expected_sha256 is not None and digest != spec.expected_sha256:
        raise SourceCatalogError(f"SHA-256 mismatch for {spec.source_id}")
    return record


def _write_json_bundle(
    spec: SourceSpec, value: dict[str, Any], recorded_sha256: str | None
) -> SourceRecord:
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if (
        spec.local_path.is_file()
        and recorded_sha256 is None
        and spec.expected_sha256 is None
    ):
        if spec.local_path.read_bytes() != payload:
            raise SourceCatalogError(
                f"existing deterministic bundle is different: {spec.local_path}"
            )
        digest = sha256_file(spec.local_path)
        return _record_for(spec, digest)
    reused = _reuse_record(spec, recorded_sha256)
    if reused is not None:
        return reused
    return acquire_source(spec, fetcher=lambda _: payload)


def _bbox_query(raw: dict[str, Any], region_id: str) -> tuple[float, float, float, float]:
    west, south, east, north = raw["regions"][region_id]["bbox_wgs84"]
    return float(west), float(south), float(east), float(north)


def _collect_tnm(
    spec: SourceSpec, raw: dict[str, Any], recorded_sha256: str | None
) -> SourceRecord:
    responses = []
    for region_id in sorted(raw["regions"]):
        west, south, east, north = _bbox_query(raw, region_id)
        query = urllib.parse.urlencode(
            {
                "bbox": f"{west},{south},{east},{north}",
                "datasets": "Digital Elevation Model (DEM) 1 meter",
                "outputFormat": "JSON",
                "max": 100,
            }
        )
        url = f"{spec.url}?{query}"
        response = json.loads(_request_bytes(url).decode("utf-8"))
        if not isinstance(response, dict) or not response.get("items"):
            raise SourceCatalogError(f"TNM returned no 1-m products for {region_id}")
        responses.append({"region_id": region_id, "query_url": url, "response": response})
    return _write_json_bundle(spec, {"responses": responses}, recorded_sha256)


def _overpass_query(source_id: str, bbox: tuple[float, float, float, float]) -> str:
    west, south, east, north = bbox
    bounds = f"({south},{west},{north},{east})"
    if source_id == "osm_power_towers":
        return f'[out:json][timeout:180];node["power"="tower"]{bounds};out body;'
    return (
        f"[out:json][timeout:180];(way[\"building\"]{bounds};"
        f"way[\"barrier\"]{bounds};node[\"natural\"=\"tree\"]{bounds};);out geom;"
    )


def _collect_overpass(
    spec: SourceSpec, raw: dict[str, Any], recorded_sha256: str | None
) -> SourceRecord:
    windows_root = spec.local_path.parent / "windows"
    if not windows_root.is_dir():
        raise SourceCatalogError(
            f"retained Overpass windows are absent for {spec.source_id}: {windows_root}"
        )
    entries = []
    for region_id in sorted(raw["regions"]):
        for window_index in (0, 1):
            window_root = windows_root / region_id / f"window_{window_index}"
            if spec.source_id == "osm_power_towers":
                expected = (("towers", window_root / "osm_towers_raw.json"),)
            else:
                combined = window_root / "osm_obstacles_raw.json"
                expected = (
                    (("obstacles", combined),)
                    if combined.is_file()
                    else tuple(
                        (kind, window_root / f"osm_{kind}_raw.json")
                        for kind in ("buildings", "barriers", "trees")
                    )
                )
            for source_kind, path in expected:
                if not path.is_file():
                    raise SourceCatalogError(
                        f"retained Overpass window file is absent: {path}"
                    )
                bundle = json.loads(path.read_text(encoding="utf-8"))
                response = bundle.get("response") if isinstance(bundle, dict) else None
                elements = response.get("elements") if isinstance(response, dict) else None
                endpoint = bundle.get("endpoint") if isinstance(bundle, dict) else None
                query = bundle.get("query") if isinstance(bundle, dict) else None
                if (
                    bundle.get("source_kind") != source_kind
                    or not isinstance(elements, list)
                    or not isinstance(endpoint, str)
                    or not endpoint
                    or not isinstance(query, str)
                    or not query
                ):
                    raise SourceCatalogError(
                        f"invalid retained Overpass window bundle: {path}"
                    )
                entries.append(
                    {
                        "region_id": region_id,
                        "window_index": window_index,
                        "source_kind": source_kind,
                        "endpoint": endpoint,
                        "query_sha256": hashlib.sha256(
                            query.encode("utf-8")
                        ).hexdigest(),
                        "element_count": len(elements),
                        "local_path": path.relative_to(
                            spec.local_path.parent
                        ).as_posix(),
                        "sha256": sha256_file(path),
                    }
                )
    return _write_json_bundle(
        spec, {"window_files": entries}, recorded_sha256
    )


def _collect_arcgis(
    spec: SourceSpec, raw: dict[str, Any], recorded_sha256: str | None
) -> SourceRecord:
    responses = []
    for region_id in sorted(raw["regions"]):
        west, south, east, north = _bbox_query(raw, region_id)
        query = urllib.parse.urlencode(
            {
                "f": "json",
                "where": "1=1",
                "geometry": f"{west},{south},{east},{north}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": 4326,
                "outSR": 4326,
                "outFields": "*",
                "returnGeometry": "true",
            }
        )
        url = f"{spec.url}?{query}"
        response = json.loads(_request_bytes(url).decode("utf-8"))
        if not isinstance(response, dict) or "features" not in response:
            raise SourceCatalogError(f"invalid HIFLD response for {region_id}")
        responses.append({"region_id": region_id, "query_url": url, "response": response})
    return _write_json_bundle(spec, {"responses": responses}, recorded_sha256)


def _collect_nasa_power(
    spec: SourceSpec, raw: dict[str, Any], recorded_sha256: str | None
) -> SourceRecord:
    date = f"{raw['weather']['test_year']}{raw['weather']['month_days'][0].replace('-', '')}"
    responses = []
    for region_id in sorted(raw["regions"]):
        west, south, east, north = _bbox_query(raw, region_id)
        query = urllib.parse.urlencode(
            {
                "parameters": "T2M,WS10M,WD10M",
                "community": "RE",
                "longitude": (west + east) / 2.0,
                "latitude": (south + north) / 2.0,
                "start": date,
                "end": date,
                "format": "JSON",
                "time-standard": "UTC",
            }
        )
        url = f"{spec.url}?{query}"
        response = json.loads(_request_bytes(url).decode("utf-8"))
        if not isinstance(response, dict) or "properties" not in response:
            raise SourceCatalogError(f"invalid NASA POWER response for {region_id}")
        responses.append({"region_id": region_id, "query_url": url, "response": response})
    return _write_json_bundle(spec, {"responses": responses}, recorded_sha256)


def _head_content_length(url: str, *, attempts: int = 3) -> int:
    if attempts <= 0:
        raise SourceCatalogError("network attempts must be positive")
    request = urllib.request.Request(
        url, headers={"User-Agent": _USER_AGENT}, method="HEAD"
    )
    last_error: OSError | None = None
    value = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120.0) as response:
                value = response.headers.get("Content-Length")
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise SourceCatalogError(
                    f"cannot read HRRR object size for {url}"
                ) from exc
            last_error = exc
        except OSError as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(min(2**attempt, 8))
    else:
        assert last_error is not None
        raise SourceCatalogError(
            f"cannot read HRRR object size for {url}"
        ) from last_error
    if value is None or int(value) <= 0:
        raise SourceCatalogError(f"missing HRRR object size for {url}")
    return int(value)


def _atomic_payload(path: Path, payload: bytes, expected_size: int | None = None) -> str:
    if not payload or (expected_size is not None and len(payload) != expected_size):
        raise SourceCatalogError(f"invalid retained payload size for {path}")
    if path.exists():
        raise SourceCatalogError(f"refusing to replace retained payload: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    if partial.exists():
        raise SourceCatalogError(f"stale partial file {partial}")
    try:
        partial.write_bytes(payload)
        digest = sha256_file(partial)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return digest


def _validate_existing_payload(
    path: Path, expected_size: int
) -> str | None:
    source = Path(path)
    if not source.exists():
        return None
    if not source.is_file() or source.stat().st_size != expected_size:
        raise SourceCatalogError(f"retained payload size mismatch: {source}")
    return sha256_file(source)


def _write_hrrr_metadata(
    path: Path,
    url: str,
    object_size: int,
    index_sha256: str,
    ranges: tuple[HRRRMessageRange, ...],
) -> None:
    if object_size <= 0 or not _valid_sha256(index_sha256) or not ranges:
        raise SourceCatalogError("invalid HRRR metadata values")
    value = {
        "schema_version": "hrrr-object-metadata-v1",
        "url": url,
        "object_size": object_size,
        "index_sha256": index_sha256,
        "ranges": [
            {
                "selector": item.selector,
                "start": item.start,
                "end": item.end,
            }
            for item in ranges
        ],
    }
    payload = (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if path.is_file():
        if path.read_bytes() != payload:
            raise SourceCatalogError(f"existing HRRR metadata is different: {path}")
        return
    _atomic_payload(path, payload, len(payload))


def _load_hrrr_metadata(
    path: Path,
    expected_url: str,
    selectors: tuple[str, ...],
) -> tuple[int, str, tuple[HRRRMessageRange, ...]]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceCatalogError(f"cannot read HRRR metadata: {path}") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "url",
        "object_size",
        "index_sha256",
        "ranges",
    }:
        raise SourceCatalogError(f"invalid HRRR metadata structure: {path}")
    if (
        value["schema_version"] != "hrrr-object-metadata-v1"
        or value["url"] != expected_url
        or isinstance(value["object_size"], bool)
        or not isinstance(value["object_size"], int)
        or value["object_size"] <= 0
        or not isinstance(value["index_sha256"], str)
        or not _valid_sha256(value["index_sha256"])
        or not isinstance(value["ranges"], list)
    ):
        raise SourceCatalogError(f"invalid HRRR metadata values: {path}")
    ranges = []
    for record in value["ranges"]:
        if not isinstance(record, dict) or set(record) != {
            "selector",
            "start",
            "end",
        }:
            raise SourceCatalogError(f"invalid HRRR range metadata: {path}")
        ranges.append(
            HRRRMessageRange(
                str(record["selector"]),
                int(record["start"]),
                int(record["end"]),
            )
        )
    parsed = tuple(ranges)
    if tuple(item.selector for item in parsed) != selectors:
        raise SourceCatalogError(f"HRRR metadata selectors differ: {path}")
    return int(value["object_size"]), str(value["index_sha256"]), parsed


def _hrrr_objects(raw: dict[str, Any]) -> tuple[tuple[str, int, int], ...]:
    weather = raw["weather"]
    objects: set[tuple[str, int, int]] = set()
    for year in (weather["train_year"], weather["validation_year"], weather["test_year"]):
        for month_day in weather["month_days"]:
            day = datetime.strptime(f"{year}-{month_day}", "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            for cycle in weather["initialization_hours_utc"]:
                initialization = day + timedelta(hours=int(cycle))
                for lead in weather["forecast_leads_hours"]:
                    objects.add((initialization.strftime("%Y%m%d"), initialization.hour, int(lead)))
                    valid_time = initialization + timedelta(hours=int(lead))
                    objects.add((valid_time.strftime("%Y%m%d"), valid_time.hour, 0))
    return tuple(sorted(objects))


def _collect_hrrr_object(
    spec: SourceSpec,
    selectors: tuple[str, ...],
    output_directory: Path,
    request: tuple[str, int, int],
) -> tuple[dict[str, Any], ...]:
    date, cycle, lead = request
    stem = f"hrrr.{date}/conus/hrrr.t{cycle:02d}z.wrfsfcf{lead:02d}.grib2"
    url = urllib.parse.urljoin(spec.url, stem)
    metadata_path = (
        output_directory.parent
        / "metadata"
        / f"{date}_t{cycle:02d}_f{lead:02d}.json"
    )
    if metadata_path.is_file():
        object_size, index_sha256, ranges = _load_hrrr_metadata(
            metadata_path, url, selectors
        )
    else:
        index_payload = _request_bytes(f"{url}.idx", attempts=8)
        object_size = _head_content_length(url, attempts=8)
        ranges = parse_hrrr_message_ranges(
            index_payload.decode("utf-8"), selectors, object_size=object_size
        )
        index_sha256 = hashlib.sha256(index_payload).hexdigest()
        _write_hrrr_metadata(
            metadata_path, url, object_size, index_sha256, ranges
        )
    metadata_sha256 = sha256_file(metadata_path)
    entries = []
    for message in ranges:
        slug = message.selector.split(":", 1)[0].lower()
        target = output_directory / f"{date}_t{cycle:02d}_f{lead:02d}_{slug}.grib2"
        expected_size = message.end - message.start + 1
        digest = _validate_existing_payload(target, expected_size)
        if digest is None:
            payload = _request_bytes(
                url,
                headers={"Range": f"bytes={message.start}-{message.end}"},
                attempts=8,
            )
            digest = _atomic_payload(target, payload, expected_size)
        entries.append(
            {
                "date": date,
                "cycle": cycle,
                "lead": lead,
                "selector": message.selector,
                "url": url,
                "byte_range": [message.start, message.end],
                "local_path": str(target),
                "sha256": digest,
                "metadata_path": str(metadata_path),
                "metadata_sha256": metadata_sha256,
                "object_size": object_size,
                "index_sha256": index_sha256,
            }
        )
    return tuple(entries)


def _collect_hrrr(
    spec: SourceSpec, raw: dict[str, Any], recorded_sha256: str | None
) -> SourceRecord:
    reused = _reuse_record(spec, recorded_sha256)
    if reused is not None:
        return reused
    output_directory = spec.local_path.parent / "messages"
    entries: list[dict[str, Any]] = []
    selectors = tuple(raw["weather"]["variables"])
    executor = ThreadPoolExecutor(max_workers=8)
    futures = {
        executor.submit(
            _collect_hrrr_object,
            spec,
            selectors,
            output_directory,
            request,
        ): request
        for request in _hrrr_objects(raw)
    }
    try:
        for future in as_completed(futures):
            entries.extend(future.result())
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    entries.sort(
        key=lambda row: (
            row["date"],
            row["cycle"],
            row["lead"],
            row["selector"],
        )
    )
    return _write_json_bundle(spec, {"messages": entries}, None)


def collect_source(
    spec: SourceSpec,
    raw_config: dict[str, Any],
    *,
    recorded_sha256: str | None = None,
) -> SourceRecord:
    if spec.collector == "existing_manifest":
        digest = spec.expected_sha256 or recorded_sha256
        if digest is None:
            raise SourceCatalogError(
                f"no registered hash for existing manifest {spec.source_id}"
            )
        record = _record_for(spec, digest)
        validate_retained_file(record)
        return record
    if spec.collector == "direct":
        return acquire_source(spec, recorded_sha256=recorded_sha256)
    if spec.collector == "tnm_1m":
        return _collect_tnm(spec, raw_config, recorded_sha256)
    if spec.collector == "overpass":
        return _collect_overpass(spec, raw_config, recorded_sha256)
    if spec.collector == "arcgis_query":
        return _collect_arcgis(spec, raw_config, recorded_sha256)
    if spec.collector == "nasa_power":
        return _collect_nasa_power(spec, raw_config, recorded_sha256)
    if spec.collector == "hrrr_ranges":
        return _collect_hrrr(spec, raw_config, recorded_sha256)
    raise SourceCatalogError(f"unsupported collector {spec.collector}")


def read_collection_hashes(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    hashes: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source_id = row.get("source_id", "")
            digest = row.get("sha256", "").lower()
            if not source_id or not _valid_sha256(digest):
                raise SourceCatalogError(f"invalid collection manifest row in {path}")
            if source_id in hashes:
                raise SourceCatalogError(f"duplicate source ID {source_id} in {path}")
            hashes[source_id] = digest
    return hashes


def write_collection_manifest(
    path: Path,
    records: Iterable[SourceRecord],
    collectors: dict[str, str],
) -> None:
    fieldnames = [
        "source_id",
        "evidence_class",
        "url",
        "access_time_utc",
        "licence",
        "local_path",
        "byte_count",
        "sha256",
        "collector",
    ]
    rows_by_id: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != fieldnames:
                raise SourceCatalogError(
                    f"unexpected collection manifest columns in {path}"
                )
            for row in reader:
                source_id = row.get("source_id", "")
                if not source_id or source_id in rows_by_id:
                    raise SourceCatalogError(
                        f"invalid or duplicate source ID in {path}: {source_id!r}"
                    )
                rows_by_id[source_id] = row

    access_time = datetime.now(timezone.utc).isoformat()
    for record in records:
        rows_by_id[record.source_id] = {
            "source_id": record.source_id,
            "evidence_class": record.evidence_class,
            "url": record.url,
            "access_time_utc": access_time,
            "licence": record.licence,
            "local_path": str(record.local_path),
            "byte_count": record.local_path.stat().st_size,
            "sha256": record.sha256,
            "collector": collectors[record.source_id],
        }
    rows = [rows_by_id[source_id] for source_id in sorted(rows_by_id)]
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    try:
        with partial.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

"""
PhysChem check-and-upload for glider NPC profiles.

For each NPC file not yet fully confirmed in the local upload record:
  1. Read mission.platform, mission.missionNumber, operation.timeStart.
  2. Query the PhysChem API.
  3. If already present: store mission_id + operation_id → fully confirmed.
  4. If not present: upload to S3, store None → re-check next sync to get IDs.

Entry point used by app.py:
    sync_all_npc_to_physchem(processed_dir, base_url, aws_key_id, aws_secret)

Public helper used by app.py to build PhysChem editor links:
    load_upload_record(processed_dir) -> dict[filename, {'mission_id', 'operation_id'} | None]
"""
import glob
import json
import logging
import os
import re

import boto3
import requests

logger = logging.getLogger(__name__)

S3_ENDPOINT   = 'https://s3.hi.no'
S3_BUCKET     = 'transient-data'
S3_KEY_PREFIX = 'physchem/incoming/regular_stations/test'

os.environ.setdefault('AWS_REQUEST_CHECKSUM_CALCULATION',  'when_required')
os.environ.setdefault('AWS_RESPONSE_CHECKSUM_VALIDATION', 'when_required')

_RECORD_FILE = 'physchem_uploaded.json'


# ── Local upload record ────────────────────────────────────────────────────────
# Format: { "glider_profile_0001_20260101T0000.npc": {"mission_id": 2353, "operation_id": 38962} }
# A None value means the file was uploaded but IDs are not yet confirmed.

def _record_path(processed_dir: str) -> str:
    return os.path.join(processed_dir, _RECORD_FILE)


def load_upload_record(processed_dir: str) -> dict:
    """
    Load the upload record from disk.

    Returns dict mapping NPC basename to either:
      {'mission_id': int, 'operation_id': int}  — fully confirmed
      None                                       — uploaded, IDs pending
    """
    path = _record_path(processed_dir)
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, list):
            # Backward compat: old format was a plain list of confirmed filenames
            return {fname: None for fname in data}
        return data
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_record(processed_dir: str, record: dict) -> None:
    path = _record_path(processed_dir)
    tmp  = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(record, fh, indent=None)
    os.replace(tmp, path)


# ── NPC metadata parser ────────────────────────────────────────────────────────

def _read_npc_meta(npc_path: str) -> dict:
    with open(npc_path, 'r', encoding='utf-8') as fh:
        content = fh.read()
    m1 = re.search(r'# Metadata:', content)
    m2 = re.search(r'% Readings:', content)
    if not m1 or not m2:
        raise ValueError(f"Not a valid NPC file: {npc_path}")
    meta = {}
    for line in content[m1.end():m2.start()].splitlines():
        line = line.strip()
        if not line or ':' not in line:
            continue
        key, _, val = line.partition(':')
        meta[key.strip()] = val.strip()
    return meta


# ── PhysChem API ───────────────────────────────────────────────────────────────

def check_if_in_physchem(npc_path: str,
                          base_url: str) -> tuple[bool, int | None, int | None]:
    """
    Check whether this profile's operation already exists in PhysChem.

    Returns (is_present, mission_id, operation_id).
    mission_id and operation_id are None when is_present is False.
    """
    meta        = _read_npc_meta(npc_path)
    platform    = meta.get('mission.platform', '')
    mission_num = int(meta.get('mission.missionNumber', 0))
    time_start  = meta.get('operation.timeStart', '')

    resp = requests.get(f"{base_url}/mission/list",
                        params={'platform': platform}, timeout=30)
    resp.raise_for_status()

    mission_id = None
    for m in resp.json():
        if m.get('missionNumber') == mission_num:
            mission_id = m['id']
            break
    if mission_id is None:
        return False, None, None

    resp = requests.get(
        f"{base_url}/mission/{mission_id}/operation/list",
        params={'extend': 'false', 'instrumentTypeList': 'false'},
        timeout=30,
    )
    resp.raise_for_status()

    for op in resp.json():
        if op.get('timeStart', '') == time_start:
            return True, mission_id, int(op['id'])

    return False, None, None


# ── S3 upload ─────────────────────────────────────────────────────────────────

def upload_npc(npc_path: str, aws_access_key_id: str,
               aws_secret_access_key: str) -> None:
    s3  = boto3.resource(
        service_name='s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    key = f"{S3_KEY_PREFIX}/{os.path.basename(npc_path)}"
    with open(npc_path, 'rb') as fh:
        s3.Bucket(S3_BUCKET).put_object(Key=key, Body=fh)


# ── Per-directory sync ─────────────────────────────────────────────────────────

def _sync_npc_dir(npc_dir: str, base_url: str,
                  aws_access_key_id: str, aws_secret_access_key: str,
                  record: dict) -> dict:
    """
    Sync one _npc/ directory.

    - Files with confirmed IDs (dict entry) are skipped entirely.
    - Files with None entry (uploaded, IDs pending) are re-checked for IDs.
    - New files are checked and uploaded if absent.

    ``record`` is updated in-place.
    """
    counts = {'checked': 0, 'uploaded': 0, 'present': 0, 'skipped': 0, 'errors': 0}

    for npc_path in sorted(glob.glob(os.path.join(npc_dir, '*.npc'))):
        fname = os.path.basename(npc_path)
        entry = record.get(fname, ...)

        if entry is not ... and isinstance(entry, dict):
            # Already fully confirmed with IDs — nothing to do
            counts['skipped'] += 1
            continue

        counts['checked'] += 1
        try:
            is_present, mission_id, operation_id = check_if_in_physchem(npc_path, base_url)
            if is_present:
                record[fname] = {'mission_id': mission_id, 'operation_id': operation_id}
                logger.info("PhysChem: already present  — %s", fname)
                counts['present'] += 1
            else:
                upload_npc(npc_path, aws_access_key_id, aws_secret_access_key)
                record[fname] = None   # IDs will be confirmed on next sync
                logger.info("PhysChem: uploaded          — %s", fname)
                counts['uploaded'] += 1
        except Exception as exc:
            logger.warning("PhysChem: error for %s — %s", fname, exc)
            counts['errors'] += 1

    return counts


# ── Top-level entry point ──────────────────────────────────────────────────────

def sync_all_npc_to_physchem(processed_dir: str, base_url: str,
                              aws_access_key_id: str,
                              aws_secret_access_key: str) -> None:
    """
    Walk every *_npc/ directory under processed_dir and sync to PhysChem.
    Skips silently if credentials or base_url are not configured.
    """
    if not base_url:
        logger.info("PhysChem sync skipped — PHYSCHEM_BASE_URL not set")
        return
    if not aws_access_key_id or not aws_secret_access_key:
        logger.info("PhysChem sync skipped — AWS credentials not set")
        return

    npc_dirs = sorted(
        d for d in glob.glob(os.path.join(processed_dir, '**', '*_npc'), recursive=True)
        if os.path.isdir(d) and '.staging' not in d.replace('\\', '/').split('/')
    )
    if not npc_dirs:
        logger.debug("PhysChem sync: no _npc directories found under %s", processed_dir)
        return

    record = load_upload_record(processed_dir)
    total  = {'checked': 0, 'uploaded': 0, 'present': 0, 'skipped': 0, 'errors': 0}

    for npc_dir in npc_dirs:
        r = _sync_npc_dir(npc_dir, base_url, aws_access_key_id, aws_secret_access_key, record)
        for k in total:
            total[k] += r[k]

    _save_record(processed_dir, record)

    logger.info(
        "PhysChem sync complete — skipped: %d  checked: %d  "
        "uploaded: %d  already present: %d  errors: %d",
        total['skipped'], total['checked'],
        total['uploaded'], total['present'], total['errors'],
    )

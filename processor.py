"""
Slocum glider binary data processor.
Reads Slocum binary files, derives temperature and salinity profiles on a
1-m depth grid, and writes per-mission NetCDF files.

Supported file combinations (tried in order per directory):
  1. .tcd + .scd  — SFMC merged science + surface/GPS
  2. .tcd + .sbd  — SFMC science + raw subset (GPS companion)
  3. .tbd + .sbd  — trimmed science + subset GPS companion
  4. .sbd alone   — subset binary with both science and GPS variables

A "mission" is a continuous stretch of data; gaps > MISSION_GAP_DAYS days
are treated as separate missions.

Cache files (.cac) are required by dbdreader to decode the binary format.
If a cache file is missing the processor will log a WARNING with the exact
filename needed so you can copy it into the CAC_DIR.
"""
import os
import re
import glob
import logging

import numpy as np
import pandas as pd
import xarray as xr
import gsw
import dbdreader
from dbdreader import DBD

logger = logging.getLogger(__name__)

# ── Depth-binning parameters ───────────────────────────────────────────────────
BIN_MAX = 200
BIN_INTERVAL = 1
_BINS_CALC = np.arange(BIN_INTERVAL / 2, BIN_MAX, BIN_INTERVAL)
BINS = _BINS_CALC[:-1] + BIN_INTERVAL / 2   # bin centres

MISSION_GAP_DAYS = 7

# ── Cache-error helpers ────────────────────────────────────────────────────────

_missing_cac_warned: set[str] = set()   # avoid repeating the same warning


def _log_missing_cac(exc: Exception, filepath: str, cac_dir: str) -> None:
    """Emit a clear WARNING for every missing .cac hash in a DbdError."""
    # e.data is MissingCacheFileData(missing_cache_files={hash: [files]}, cache_dir=...)
    try:
        missing: dict = exc.data.missing_cache_files
    except AttributeError:
        missing = {}
        # fall back to regex on the message string
        for m in re.finditer(r'\b([0-9a-f]{8})\b', str(exc)):
            missing[m.group(1)] = [filepath]

    for hash_name, files in missing.items():
        if hash_name not in _missing_cac_warned:
            _missing_cac_warned.add(hash_name)
            logger.warning(
                "Missing cache file: %s.cac  (needed for %s)\n"
                "  → Copy %s.cac into %s and restart.",
                hash_name,
                ", ".join(os.path.basename(f) for f in files),
                hash_name,
                cac_dir,
            )


def _open_dbd(filepath: str, cac_dir: str):
    """
    Open a DBD file with the given cache directory.
    Returns a DBD object or None, logging a clear message on cache miss.
    """
    try:
        return DBD(filepath, cacheDir=cac_dir)
    except dbdreader.DbdError as exc:
        _log_missing_cac(exc, filepath, cac_dir)
        return None
    except Exception as exc:
        logger.debug("Cannot open %s: %s", filepath, exc)
        return None


# ── Low-level DBD helpers ──────────────────────────────────────────────────────

def _safe_get(dbd, var):
    try:
        return dbd.get(var)
    except Exception:
        return np.array([]), np.array([])


def _extract_science(dbd):
    t,  temp = _safe_get(dbd, 'sci_water_temp')
    _,  pres = _safe_get(dbd, 'sci_water_pressure')
    _,  cond = _safe_get(dbd, 'sci_water_cond')
    return t, temp, pres, cond


def _extract_gps(dbd):
    t,  lon = _safe_get(dbd, 'm_lon')
    _,  lat = _safe_get(dbd, 'm_lat')
    return t, lon, lat


def _compute_profile(t_sci, temp, pres, cond, t_gps, lon, lat):
    if not (len(pres) > 0 and len(lon) > 0 and len(lat) > 0 and len(t_gps) > 0):
        return None

    depth = pres * 10
    ix = (BINS >= depth.min()) & (BINS <= depth.max())
    if not ix.any():
        return None

    temp_profile = np.full(len(BINS), np.nan)
    temp_profile[ix] = np.interp(BINS[ix], depth, temp)

    cond_mscm = cond * 10
    psal = gsw.SP_from_C(cond_mscm, temp, pres)
    sal_profile = np.full(len(BINS), np.nan)
    sal_profile[ix] = np.interp(BINS[ix], depth, psal)

    return {
        'lat':         float(np.nanmean(lat)),
        'lon':         float(np.nanmean(lon)),
        'time':        pd.Timestamp(float(np.nanmean(t_gps)), unit='s'),
        'temperature': temp_profile,
        'salinity':    sal_profile,
    }


# ── Per-file-type processors ───────────────────────────────────────────────────

def _process_pair(sci_path, gps_path, cac_dir):
    sci_dbd = _open_dbd(sci_path, cac_dir)
    if sci_dbd is None:
        return None
    gps_dbd = _open_dbd(gps_path, cac_dir)
    if gps_dbd is None:
        return None
    try:
        t_sci, temp, pres, cond = _extract_science(sci_dbd)
        t_gps, lon, lat          = _extract_gps(gps_dbd)
        return _compute_profile(t_sci, temp, pres, cond, t_gps, lon, lat)
    except Exception as exc:
        logger.debug("Error processing pair %s + %s: %s", sci_path, gps_path, exc)
        return None


def _process_sbd_standalone(sbd_path, cac_dir):
    dbd = _open_dbd(sbd_path, cac_dir)
    if dbd is None:
        return None
    try:
        t_sci, temp, pres, cond = _extract_science(dbd)
        t_gps, lon, lat          = _extract_gps(dbd)
        return _compute_profile(t_sci, temp, pres, cond, t_gps, lon, lat)
    except Exception as exc:
        logger.debug("Error processing standalone %s: %s", sbd_path, exc)
        return None


# ── Directory-level processor ──────────────────────────────────────────────────

def process_glider(data_dir, output_dir, glider_name, cac_dir):
    """
    Process all supported binary files in data_dir.
    cac_dir must point to a directory containing the .cac cache files
    required by dbdreader.  If a cache file is missing a WARNING is logged
    naming the exact file needed.
    """
    profiles = []
    used_as_companion = set()

    # ── 1. .tcd science files ──────────────────────────────────────────────────
    for tcd in sorted(glob.glob(os.path.join(data_dir, '*.tcd'))):
        base = os.path.splitext(tcd)[0]
        for gps_ext in ('.scd', '.sbd'):
            gps_path = base + gps_ext
            if os.path.exists(gps_path):
                r = _process_pair(tcd, gps_path, cac_dir)
                if r:
                    profiles.append(r)
                used_as_companion.add(os.path.abspath(gps_path))
                break

    # ── 2. .tbd science files ──────────────────────────────────────────────────
    for tbd in sorted(glob.glob(os.path.join(data_dir, '*.tbd'))):
        base     = os.path.splitext(tbd)[0]
        sbd_path = base + '.sbd'
        if os.path.exists(sbd_path):
            r = _process_pair(tbd, sbd_path, cac_dir)
            if r:
                profiles.append(r)
            used_as_companion.add(os.path.abspath(sbd_path))

    # ── 3. Standalone .sbd files ───────────────────────────────────────────────
    for sbd in sorted(glob.glob(os.path.join(data_dir, '*.sbd'))):
        if os.path.abspath(sbd) not in used_as_companion:
            r = _process_sbd_standalone(sbd, cac_dir)
            if r:
                profiles.append(r)

    if not profiles:
        logger.info("No valid profiles from '%s' (check for missing .cac files above)",
                    glider_name)
        return 0

    logger.info("'%s': %d profiles collected", glider_name, len(profiles))
    ds       = _build_dataset(profiles)
    missions = _split_by_time_gap(ds)

    os.makedirs(output_dir, exist_ok=True)
    for i, m in enumerate(missions):
        out_path = os.path.join(output_dir, f'mission_{i:03d}.nc')
        m.attrs['glider']        = glider_name
        m.attrs['mission_index'] = i
        m.to_netcdf(out_path)
        t0 = pd.Timestamp(m['time'].values[0]).strftime('%Y-%m-%d')
        t1 = pd.Timestamp(m['time'].values[-1]).strftime('%Y-%m-%d')
        logger.info("Saved %s mission %d (%s → %s, %d profiles) → %s",
                    glider_name, i, t0, t1, m.dims['time'], out_path)
    return len(missions)


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _build_dataset(profiles):
    profiles.sort(key=lambda p: p['time'])
    ds = xr.Dataset(
        {
            'temperature': (['depth', 'time'], np.array([p['temperature'] for p in profiles]).T),
            'salinity':    (['depth', 'time'], np.array([p['salinity']    for p in profiles]).T),
            'latitude':    ('time', [p['lat'] for p in profiles]),
            'longitude':   ('time', [p['lon'] for p in profiles]),
        },
        coords={
            'time':  np.array([p['time'] for p in profiles]).astype('datetime64[s]'),
            'depth': BINS,
        },
    )
    ds['temperature'].attrs.update({'units': 'degrees_Celsius', 'long_name': 'Sea water temperature'})
    ds['salinity'].attrs.update({'units': 'PSU', 'long_name': 'Practical salinity'})
    ds['depth'].attrs.update({'units': 'm', 'positive': 'down'})
    return ds


def _split_by_time_gap(ds, threshold_days=MISSION_GAP_DAYS):
    times = ds['time'].values
    if len(times) < 2:
        return [ds]
    diffs_s = np.diff(times).astype('timedelta64[s]').astype(float)
    gap_idx  = np.where(diffs_s > threshold_days * 86400)[0]
    if len(gap_idx) == 0:
        return [ds]
    splits = [0] + (gap_idx + 1).tolist() + [len(times)]
    return [ds.isel(time=slice(splits[i], splits[i + 1])).copy() for i in range(len(splits) - 1)]


# ── Top-level entry point ──────────────────────────────────────────────────────

def process_all_gliders(raw_dir, processed_dir, cac_dir):
    """
    Scan raw_dir for Slocum binary data and process every glider found.
    cac_dir must contain the .cac cache files required by dbdreader.
    """
    os.makedirs(cac_dir, exist_ok=True)

    from_glider_dirs = [
        d for d in glob.glob(os.path.join(raw_dir, '**', 'from-glider'), recursive=True)
        if os.path.isdir(d)
    ]

    if not from_glider_dirs:
        all_bin = (
            glob.glob(os.path.join(raw_dir, '**', '*.tcd'), recursive=True)
            + glob.glob(os.path.join(raw_dir, '**', '*.tbd'), recursive=True)
            + glob.glob(os.path.join(raw_dir, '**', '*.sbd'), recursive=True)
        )
        from_glider_dirs = list(set(os.path.dirname(f) for f in all_bin))

    if not from_glider_dirs:
        logger.info("No glider binary data found under %s", raw_dir)
        return 0

    total = 0
    for fgdir in from_glider_dirs:
        parent      = os.path.dirname(fgdir)
        glider_name = (
            os.path.basename(parent)
            if os.path.basename(fgdir) == 'from-glider'
            else os.path.basename(fgdir)
        )
        out_dir = os.path.join(processed_dir, glider_name)
        logger.info("Processing glider '%s' from %s", glider_name, fgdir)
        total += process_glider(fgdir, out_dir, glider_name, cac_dir)

    logger.info("Processing complete: %d mission files written", total)
    return total

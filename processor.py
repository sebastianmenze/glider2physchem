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
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import json
import re
import glob
import logging
import shutil
import tarfile
import tempfile

import numpy as np
import pandas as pd
import xarray as xr
import gsw
import dbdreader
from dbdreader import DBD

from npc_export import generate_profile_npc

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

    # Per-bin science timestamp (seconds since epoch, interpolated onto depth grid)
    time_profile = np.full(len(BINS), np.nan)
    if len(t_sci) > 0:
        time_profile[ix] = np.interp(BINS[ix], depth, t_sci.astype(float))

    # Raw (uninterpolated) measurement points for NPC export
    raw_time = (t_sci.astype(float)
                if len(t_sci) == len(pres)
                else np.full(len(pres), np.nan))

    return {
        'lat':          float(np.nanmean(lat)),
        'lon':          float(np.nanmean(lon)),
        'time':         pd.Timestamp(float(np.nanmean(t_gps)), unit='s'),
        'temperature':  temp_profile,
        'salinity':     sal_profile,
        'time_profile': time_profile,
        'raw_temp':     temp,
        'raw_sal':      psal,
        'raw_pres':     pres,
        'raw_time':     raw_time,
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

def process_glider(data_dir, output_dir, glider_name, cac_dir, platform_ids=None):
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
    safe_name = re.sub(r'[^\w-]', '_', glider_name)

    # Remove old mission_NNN.nc files (legacy format only — safe to do immediately)
    for old in glob.glob(os.path.join(output_dir, 'mission_*.nc')):
        try:
            os.remove(old)
            logger.debug("Removed old-format file: %s", old)
        except OSError:
            pass

    # ── Stage → promote → NPC (three separate passes) ─────────────────────────
    # Keeping them separate means an NPC failure never prevents a mission from
    # being promoted, and a promotion failure never leaves a partial NC on disk.
    staging_dir = os.path.join(output_dir, '.staging')
    os.makedirs(staging_dir, exist_ok=True)

    # Pass 1: write every mission NC to staging (old final files stay visible)
    staged = []   # [(staged_path, final_path, dataset, t0, t1)]
    try:
        for m in missions:
            t0 = pd.Timestamp(m['time'].values[0]).strftime('%Y-%m-%d')
            t1 = pd.Timestamp(m['time'].values[-1]).strftime('%Y-%m-%d')
            nc_name     = f'{safe_name}_{t0}.nc'
            staged_path = os.path.join(staging_dir, nc_name)
            final_path  = os.path.join(output_dir,  nc_name)
            m.attrs['glider'] = glider_name
            m.to_netcdf(staged_path)
            staged.append((staged_path, final_path, m, t0, t1))
    finally:
        # On write failure, clean up staging; old finals are untouched.
        if len(staged) < len(missions):
            shutil.rmtree(staging_dir, ignore_errors=True)
            return len(staged)   # return however many were already on disk

    # Pass 2: promote all staged files atomically — no NPC work here so
    # an HDF error cannot abort a mid-loop promotion.
    promoted = []
    for staged_path, final_path, m, t0, t1 in staged:
        try:
            os.replace(staged_path, final_path)
            _write_mission_sidecar(final_path, m, glider_name)
            logger.info("Saved %s (%s → %s, %d profiles) → %s",
                        glider_name, t0, t1, m.sizes['time'], final_path)
            promoted.append((final_path, m))
        except Exception as exc:
            logger.warning("Could not promote %s: %s — keeping old file", final_path, exc)
    shutil.rmtree(staging_dir, ignore_errors=True)

    # Pass 3: generate NPC files from the newly promoted NC files.
    # Errors here are logged and skipped; they never touch the NC files.
    for final_path, m in promoted:
        npc_dir = final_path[:-3] + '_npc'
        os.makedirs(npc_dir, exist_ok=True)
        n_written = 0
        try:
            with xr.open_dataset(final_path) as npc_ds:
                for pi in range(m.sizes['time']):
                    try:
                        content, npc_name = generate_profile_npc(
                            final_path, pi, _ds=npc_ds,
                            platform_id=(platform_ids or {}).get(glider_name, 0),
                        )
                        with open(os.path.join(npc_dir, npc_name), 'w', encoding='utf-8') as fh:
                            fh.write(content)
                        n_written += 1
                    except Exception as exc:
                        logger.warning("NPC export failed for %s profile %d: %s",
                                       glider_name, pi, exc)
        except Exception as exc:
            logger.warning("NPC generation skipped for %s: %s", final_path, exc)
        logger.info("  → %d NPC files → %s", n_written, npc_dir)

    return len(missions)


# ── Dataset helpers ────────────────────────────────────────────────────────────

def _floats_to_datetime64(arr2d):
    """Convert float seconds-since-epoch array to datetime64[s], NaN → NaT."""
    out = np.full(arr2d.shape, np.datetime64('NaT'), dtype='datetime64[s]')
    valid = ~np.isnan(arr2d)
    out[valid] = arr2d[valid].astype('int64').astype('datetime64[s]')
    return out


def _build_dataset(profiles):
    profiles.sort(key=lambda p: p['time'])
    time_profile_arr = np.array([p['time_profile'] for p in profiles]).T  # [depth, n]
    ds = xr.Dataset(
        {
            'temperature':  (['depth', 'time'], np.array([p['temperature'] for p in profiles]).T),
            'salinity':     (['depth', 'time'], np.array([p['salinity']    for p in profiles]).T),
            'time_profile': (['depth', 'time'], _floats_to_datetime64(time_profile_arr)),
            'latitude':     ('time', [p['lat'] for p in profiles]),
            'longitude':    ('time', [p['lon'] for p in profiles]),
        },
        coords={
            'time':  np.array([p['time'] for p in profiles]).astype('datetime64[s]'),
            'depth': BINS,
        },
    )
    ds['temperature'].attrs.update({'units': 'degrees_Celsius', 'long_name': 'Sea water temperature'})
    ds['salinity'].attrs.update({'units': 'PSU', 'long_name': 'Practical salinity'})
    ds['time_profile'].attrs.update({'long_name': 'CTD measurement timestamp per depth bin'})
    ds['depth'].attrs.update({'units': 'm', 'positive': 'down'})

    # Raw (uninterpolated) measurement points — padded to max observation count
    max_obs = max(len(p['raw_temp']) for p in profiles)
    if max_obs > 0:
        def _pad(arr):
            out = np.full(max_obs, np.nan)
            n = min(len(arr), max_obs)
            out[:n] = arr[:n]
            return out

        ds['raw_temperature'] = xr.DataArray(
            np.array([_pad(p['raw_temp']) for p in profiles]).T,
            dims=['obs', 'time'],
            attrs={'units': 'degrees_Celsius', 'long_name': 'Raw CTD temperature'},
        )
        ds['raw_salinity'] = xr.DataArray(
            np.array([_pad(p['raw_sal']) for p in profiles]).T,
            dims=['obs', 'time'],
            attrs={'units': 'PSU', 'long_name': 'Raw practical salinity'},
        )
        ds['raw_pressure'] = xr.DataArray(
            np.array([_pad(p['raw_pres']) for p in profiles]).T,
            dims=['obs', 'time'],
            attrs={'units': 'dbar', 'long_name': 'Raw CTD pressure'},
        )
        raw_time_float = np.array([_pad(p['raw_time']) for p in profiles]).T
        ds['raw_time_profile'] = xr.DataArray(
            _floats_to_datetime64(raw_time_float),
            dims=['obs', 'time'],
            attrs={'long_name': 'Raw CTD measurement timestamp'},
        )
        ds = ds.assign_coords(obs=np.arange(max_obs))

    return ds


def _write_mission_sidecar(nc_path: str, m: 'xr.Dataset', glider_name: str) -> None:
    """Write a lightweight JSON metadata sidecar alongside an NC file.

    scan_missions() reads these instead of the NC file itself, so the UI
    overview never has to open HDF5 files.
    """
    times = pd.to_datetime(m['time'].values)
    lats  = m['latitude'].values
    lons  = m['longitude'].values
    meta  = {
        'glider':     glider_name,
        'start':      times[0].isoformat(),
        'end':        times[-1].isoformat(),
        'n_profiles': int(len(times)),
        'lats':       [round(float(v), 6) for v in lats],
        'lons':       [round(float(v), 6) for v in lons],
        'center_lat': round(float(np.nanmean(lats)), 6),
        'center_lon': round(float(np.nanmean(lons)), 6),
    }
    json_path = nc_path[:-3] + '.json'
    tmp_path  = json_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh)
    os.replace(tmp_path, json_path)


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

def _process_archived_deployments(glider_raw_dir, glider_name, processed_dir, cac_dir, platform_ids=None):
    """
    Find every .tar.gz under <glider_raw_dir>/.archived-deployments/<deployment>/archive/,
    extract it to a temp dir, process the from-glider binaries inside, and write
    mission NC files to processed/<glider_name>/archived/<deployment_name>/.

    Deployments whose output directory already contains NC files are skipped so
    re-runs don't reprocess the same archive twice.
    """
    archived_base = os.path.join(glider_raw_dir, '.archived-deployments')
    if not os.path.isdir(archived_base):
        return 0

    total = 0
    for deployment_name in sorted(os.listdir(archived_base)):
        deployment_dir = os.path.join(archived_base, deployment_name)
        if not os.path.isdir(deployment_dir):
            continue

        # Collect all tar.gz files: check both archive/ subdir and the deployment dir itself
        archive_subdir = os.path.join(deployment_dir, 'archive')
        tar_files = sorted(
            glob.glob(os.path.join(archive_subdir, '*.tar.gz'))
            + glob.glob(os.path.join(deployment_dir, '*.tar.gz'))
        )
        if not tar_files:
            continue

        safe_deployment = re.sub(r'[^\w.\-]', '-', deployment_name)
        out_dir = os.path.join(processed_dir, glider_name, 'archived', safe_deployment)

        # Skip if already processed and all NC files are readable
        existing_nc = glob.glob(os.path.join(out_dir, '*.nc'))
        if existing_nc:
            all_valid = True
            for nc in existing_nc:
                try:
                    with xr.open_dataset(nc):
                        pass
                except Exception as exc:
                    logger.warning("Corrupt archived NC %s: %s — will reprocess", nc, exc)
                    try:
                        os.remove(nc)
                    except OSError:
                        pass
                    all_valid = False
            if all_valid:
                logger.debug("Archived deployment '%s/%s' already processed — skipping",
                             glider_name, deployment_name)
                total += len(existing_nc)
                continue

        logger.info("Processing archived deployment '%s/%s' (%d archive file(s))",
                    glider_name, deployment_name, len(tar_files))

        tmpdir = tempfile.mkdtemp(prefix='glider_arc_')
        try:
            for tar_path in tar_files:
                try:
                    with tarfile.open(tar_path, 'r:gz') as tar:
                        tar.extractall(tmpdir)
                    logger.debug("Extracted %s", os.path.basename(tar_path))
                except Exception as exc:
                    logger.warning("Cannot extract %s: %s", tar_path, exc)

            # Find from-glider dirs inside the extracted tree
            from_glider_dirs = [
                root for root, _, _ in os.walk(tmpdir)
                if os.path.basename(root) == 'from-glider'
            ]
            if not from_glider_dirs:
                # Fallback: any directory that contains binary files
                from_glider_dirs = list({
                    os.path.dirname(f)
                    for ext in ('*.tcd', '*.tbd', '*.sbd')
                    for f in glob.glob(os.path.join(tmpdir, '**', ext), recursive=True)
                })

            for fgdir in from_glider_dirs:
                total += process_glider(fgdir, out_dir, glider_name, cac_dir, platform_ids)

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return total


def process_all_gliders(raw_dir, processed_dir, cac_dir, platform_ids=None):
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
    seen_glider_dirs: set[str] = set()

    for fgdir in from_glider_dirs:
        parent      = os.path.dirname(fgdir)
        glider_name = (
            os.path.basename(parent)
            if os.path.basename(fgdir) == 'from-glider'
            else os.path.basename(fgdir)
        )
        glider_raw_dir = parent if os.path.basename(fgdir) == 'from-glider' else fgdir
        out_dir = os.path.join(processed_dir, glider_name)
        logger.info("Processing glider '%s' from %s", glider_name, fgdir)
        total += process_glider(fgdir, out_dir, glider_name, cac_dir, platform_ids)
        seen_glider_dirs.add(os.path.abspath(glider_raw_dir))
        total += _process_archived_deployments(glider_raw_dir, glider_name, processed_dir, cac_dir, platform_ids)

    # Also check direct children of raw_dir that have .archived-deployments but
    # no active from-glider directory (e.g. retired gliders).
    for entry in sorted(os.listdir(raw_dir)):
        if entry.startswith('.'):
            continue
        glider_raw_dir = os.path.join(raw_dir, entry)
        if not os.path.isdir(glider_raw_dir):
            continue
        if os.path.abspath(glider_raw_dir) in seen_glider_dirs:
            continue
        if os.path.isdir(os.path.join(glider_raw_dir, '.archived-deployments')):
            total += _process_archived_deployments(glider_raw_dir, entry, processed_dir, cac_dir, platform_ids)

    logger.info("Processing complete: %d mission files written", total)
    return total

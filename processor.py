"""
Slocum glider binary data processor.
Reads paired .tcd / .scd files, derives temperature and salinity
profiles on a 1-m depth grid, and writes per-mission NetCDF files.

A "mission" is a continuous stretch of data; gaps > MISSION_GAP_DAYS
days are treated as separate missions.
"""
import os
import glob
import logging

import numpy as np
import pandas as pd
import xarray as xr
import gsw
from dbdreader import DBD

logger = logging.getLogger(__name__)

# ── Depth-binning parameters ───────────────────────────────────────────────────
BIN_MAX = 200          # metres
BIN_INTERVAL = 1       # metres
_BINS_CALC = np.arange(BIN_INTERVAL / 2, BIN_MAX, BIN_INTERVAL)
BINS = _BINS_CALC[:-1] + BIN_INTERVAL / 2   # bin centres  (198 values)

MISSION_GAP_DAYS = 7


def _process_tcd_pair(tcd_path):
    """
    Process a single .tcd / .scd file pair.
    Returns a dict {lat, lon, time, temperature, salinity} or None on failure.
    """
    scd_path = tcd_path[:-4] + '.scd'
    if not os.path.exists(scd_path):
        return None
    try:
        tbd = DBD(tcd_path)
        time_sci, sci_water_temp     = tbd.get('sci_water_temp')
        _,        sci_water_pressure = tbd.get('sci_water_pressure')
        _,        sci_water_cond     = tbd.get('sci_water_cond')

        sbd = DBD(scd_path)
        gps_time, lon = sbd.get('m_lon')
        _,        lat = sbd.get('m_lat')

        if not (len(sci_water_pressure) > 0 and len(lon) > 0 and len(lat) > 0):
            return None

        depth = sci_water_pressure * 10   # dbar → metres (approx)
        ix = (BINS >= depth.min()) & (BINS <= depth.max())

        # Interpolate temperature onto 1-m bins
        temp_profile = np.full(len(BINS), np.nan)
        temp_profile[ix] = np.interp(BINS[ix], depth, sci_water_temp)

        # Derive practical salinity then interpolate
        cond_mscm = sci_water_cond * 10   # S/m → mS/cm
        psal = gsw.SP_from_C(cond_mscm, sci_water_temp, sci_water_pressure)
        sal_profile = np.full(len(BINS), np.nan)
        sal_profile[ix] = np.interp(BINS[ix], depth, psal)

        return {
            'lat':         float(np.nanmean(lat)),
            'lon':         float(np.nanmean(lon)),
            'time':        pd.Timestamp(float(np.nanmean(gps_time)), unit='s'),
            'temperature': temp_profile,
            'salinity':    sal_profile,
        }
    except Exception as exc:
        logger.debug("Skipping %s: %s", tcd_path, exc)
        return None


def _build_dataset(profiles):
    """Convert a list of profile dicts into an xarray Dataset."""
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
    """Return a list of datasets split wherever consecutive time gaps exceed threshold."""
    times = ds['time'].values
    if len(times) < 2:
        return [ds]
    diffs_s = np.diff(times).astype('timedelta64[s]').astype(float)
    gap_idx  = np.where(diffs_s > threshold_days * 86400)[0]
    if len(gap_idx) == 0:
        return [ds]
    splits = [0] + (gap_idx + 1).tolist() + [len(times)]
    return [ds.isel(time=slice(splits[i], splits[i + 1])).copy() for i in range(len(splits) - 1)]


def process_glider(tcd_dir, output_dir, glider_name):
    """
    Process all .tcd files in tcd_dir, write mission NetCDF files to output_dir.
    Returns the number of mission files written.
    """
    tcd_files = sorted(glob.glob(os.path.join(tcd_dir, '*.tcd')))
    if not tcd_files:
        logger.info("No .tcd files in %s", tcd_dir)
        return 0

    profiles = [r for r in (_process_tcd_pair(f) for f in tcd_files) if r is not None]
    if not profiles:
        logger.warning("No valid profiles from %s", tcd_dir)
        return 0

    ds = _build_dataset(profiles)
    missions = _split_by_time_gap(ds)

    os.makedirs(output_dir, exist_ok=True)
    for i, m in enumerate(missions):
        out_path = os.path.join(output_dir, f'mission_{i:03d}.nc')
        m.attrs['glider'] = glider_name
        m.attrs['mission_index'] = i
        m.to_netcdf(out_path)
        t0 = pd.Timestamp(m['time'].values[0]).strftime('%Y-%m-%d')
        t1 = pd.Timestamp(m['time'].values[-1]).strftime('%Y-%m-%d')
        logger.info("Saved %s mission %d (%s → %s, %d profiles) → %s",
                    glider_name, i, t0, t1, m.dims['time'], out_path)
    return len(missions)


def process_all_gliders(raw_dir, processed_dir):
    """
    Scan raw_dir for Slocum data, process every glider found.

    Expects the mirrored SFMC structure:
        raw_dir/<any_path>/<glider_name>/from-glider/*.tcd

    Falls back to scanning any subdirectory containing .tcd files.
    Returns total number of mission files written.
    """
    # Find all 'from-glider' directories
    from_glider_dirs = [
        d for d in glob.glob(os.path.join(raw_dir, '**', 'from-glider'), recursive=True)
        if os.path.isdir(d)
    ]

    if not from_glider_dirs:
        # Fallback: any directory that contains .tcd files
        tcd_files = glob.glob(os.path.join(raw_dir, '**', '*.tcd'), recursive=True)
        from_glider_dirs = list(set(os.path.dirname(f) for f in tcd_files))

    if not from_glider_dirs:
        logger.info("No glider data found under %s", raw_dir)
        return 0

    total = 0
    for fgdir in from_glider_dirs:
        parent = os.path.dirname(fgdir)
        glider_name = (
            os.path.basename(parent)
            if os.path.basename(fgdir) == 'from-glider'
            else os.path.basename(fgdir)
        )
        out_dir = os.path.join(processed_dir, glider_name)
        logger.info("Processing glider '%s' from %s", glider_name, fgdir)
        total += process_glider(fgdir, out_dir, glider_name)

    logger.info("Processing complete: %d mission files written", total)
    return total

"""
Generate PhysChem NPC files from processed glider mission NetCDF data.

NPC format (from NPCFileHandler in rbr2physchem_prod.py):
  # Metadata:
  key:\tvalue
  ...
  % Readings:
  <tab-separated CSV>
"""
import uuid
import numpy as np
import pandas as pd
import xarray as xr


def generate_profile_npc(nc_path: str, profile_idx: int,
                         _ds: 'xr.Dataset | None' = None,
                         platform_id: int = 0) -> tuple[str, str]:
    """
    Build the NPC file content for one profile from a mission NC file.

    Pass an already-open dataset as ``_ds`` to avoid re-opening the file
    (useful when generating all profiles in a batch).

    Returns (npc_text, suggested_filename).
    """
    _opened = _ds is None
    if _opened:
        _ds = xr.open_dataset(nc_path)
    try:
        times        = pd.to_datetime(_ds['time'].values)
        lats         = _ds['latitude'].values
        lons         = _ds['longitude'].values
        depth        = _ds['depth'].values
        temp         = _ds['temperature'].values    # [depth, time]
        sal          = _ds['salinity'].values       # [depth, time]
        time_profile = _ds['time_profile'].values if 'time_profile' in _ds else None
        glider       = str(_ds.attrs.get('glider', 'glider'))

        pi    = max(0, min(profile_idx, len(times) - 1))
        t_pro = times[pi]
        lat_p = float(lats[pi])
        lon_p = float(lons[pi])

        year   = int(t_pro.year)
        t0_str = times[0].strftime('%Y-%m-%dT%H:%M:%SZ')
        t1_candidate = times[0] + pd.DateOffset(years=1)
        t1_str = (times[-1] if times[-1] > t1_candidate else t1_candidate).strftime('%Y-%m-%dT%H:%M:%SZ')
        t_str  = t_pro.strftime('%Y-%m-%dT%H:%M:%SZ')

        # Prefer raw timestamps for operation start/end; fall back to binned bins
        _has_raw = all(v in _ds for v in ('raw_temperature', 'raw_salinity',
                                          'raw_pressure', 'raw_time_profile'))
        if _has_raw:
            raw_time_col = pd.to_datetime(_ds['raw_time_profile'].values[:, pi])
            raw_valid_mask = ~pd.isna(raw_time_col)
            raw_times_valid = raw_time_col[raw_valid_mask]
            if len(raw_times_valid):
                t_str_start = raw_times_valid.min().strftime('%Y-%m-%dT%H:%M:%SZ')
                t_str_end   = raw_times_valid.max().strftime('%Y-%m-%dT%H:%M:%SZ')
            else:
                t_str_start = t_str_end = t_str
        elif time_profile is not None:
            t_bins_pi    = pd.to_datetime(time_profile[:, pi])
            t_bins_valid = t_bins_pi[~pd.isna(t_bins_pi)]
            if len(t_bins_valid):
                t_str_start = t_bins_valid.min().strftime('%Y-%m-%dT%H:%M:%SZ')
                t_str_end   = t_bins_valid.max().strftime('%Y-%m-%dT%H:%M:%SZ')
            else:
                t_str_start = t_str_end = t_str
        else:
            t_str_start = t_str_end = t_str

        # ── Mission metadata ───────────────────────────────────────────────────
        meta = {}
        meta['mission.missionNumber']         = int(times[0].strftime('%Y%m%d'))
        meta['mission.missionStartDate']      = t0_str
        meta['mission.missionStopDate']       = t1_str
        meta['mission.missionType']           = '14'
        meta['mission.platform']              = platform_id
        meta['mission.startYear']             = year
        meta['mission.platformName']          = glider
        meta['mission.missionTypeName']       = 'Glider mission'
        meta['mission.purpose']               = 'Along transect glider survey'
        meta['mission.missionName']           = f'{glider} {times[0].strftime("%Y-%m-%d")}'
        meta['mission.responsibleLaboratory'] = 3

        # ── Operation metadata ─────────────────────────────────────────────────
        meta['operation.operationType']        = 'CTD'
        meta['operation.operationNumber']      = pi + 1
        meta['operation.operationPlatform']    = platform_id
        meta['operation.timeStart']            = t_str_start
        meta['operation.timeEnd']              = t_str_end
        meta['operation.timeStartQuality']     = 0
        meta['operation.timeEndQuality']       = 0
        meta['operation.featureType']          = 4
        meta['operation.latitudeStart']        = round(lat_p, 6)
        meta['operation.longitudeStart']       = round(lon_p, 6)
        meta['operation.positionStartQuality'] = 0
        meta['operation.stationType']          = 1000
        meta['operation.localCdiId']           = str(uuid.uuid1())

        # ── Parameter metadata ─────────────────────────────────────────────────
        params = [
            ('DATETIME', 'ISO8601 UTC', 'Date and time',        'DateTime'),
            ('TEMP',     'degC',        'Temperature of water',  'Temperature'),
            ('PSAL',     'PSU',         'Practical salinity',    'Salinity'),
            ('PRES',     'dbar',        'Sea Pressure',          'Sea Pressure'),
            ('DEPTH',    'm',           'Depth below sea level', 'Depth'),
        ]
        for i, (code, units, name, supplied) in enumerate(params, 1):
            meta[f'parameter{{{i}}}.parameterCode']         = code
            meta[f'parameter{{{i}}}.units']                 = units
            meta[f'parameter{{{i}}}.suppliedUnits']         = units
            meta[f'parameter{{{i}}}.parameterName']         = name
            meta[f'parameter{{{i}}}.suppliedParameterName'] = supplied
            meta[f'parameter{{{i}}}.acquirementMethod']     = '1019900'
            meta[f'parameter{{{i}}}.processingLevel']       = 'L0'

        meta['instrument.instrumentNumber']                    = 1
        meta['instrument.instrumentType']                      = 'CTD'
        meta['instrument.instrumentDataOwner']                 = 3
        meta['instrument.instrumentProperty.profileDirection'] = 'D'

        # ── Data table ─────────────────────────────────────────────────────────
        if _has_raw:
            raw_temp_col = _ds['raw_temperature'].values[:, pi]
            raw_sal_col  = _ds['raw_salinity'].values[:, pi]
            raw_pres_col = _ds['raw_pressure'].values[:, pi]
            raw_time_col = pd.to_datetime(_ds['raw_time_profile'].values[:, pi])

            valid = ~np.isnan(raw_temp_col) & ~np.isnan(raw_sal_col)
            n = int(valid.sum())

            raw_depth_col = raw_pres_col * 10.0

            bin_times = []
            for t in raw_time_col[valid]:
                try:
                    ts = pd.Timestamp(t)
                    bin_times.append(ts.strftime('%Y-%m-%dT%H:%M:%SZ')
                                     if not pd.isna(ts) else t_str)
                except Exception:
                    bin_times.append(t_str)

            df = pd.DataFrame({
                'sampleNumber':        np.arange(1, n + 1),
                'DATETIME.value':      bin_times,
                'DATETIME.sampleSize': 1,
                'TEMP.value':          np.round(raw_temp_col[valid], 4),
                'TEMP.std':            0.0,
                'TEMP.sampleSize':     1,
                'PSAL.value':          np.round(raw_sal_col[valid], 4),
                'PSAL.std':            0.0,
                'PSAL.sampleSize':     1,
                'PRES.value':          np.round(raw_pres_col[valid], 4),
                'PRES.std':            0.0,
                'PRES.sampleSize':     1,
                'DEPTH.value':         np.round(raw_depth_col[valid], 2),
            })
        else:
            # Fallback: use 1 m-binned interpolated data (old NC files)
            temp_p = temp[:, pi]
            sal_p  = sal[:, pi]
            pres_p = depth / 10.0

            valid = ~np.isnan(temp_p) & ~np.isnan(sal_p)
            n = int(valid.sum())

            if time_profile is not None:
                t_bins    = time_profile[:, pi]
                bin_times = []
                for t in t_bins[valid]:
                    try:
                        ts = pd.Timestamp(t)
                        bin_times.append(ts.strftime('%Y-%m-%dT%H:%M:%SZ')
                                         if not pd.isna(ts) else t_str)
                    except Exception:
                        bin_times.append(t_str)
            else:
                bin_times = [t_str] * n

            df = pd.DataFrame({
                'sampleNumber':        np.arange(1, n + 1),
                'DATETIME.value':      bin_times,
                'DATETIME.sampleSize': 1,
                'TEMP.value':          np.round(temp_p[valid], 4),
                'TEMP.std':            0.0,
                'TEMP.sampleSize':     1,
                'PSAL.value':          np.round(sal_p[valid], 4),
                'PSAL.std':            0.0,
                'PSAL.sampleSize':     1,
                'PRES.value':          np.round(pres_p[valid], 4),
                'PRES.std':            0.0,
                'PRES.sampleSize':     1,
                'DEPTH.value':         depth[valid].astype(int),
            })

        # ── Serialise ──────────────────────────────────────────────────────────
        lines = ['# Metadata:\n']
        for key, value in meta.items():
            lines.append(f'{key}:\t{value}\n')
        lines.append('% Readings:\n')
        lines.append(df.to_csv(sep='\t', index=False, lineterminator='\n'))

        content  = ''.join(lines)
        filename = (f'{glider}_profile_{pi + 1:04d}_'
                    f'{t_pro.strftime("%Y%m%dT%H%M")}.npc')
        return content, filename
    finally:
        if _opened:
            _ds.close()

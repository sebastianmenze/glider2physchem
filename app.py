"""
Slocum Glider Monitor — Dash web application.

Provides:
  • All Missions tab  — overview map of every mission track + mission cards
  • Mission Explorer  — interactive 3D salinity/temperature surfaces,
                        track map, vertical profile viewer, time filter,
                        profile-step slider

A background APScheduler job:
  1. SSHes into the SFMC server and mirrors new binary files to data/raw/
  2. Reprocesses all gliders, writing per-mission NetCDF to data/processed/

Configuration is read from a .env file (see .env.example).
"""

import os
# Must be set before any HDF5/NetCDF4 C library call.
# Docker volumes (overlayfs / virtiofs) don't support POSIX flock(); HDF5
# raises [Errno -101] NC_EHDFERR when it tries to lock. Safe to disable
# because the staging-then-rename write strategy ensures no concurrent writers
# on any final NC file.
os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

import glob
import json
import logging
import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
import utm
import dash
import dash_bootstrap_components as dbc
from apscheduler.schedulers.background import BackgroundScheduler
from dash import Input, Output, State, callback_context, dcc, html, no_update
from dotenv import load_dotenv

from downloader import sync_glider_data
from processor import process_all_gliders
from npc_export import generate_profile_npc
from physchem_upload import sync_all_npc_to_physchem, load_upload_record

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.environ.get('DATA_DIR', os.path.join(_HERE, 'data'))
RAW_DIR       = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
CAC_DIR       = os.environ.get('CAC_DIR', os.path.join(DATA_DIR, 'cac'))
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CAC_DIR, exist_ok=True)

SFMC_HOST      = os.environ.get('SFMC_HOST', 'sfmc.webbresearch.com')
SFMC_USER      = os.environ.get('SFMC_USER', '')
SFMC_PASSWORD  = os.environ.get('SFMC_PASSWORD', '') or None
SFMC_KEY       = os.environ.get('SFMC_KEY', '') or None
SYNC_INTERVAL  = int(os.environ.get('SYNC_INTERVAL_MINUTES', '30'))

PHYSCHEM_BASE_URL      = os.environ.get('PHYSCHEM_BASE_URL', '')
PHYSCHEM_EDITOR_URL    = os.environ.get('PHYSCHEM_EDITOR_URL', 'https://physchem-editor.hi.no')
AWS_ACCESS_KEY_ID      = os.environ.get('AWS_ACCESS_KEY_ID', '')
AWS_SECRET_ACCESS_KEY  = os.environ.get('AWS_SECRET_ACCESS_KEY', '')

SFMC_BASE_PATH = os.environ.get(
    'SFMC_BASE_PATH',
    '/var/opt/sfmc-dockserver/stations/bergen/gliders',
)
GLIDER_NAMES = [
    g.strip()
    for g in os.environ.get('GLIDER_NAMES', 'var,fulla').split(',')
    if g.strip()
]

# Parse GLIDER_PLATFORM_ID={name:id,...} — e.g. {var:666,fulla:667}
def _parse_platform_ids(raw: str) -> dict:
    ids = {}
    for item in raw.strip().strip('{}').split(','):
        if ':' in item:
            k, _, v = item.partition(':')
            try:
                ids[k.strip()] = int(v.strip())
            except ValueError:
                pass
    return ids

GLIDER_PLATFORM_IDS: dict = _parse_platform_ids(
    os.environ.get('GLIDER_PLATFORM_ID', '')
)

# Full remote paths: explicit list overrides the base+names construction
_raw_paths = (
    os.environ.get('SFMC_REMOTE_PATHS')
    or os.environ.get('SFMC_REMOTE_PATH', '')
)
SFMC_REMOTE_PATHS = [p.strip() for p in _raw_paths.split(',') if p.strip()]
if not SFMC_REMOTE_PATHS:
    SFMC_REMOTE_PATHS = [f"{SFMC_BASE_PATH}/{g}" for g in GLIDER_NAMES]

GLIDER_PALETTE = [
    '#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0',
    '#00BCD4', '#FF5722', '#8BC34A', '#FFC107', '#607D8B',
]

# ── Global sync state ──────────────────────────────────────────────────────────
_last_sync_time: datetime | None = None
_sync_lock = threading.Lock()
_is_syncing = False


def _glider_color(glider_name: str, all_gliders: list[str]) -> str:
    idx = sorted(all_gliders).index(glider_name) if glider_name in all_gliders else 0
    return GLIDER_PALETTE[idx % len(GLIDER_PALETTE)]


# ── Data helpers ───────────────────────────────────────────────────────────────

_mission_cache: list[dict] = []


def scan_missions() -> list[dict]:
    """Return lightweight metadata for every processed mission.

    Reads the JSON sidecar written by the processor alongside each NC file.
    This avoids opening HDF5 files in the callback thread, preventing
    contention with the background sync thread.
    """
    import json as _json
    global _mission_cache
    missions = []
    json_files = sorted(
        f for f in glob.glob(os.path.join(PROCESSED_DIR, '**', '*.json'), recursive=True)
        if '.staging' not in f.replace('\\', '/').split('/')
        and os.path.basename(f) != 'physchem_uploaded.json'
    )
    for json_path in json_files:
        nc_path = json_path[:-5] + '.nc'
        if not os.path.exists(nc_path):
            continue   # NC file deleted or not yet promoted
        try:
            with open(json_path, 'r', encoding='utf-8') as fh:
                meta = _json.load(fh)

            rel   = os.path.relpath(nc_path, PROCESSED_DIR)
            parts = rel.replace('\\', '/').split('/')
            glider      = meta.get('glider', parts[0])
            is_archived = len(parts) >= 4 and parts[1] == 'archived'
            deployment  = parts[2] if is_archived else ''
            mission     = os.path.splitext(parts[-1])[0]

            start_str  = meta['start']
            end_str    = meta['end']
            start_date = start_str[:10]
            last_ts    = pd.Timestamp(end_str).to_pydatetime().replace(tzinfo=timezone.utc)
            age_h      = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600

            missions.append({
                'path':        nc_path,
                'glider':      glider,
                'mission':     mission,
                'deployment':  deployment,
                'label':       f"{glider}  {start_date}",
                'start':       start_date,
                'end':         end_str[:10],
                'n_profiles':  int(meta['n_profiles']),
                'center_lat':  float(meta['center_lat']),
                'center_lon':  float(meta['center_lon']),
                'lats':        meta['lats'],
                'lons':        meta['lons'],
                'is_active':   age_h < 48,
                'is_archived': is_archived,
                'age_h':       age_h,
            })
        except Exception as exc:
            logger.warning("Cannot read mission sidecar %s: %s — skipping", json_path, exc)

    # Deduplicate: same glider + same start date → keep the one with most profiles
    seen: dict = {}
    for m in missions:
        key = (m['glider'], m['start'])
        if key not in seen or m['n_profiles'] > seen[key]['n_profiles']:
            seen[key] = m
    result = sorted(seen.values(), key=lambda m: (m['glider'], m['start']))
    if result:
        _mission_cache = result
    elif _mission_cache:
        logger.warning("scan_missions returned empty — serving cached result (%d missions)",
                       len(_mission_cache))
        return _mission_cache
    return result


def load_mission(nc_path: str) -> dict:
    """Load full profile data for the Mission Explorer."""
    with xr.open_dataset(nc_path, engine='h5netcdf') as ds:
        lat   = ds['latitude'].values
        lon   = ds['longitude'].values
        depth = ds['depth'].values
        times = pd.to_datetime(ds['time'].values)
        temp  = ds['temperature'].values
        sal   = ds['salinity'].values

    try:
        east, north, _, _ = utm.from_latlon(lat, lon)
    except Exception:
        east  = (lon - lon.mean()) * 111_000 * np.cos(np.radians(lat.mean()))
        north = (lat - lat.mean()) * 111_000

    return dict(lat=lat, lon=lon, depth=depth, times=times,
                temp=temp, sal=sal, east=east, north=north)


def _build_surface(val, east, north, depth):
    """Create (X, Y, Z, V) arrays for a Plotly Surface trace."""
    X = np.repeat(east[np.newaxis, :],  len(depth), axis=0)
    Y = np.repeat(north[np.newaxis, :], len(depth), axis=0)
    Z = -np.repeat(depth[:, np.newaxis], len(east), axis=1)
    V = val
    valid = ~np.all(np.isnan(V), axis=1)
    return X[valid], Y[valid], Z[valid], V[valid]


def _bbox_zoom(lats, lons) -> int:
    """Estimate a Plotly map zoom level that fits the lat/lon bounding box."""
    lat_range = float(np.max(lats) - np.min(lats))
    lon_range = float(np.max(lons) - np.min(lons))
    max_extent = max(lat_range, lon_range, 0.001)
    return max(2, min(13, int(np.log2(360.0 / max_extent) - 1)))


def _dark_fig(**kwargs) -> go.Figure:
    fig = go.Figure(**kwargs)
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(25,25,25,1)',
        font=dict(color='#ccc'),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


# ── Sync ───────────────────────────────────────────────────────────────────────

def run_sync():
    global _last_sync_time, _is_syncing
    if _is_syncing:
        return
    if not _sync_lock.acquire(blocking=False):
        return  # another thread won the race
    try:
        _is_syncing = True
        try:
            logger.info("Starting data sync…")
            if SFMC_USER:
                for remote_path in SFMC_REMOTE_PATHS:
                    local_subdir = os.path.basename(remote_path.rstrip('/'))
                    logger.info("Syncing %s → raw/%s", remote_path, local_subdir)
                    sync_glider_data(
                        hostname=SFMC_HOST,
                        username=SFMC_USER,
                        password=SFMC_PASSWORD,
                        key_filename=SFMC_KEY,
                        remote_path=remote_path,
                        local_path=os.path.join(RAW_DIR, local_subdir),
                    )
            process_all_gliders(RAW_DIR, PROCESSED_DIR, CAC_DIR, GLIDER_PLATFORM_IDS)
            sync_all_npc_to_physchem(
                PROCESSED_DIR,
                PHYSCHEM_BASE_URL,
                AWS_ACCESS_KEY_ID,
                AWS_SECRET_ACCESS_KEY,
            )
            _last_sync_time = datetime.now(timezone.utc)
            logger.info("Sync complete at %s", _last_sync_time)
        except Exception as exc:
            logger.error("Sync error: %s", exc)
        finally:
            _is_syncing = False
    finally:
        _sync_lock.release()


# ── Dash app layout ────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title='IMR Glider Dashboard',
    suppress_callback_exceptions=True,
)
server = app.server  # expose Flask server for gunicorn

_GRAPH_STYLE = {'height': '44vh'}
_DROPDOWN_STYLE = {'color': '#111'}

app.layout = dbc.Container(
    [
        # ── hidden state / timers ──────────────────────────────────────────────
        dcc.Interval(id='auto-refresh', interval=5 * 60 * 1000, n_intervals=0),
        dcc.Store(id='store-nav-path',      data=None),
        dcc.Store(id='store-date-bounds',   data=None),
        dcc.Store(id='store-profile-times', data=None),
        dcc.Download(id='download-npc'),

        # ── navbar ────────────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(html.H4('IMR Glider Dashboard', className='mb-0'), width='auto'),
                dbc.Col(
                    [
                        html.Span(id='sync-label',
                                  className='text-muted small',
                                  children='Not synced yet'),
                    ],
                    className='d-flex align-items-center justify-content-end',
                ),
            ],
            className='py-3 border-bottom mb-3 align-items-center',
        ),

        # ── tabs with fixed component structure ───────────────────────────────
        dbc.Tabs(
            [
                # ═══ All Missions tab ══════════════════════════════════════════
                dbc.Tab(
                    [
                        dcc.Graph(id='overview-map', config={'displayModeBar': False},
                                  style={'height': '42vh'}),
                        html.Hr(className='my-2'),
                        html.Div(id='mission-cards'),
                    ],
                    label='All Missions',
                    tab_id='tab-overview',
                ),

                # ═══ Mission Explorer tab ══════════════════════════════════════
                dbc.Tab(
                    [
                        # ── controls row 1: mission + date range + reset ──
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label('Mission', className='mb-1 small'),
                                        dcc.Dropdown(id='mission-select',
                                                     clearable=False,
                                                     style=_DROPDOWN_STYLE),
                                    ],
                                    width=12, md=6,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label('From', className='mb-1 small'),
                                        dcc.Dropdown(id='date-start', clearable=True,
                                                     placeholder='start date',
                                                     style=_DROPDOWN_STYLE),
                                    ],
                                    width=5, md=2,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label('To', className='mb-1 small'),
                                        dcc.Dropdown(id='date-end', clearable=True,
                                                     placeholder='end date',
                                                     style=_DROPDOWN_STYLE),
                                    ],
                                    width=5, md=2,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label(' ', className='mb-1 small d-block'),
                                        dbc.Button('Reset', id='btn-reset-dates',
                                                   color='outline-secondary', size='sm',
                                                   className='w-100'),
                                    ],
                                    width=2, md=2,
                                ),
                            ],
                            className='mb-1 mt-2 g-2',
                        ),
                        # ── controls row 2: slider + profile info ──────────
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Slider(
                                        id='profile-slider',
                                        min=0, max=0, value=0, step=1,
                                        marks=None,
                                    ),
                                    width=12, md=8,
                                    className='d-flex align-items-center',
                                ),
                                dbc.Col(
                                    html.Div(id='profile-info',
                                             className='small text-light'),
                                    width=12, md=4,
                                    className='d-flex align-items-center',
                                ),
                            ],
                            className='mb-3 align-items-center',
                        ),
                        # ── plots grid ────────────────────────────────────
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id='plot-sal', style=_GRAPH_STYLE,
                                              config={'displayModeBar': False}),
                                    width=12, lg=6, className='mb-2',
                                ),
                                dbc.Col(
                                    dcc.Graph(id='plot-temp', style=_GRAPH_STYLE,
                                              config={'displayModeBar': False}),
                                    width=12, lg=6, className='mb-2',
                                ),
                            ]
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dcc.Graph(id='plot-map', style=_GRAPH_STYLE,
                                              config={'displayModeBar': False}),
                                    width=12, lg=6, className='mb-2',
                                ),
                                dbc.Col(
                                    [
                                        dcc.Graph(id='plot-profile', style=_GRAPH_STYLE,
                                                  config={'displayModeBar': False}),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Button(
                                                        '⬇ Download NPC',
                                                        id='btn-download-npc',
                                                        color='outline-info',
                                                        size='sm',
                                                        className='w-100',
                                                    ),
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        'View in PhysChem',
                                                        id='btn-physchem-link',
                                                        href='',
                                                        target='_blank',
                                                        color='outline-success',
                                                        size='sm',
                                                        className='w-100',
                                                        disabled=True,
                                                    ),
                                                    width=6,
                                                ),
                                            ],
                                            className='mt-1 g-1',
                                        ),
                                    ],
                                    width=12, lg=6, className='mb-2',
                                ),
                            ]
                        ),
                    ],
                    label='Mission Explorer',
                    tab_id='tab-explorer',
                ),
            ],
            id='main-tabs',
            active_tab='tab-overview',
        ),
    ],
    fluid=True,
)


# ── Callbacks ──────────────────────────────────────────────────────────────────

# ── 1. Overview tab content ────────────────────────────────────────────────────

@app.callback(
    Output('overview-map',   'figure'),
    Output('mission-cards',  'children'),
    Output('sync-label',     'children'),
    Output('mission-select', 'options'),
    Output('mission-select', 'value'),
    Input('auto-refresh',    'n_intervals'),
    Input('store-nav-path',  'data'),
    State('mission-select',  'value'),
)
def refresh_overview(_, nav_path, current_select):
    missions   = scan_missions()
    sync_label = (
        f"Last sync: {_last_sync_time.strftime('%Y-%m-%d %H:%M UTC')}"
        if _last_sync_time else 'Never synced'
    )

    # Build dropdown options for explorer tab
    options = [
        {'label': m['label'] + (' [ACTIVE]' if m['is_active'] else ''), 'value': m['path']}
        for m in missions
    ]
    # Preserve current selection or default to newest
    new_select = nav_path or current_select
    if new_select not in [m['path'] for m in missions]:
        new_select = missions[-1]['path'] if missions else None

    # ── overview map ──────────────────────────────────────────────────────────
    if not missions:
        empty_fig = _dark_fig()
        empty_fig.update_layout(
            title=dict(text='No mission data yet — awaiting sync', font=dict(size=16))
        )
        no_data_msg = dbc.Alert(
            [
                html.H5('No processed data yet'),
                html.P('Data will appear after the first scheduled sync, '
                       'or place .tcd/.scd files in data/raw/ and restart.'),
            ],
            color='info', className='mt-3',
        )
        return empty_fig, no_data_msg, sync_label, [], None

    all_gliders = sorted(set(m['glider'] for m in missions))
    map_fig = _dark_fig()
    for glider in all_gliders:
        color = _glider_color(glider, all_gliders)
        for m in [x for x in missions if x['glider'] == glider]:
            if not m['lats']:
                continue
            opacity = 1.0 if m['is_active'] else 0.45
            map_fig.add_trace(go.Scattermap(
                lat=m['lats'], lon=m['lons'],
                mode='lines+markers',
                line=dict(width=2, color=color),
                marker=dict(size=4, color=color, opacity=opacity),
                name=m['label'],
                opacity=opacity,
                hovertemplate=(
                    f'<b>{m["label"]}</b><br>'
                    '%{lat:.4f}°N  %{lon:.4f}°E<extra></extra>'
                ),
            ))

    all_lats = [v for m in missions for v in m['lats']]
    all_lons = [v for m in missions for v in m['lons']]
    clat = float(np.nanmean(all_lats)) if all_lats else float(np.mean([m['center_lat'] for m in missions]))
    clon = float(np.nanmean(all_lons)) if all_lons else float(np.mean([m['center_lon'] for m in missions]))
    zoom = _bbox_zoom(all_lats, all_lons) if all_lats else 5
    map_fig.update_layout(
        map=dict(style='open-street-map',
                 center=dict(lat=clat, lon=clon),
                 zoom=zoom),
        legend=dict(bgcolor='rgba(40,40,40,0.8)', font=dict(color='white')),
        showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # ── mission cards ─────────────────────────────────────────────────────────
    def _card(m):
        if m['is_active']:
            badge = dbc.Badge('ACTIVE', color='success', className='me-2')
        else:
            badge = dbc.Badge('completed', color='secondary', className='me-2')
        color = _glider_color(m['glider'], all_gliders)
        border_style = {'borderLeft': f'4px solid {color}'}
        return dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(
                        [badge, html.Strong(m['glider']),
                         html.Span(f"  {m['start']}", className='text-muted ms-1 small')],
                        style=border_style,
                    ),
                    dbc.CardBody(
                        [
                            html.P(f"{m['start']} → {m['end']}", className='mb-1 small'),
                            html.P(f"{m['n_profiles']} profiles",
                                   className='mb-2 small text-muted'),
                            dbc.Button(
                                'Explore →',
                                id={'type': 'explore-btn', 'path': m['path']},
                                color='outline-primary', size='sm',
                            ),
                        ]
                    ),
                ],
                color='dark', outline=True,
            ),
            width=12, md=6, lg=4, xl=3, className='mb-2',
        )

    active   = [m for m in missions if m['is_active']]
    archived = [m for m in missions if not m['is_active']]

    header = html.H6(
        f"{len(missions)} mission(s) across {len(all_gliders)} glider(s)"
        + (f" • {len(active)} active" if active else ''),
        className='mb-2 mt-1 text-muted',
    )
    cards = dbc.Row([_card(m) for m in (active + archived)])

    return map_fig, html.Div([header, cards]), sync_label, options, new_select


# ── 2. Navigate from card "Explore →" button ──────────────────────────────────

@app.callback(
    Output('store-nav-path', 'data'),
    Output('main-tabs',      'active_tab'),
    Input({'type': 'explore-btn', 'path': dash.ALL}, 'n_clicks'),
    State({'type': 'explore-btn', 'path': dash.ALL}, 'id'),
    prevent_initial_call=True,
)
def on_explore_click(n_clicks_list, ids):
    ctx = callback_context
    if not ctx.triggered:
        return no_update, no_update
    raw = ctx.triggered[0]['prop_id'].rsplit('.', 1)[0]
    try:
        path = json.loads(raw)['path']
        return path, 'tab-explorer'
    except Exception:
        return no_update, no_update


# ── 4. Update mission controls when mission changes ───────────────────────────

@app.callback(
    Output('date-start',        'value'),
    Output('date-end',          'value'),
    Output('profile-slider',    'max'),
    Output('profile-slider',    'value'),
    Output('store-date-bounds', 'data'),
    Input('mission-select',     'value'),
)
def update_controls(nc_path):
    none5 = (None, None, 0, 0, None)
    if not nc_path or not os.path.exists(nc_path):
        return none5
    try:
        with xr.open_dataset(nc_path, engine='h5netcdf') as ds:
            times = pd.to_datetime(ds['time'].values)
        d0 = times[0].strftime('%Y-%m-%d')
        d1 = times[-1].strftime('%Y-%m-%d')
        return d0, d1, len(times) - 1, 0, {'start': d0, 'end': d1}
    except Exception:
        return none5


# ── 5. Reset date range ───────────────────────────────────────────────────────

@app.callback(
    Output('date-start', 'value', allow_duplicate=True),
    Output('date-end',   'value', allow_duplicate=True),
    Input('btn-reset-dates',    'n_clicks'),
    State('store-date-bounds',  'data'),
    prevent_initial_call=True,
)
def on_reset_dates(_, bounds):
    if bounds:
        return bounds['start'], bounds['end']
    return no_update, no_update


# ── 6. Populate date dropdown options from mission data ───────────────────────

@app.callback(
    Output('date-start', 'options'),
    Output('date-end',   'options'),
    Input('mission-select', 'value'),
)
def update_date_options(nc_path):
    if not nc_path or not os.path.exists(nc_path):
        return [], []
    try:
        with xr.open_dataset(nc_path, engine='h5netcdf') as ds:
            times = pd.to_datetime(ds['time'].values)
        dates = sorted({t.strftime('%Y-%m-%d') for t in times})
        options = [{'label': d, 'value': d} for d in dates]
        return options, options
    except Exception:
        return [], []


@app.callback(
    Output('plot-sal',     'figure'),
    Output('plot-temp',    'figure'),
    Output('plot-map',     'figure'),
    Output('plot-profile', 'figure'),
    Output('profile-info', 'children'),
    Input('mission-select',  'value'),
    Input('date-start',      'value'),
    Input('date-end',        'value'),
    Input('profile-slider',  'value'),
)
def update_plots(nc_path, start_date, end_date, slider_idx):
    empty = _dark_fig()
    empty5 = empty, empty, empty, empty, ''

    if not nc_path or not os.path.exists(nc_path):
        return empty5

    try:
        d = load_mission(nc_path)
    except Exception as exc:
        return empty, empty, empty, empty, f'Error: {exc}'

    lat, lon   = d['lat'], d['lon']
    depth      = d['depth']
    times      = d['times']
    temp, sal  = d['temp'], d['sal']
    east, north = d['east'], d['north']

    # ── apply date filter ─────────────────────────────────────────────────────
    mask = np.ones(len(times), dtype=bool)
    if start_date:
        mask &= times >= pd.Timestamp(start_date)
    if end_date:
        mask &= times <= pd.Timestamp(end_date) + pd.Timedelta(days=1)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return empty, empty, empty, empty, 'No profiles in selected date range'

    lat_f  = lat[idx];   lon_f  = lon[idx]
    east_f = east[idx];  north_f = north[idx]
    times_f = times[idx]
    temp_f = temp[:, idx]
    sal_f  = sal[:, idx]

    pi = min(slider_idx or 0, len(idx) - 1)   # profile index within filtered data

    # ── 3D surface helper ─────────────────────────────────────────────────────
    def surface_fig(val, colorscale, title, unit):
        X, Y, Z, V = _build_surface(val, east_f, north_f, depth)
        xr_ = X.max() - X.min() if X.size else 1
        yr_ = Y.max() - Y.min() if Y.size else 1
        mr  = max(xr_, yr_, 1)

        fig = _dark_fig()
        if X.size:
            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z, surfacecolor=V,
                colorscale=colorscale,
                colorbar=dict(title=dict(text=unit, font=dict(color='#ccc')),
                              len=0.6, x=1.01,
                              tickfont=dict(color='#ccc')),
            ))
            # red vertical line at selected profile
            xp, yp = east_f[pi], north_f[pi]
            fig.add_trace(go.Scatter3d(
                x=[xp] * len(depth), y=[yp] * len(depth), z=(-depth).tolist(),
                mode='lines', line=dict(color='red', width=6),
                showlegend=False,
            ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=13, color='#ddd')),
            scene=dict(
                xaxis=dict(title='East (m)',  color='#888', gridcolor='#333',
                           backgroundcolor='rgba(20,20,20,0.8)'),
                yaxis=dict(title='North (m)', color='#888', gridcolor='#333',
                           backgroundcolor='rgba(20,20,20,0.8)'),
                zaxis=dict(title='Depth (m)', color='#888', gridcolor='#333',
                           backgroundcolor='rgba(20,20,20,0.8)'),
                aspectratio=dict(x=xr_/mr * 2, y=yr_/mr * 2, z=1),
                bgcolor='rgba(20,20,20,1)',
            ),
            margin=dict(l=0, r=60, b=0, t=35),
        )
        return fig

    sal_fig  = surface_fig(sal_f,  'Viridis', 'Salinity',    'PSU')
    temp_fig = surface_fig(temp_f, 'Plasma',  'Temperature', '°C')

    # ── track map — full mission track, selected profile highlighted ──────────
    map_fig = _dark_fig()
    map_fig.add_trace(go.Scattermap(
        lat=lat.tolist(), lon=lon.tolist(),
        mode='lines+markers',
        line=dict(width=2, color='rgba(160,160,160,0.35)'),
        marker=dict(size=4, color=list(range(len(lat))),
                    colorscale='Viridis', opacity=0.7),
        hovertext=[t.strftime('%Y-%m-%d %H:%M') for t in times],
        hoverinfo='text+lat+lon',
        name='Track',
    ))
    map_fig.add_trace(go.Scattermap(
        lat=[lat_f[pi]], lon=[lon_f[pi]],
        mode='markers',
        marker=dict(size=14, color='red'),
        hovertext=times_f[pi].strftime('%Y-%m-%d %H:%M'),
        hoverinfo='text+lat+lon',
        name='Selected',
    ))
    map_fig.update_layout(
        title=dict(text='Track Map', font=dict(size=13, color='#ddd')),
        uirevision=nc_path,   # preserve zoom/pan across slider moves; resets only when mission changes
        map=dict(
            style='open-street-map',
            center=dict(lat=float(np.mean(lat)), lon=float(np.mean(lon))),
            zoom=_bbox_zoom(lat, lon),
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=35, b=0),
    )

    # ── vertical profile ──────────────────────────────────────────────────────
    abs_pi    = idx[pi]   # index into the full (unfiltered) arrays
    prof_temp = temp[:, abs_pi]
    prof_sal  = sal[:, abs_pi]
    vt = ~np.isnan(prof_temp)
    vs = ~np.isnan(prof_sal)

    prof_fig = _dark_fig()
    if vs.any():
        prof_fig.add_trace(go.Scatter(
            x=prof_sal[vs], y=-depth[vs],
            mode='lines+markers', name='Salinity (PSU)',
            line=dict(color='#2196F3', width=3), marker=dict(size=4),
        ))
    if vt.any():
        prof_fig.add_trace(go.Scatter(
            x=prof_temp[vt], y=-depth[vt],
            mode='lines+markers', name='Temp (°C)',
            line=dict(color='#FF5722', width=3), marker=dict(size=4),
            xaxis='x2',
        ))
    prof_fig.update_layout(
        title=dict(
            text=f"Profile — {times_f[pi].strftime('%Y-%m-%d %H:%M')}",
            font=dict(size=13, color='#ddd'),
        ),
        xaxis=dict(title='Salinity (PSU)', color='#2196F3',
                   gridcolor='#333', side='bottom'),
        xaxis2=dict(title='Temp (°C)', color='#FF5722',
                    overlaying='x', side='top', showgrid=False),
        yaxis=dict(title='Depth (m)', color='#aaa', gridcolor='#333'),
        legend=dict(bgcolor='rgba(40,40,40,0.8)'),
        margin=dict(l=55, r=55, t=45, b=35),
    )

    info = (
        f"Profile {pi + 1} / {len(idx)}  •  "
        f"{times_f[pi].strftime('%Y-%m-%d %H:%M')}  •  "
        f"{lat_f[pi]:.4f}°N  {lon_f[pi]:.4f}°E"
    )
    return sal_fig, temp_fig, map_fig, prof_fig, info


# ── Profile times store (populated when mission changes) ──────────────────────

@app.callback(
    Output('store-profile-times', 'data'),
    Input('mission-select', 'value'),
)
def update_profile_times(nc_path):
    if not nc_path or not os.path.exists(nc_path):
        return None
    try:
        with xr.open_dataset(nc_path, engine='h5netcdf') as ds:
            glider = str(ds.attrs.get('glider', ''))
            times  = [pd.Timestamp(t).isoformat() for t in ds['time'].values]
        return {'glider': glider, 'times': times}
    except Exception:
        return None


# ── PhysChem editor link ───────────────────────────────────────────────────────

@app.callback(
    Output('btn-physchem-link', 'href'),
    Output('btn-physchem-link', 'disabled'),
    Input('store-profile-times', 'data'),
    Input('profile-slider',      'value'),
)
def update_physchem_link(profile_data, slider_idx):
    if not profile_data or slider_idx is None:
        return '', True
    glider = profile_data.get('glider', '')
    times  = profile_data.get('times', [])
    if not times:
        return '', True
    pi    = min(int(slider_idx), len(times) - 1)
    t_pro = pd.Timestamp(times[pi])
    fname = f'{glider}_profile_{pi + 1:04d}_{t_pro.strftime("%Y%m%dT%H%M")}.npc'
    record = load_upload_record(PROCESSED_DIR)
    entry  = record.get(fname)
    if isinstance(entry, dict):
        mid = entry.get('mission_id')
        oid = entry.get('operation_id')
        if mid and oid:
            url = (f'{PHYSCHEM_EDITOR_URL}/mission/{mid}'
                   f'/operation/{oid}/instrument')
            return url, False
    return '', True


# ── NPC download ──────────────────────────────────────────────────────────────

@app.callback(
    Output('download-npc',      'data'),
    Input('btn-download-npc',   'n_clicks'),
    State('mission-select',     'value'),
    State('profile-slider',     'value'),
    prevent_initial_call=True,
)
def download_npc(_, nc_path, slider_idx):
    if not nc_path or not os.path.exists(nc_path):
        return no_update
    try:
        glider = os.path.relpath(nc_path, PROCESSED_DIR).replace('\\', '/').split('/')[0]
        content, filename = generate_profile_npc(
            nc_path, slider_idx or 0,
            platform_id=GLIDER_PLATFORM_IDS.get(glider, 0),
        )
        return dcc.send_string(content, filename)
    except Exception as exc:
        logger.error("NPC export failed: %s", exc)
        return no_update


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    # Delay the first scheduled run by one full interval so it doesn't overlap
    # with the immediate startup sync fired below.
    first_run = datetime.now(timezone.utc) + timedelta(minutes=SYNC_INTERVAL)
    scheduler.add_job(run_sync, 'interval', minutes=SYNC_INTERVAL, id='sync',
                      next_run_time=first_run)
    scheduler.start()
    logger.info("Scheduler started: sync every %d minutes (first at %s)",
                SYNC_INTERVAL, first_run.strftime('%H:%M UTC'))

    # Run one sync immediately so data is fresh on startup
    threading.Thread(target=run_sync, daemon=True).start()

    app.run(
        host=os.environ.get('APP_HOST', '0.0.0.0'),
        port=int(os.environ.get('APP_PORT', '8090')),
        debug=os.environ.get('APP_DEBUG', 'false').lower() == 'true',
    )

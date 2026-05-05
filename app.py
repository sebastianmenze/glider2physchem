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

import glob
import json
import logging
import os
import threading
from datetime import datetime, timezone

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
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

SFMC_HOST        = os.environ.get('SFMC_HOST', 'sfmc.webbresearch.com')
SFMC_USER        = os.environ.get('SFMC_USER', '')
SFMC_PASSWORD    = os.environ.get('SFMC_PASSWORD', '') or None
SFMC_KEY         = os.environ.get('SFMC_KEY', '') or None
SFMC_REMOTE_PATH = os.environ.get('SFMC_REMOTE_PATH',
                                   '/var/opt/sfmc-dockserver/stations/bergen/gliders/var')
SYNC_INTERVAL    = int(os.environ.get('SYNC_INTERVAL_MINUTES', '30'))

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

def scan_missions() -> list[dict]:
    """Return lightweight metadata for every processed mission NC file."""
    missions = []
    nc_files = sorted(
        glob.glob(os.path.join(PROCESSED_DIR, '**', '*.nc'), recursive=True)
    )
    for nc_path in nc_files:
        try:
            with xr.open_dataset(nc_path) as ds:
                times = pd.to_datetime(ds['time'].values)
                lats  = ds['latitude'].values
                lons  = ds['longitude'].values

            glider   = os.path.basename(os.path.dirname(nc_path))
            mission  = os.path.splitext(os.path.basename(nc_path))[0]
            last_ts  = times[-1].to_pydatetime().replace(tzinfo=timezone.utc)
            age_h    = (datetime.now(timezone.utc) - last_ts).total_seconds() / 3600

            missions.append({
                'path':       nc_path,
                'glider':     glider,
                'mission':    mission,
                'label':      f"{glider} / {mission}",
                'start':      times[0].strftime('%Y-%m-%d'),
                'end':        times[-1].strftime('%Y-%m-%d'),
                'n_profiles': len(times),
                'center_lat': float(np.nanmean(lats)),
                'center_lon': float(np.nanmean(lons)),
                'lats':       lats.tolist(),
                'lons':       lons.tolist(),
                'is_active':  age_h < 48,
                'age_h':      age_h,
            })
        except Exception as exc:
            logger.warning("Cannot read %s: %s", nc_path, exc)
    return missions


def load_mission(nc_path: str) -> dict:
    """Load full profile data for the Mission Explorer."""
    with xr.open_dataset(nc_path) as ds:
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
    with _sync_lock:
        _is_syncing = True
        try:
            logger.info("Starting data sync…")
            if SFMC_USER:
                sync_glider_data(
                    hostname=SFMC_HOST,
                    username=SFMC_USER,
                    password=SFMC_PASSWORD,
                    key_filename=SFMC_KEY,
                    remote_path=SFMC_REMOTE_PATH,
                    local_path=os.path.join(RAW_DIR, 'var'),
                )
            process_all_gliders(RAW_DIR, PROCESSED_DIR)
            _last_sync_time = datetime.now(timezone.utc)
            logger.info("Sync complete at %s", _last_sync_time)
        except Exception as exc:
            logger.error("Sync error: %s", exc)
        finally:
            _is_syncing = False


# ── Dash app layout ────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title='Slocum Glider Monitor',
    suppress_callback_exceptions=True,
)
server = app.server  # expose Flask server for gunicorn

_GRAPH_STYLE = {'height': '44vh'}
_DROPDOWN_STYLE = {'color': '#111', 'backgroundColor': '#fff'}

app.layout = dbc.Container(
    [
        # ── hidden state / timers ──────────────────────────────────────────────
        dcc.Interval(id='auto-refresh', interval=5 * 60 * 1000, n_intervals=0),
        dcc.Store(id='store-nav-path', data=None),

        # ── navbar ────────────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(html.H4('🌊 Slocum Glider Monitor', className='mb-0'), width='auto'),
                dbc.Col(
                    [
                        html.Span(id='sync-label',
                                  className='text-muted small me-3',
                                  children='Not synced yet'),
                        dbc.Button('⟳ Sync Now', id='btn-sync',
                                   color='outline-light', size='sm'),
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
                        # ── controls row ──────────────────────────────────────
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        dbc.Label('Mission', className='mb-1 small'),
                                        dcc.Dropdown(id='mission-select',
                                                     clearable=False,
                                                     style=_DROPDOWN_STYLE),
                                    ],
                                    width=12, md=5,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label('Date Range', className='mb-1 small'),
                                        dcc.DatePickerRange(
                                            id='date-range',
                                            display_format='DD/MM/YYYY',
                                            className='w-100',
                                        ),
                                    ],
                                    width=12, md=4,
                                ),
                                dbc.Col(
                                    [
                                        dbc.Label('Profile', className='mb-1 small'),
                                        html.Div(id='profile-info',
                                                 className='small text-muted mt-1'),
                                    ],
                                    width=12, md=3,
                                ),
                            ],
                            className='mb-2 mt-2 g-2',
                        ),
                        dbc.Row(
                            dbc.Col(
                                dcc.Slider(
                                    id='profile-slider',
                                    min=0, max=0, value=0, step=1,
                                    marks=None,
                                    tooltip={'placement': 'bottom',
                                             'always_visible': True},
                                ),
                                width=12,
                            ),
                            className='mb-3',
                        ),
                        # ── plots grid ────────────────────────────────────────
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
                                    dcc.Graph(id='plot-profile', style=_GRAPH_STYLE,
                                              config={'displayModeBar': False}),
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
            title=dict(text='No mission data — click Sync Now', font=dict(size=16))
        )
        no_data_msg = dbc.Alert(
            [
                html.H5('No processed data yet'),
                html.P('Click "Sync Now" to download from SFMC, '
                       'or place .tcd/.scd files in data/raw/ and sync again.'),
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
            map_fig.add_trace(go.Scattermapbox(
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

    clat = np.mean([m['center_lat'] for m in missions])
    clon = np.mean([m['center_lon'] for m in missions])
    map_fig.update_layout(
        mapbox=dict(style='open-street-map',
                    center=dict(lat=float(clat), lon=float(clon)),
                    zoom=7),
        legend=dict(bgcolor='rgba(40,40,40,0.8)', font=dict(color='white')),
        showlegend=True,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # ── mission cards ─────────────────────────────────────────────────────────
    def _card(m):
        badge = (
            dbc.Badge('ACTIVE', color='success', className='me-2')
            if m['is_active'] else
            dbc.Badge('completed', color='secondary', className='me-2')
        )
        color = _glider_color(m['glider'], all_gliders)
        border_style = {'borderLeft': f'4px solid {color}'}
        return dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader(
                        [badge, html.Strong(m['glider']),
                         html.Span(f" / {m['mission']}", className='text-muted ms-1 small')],
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
    raw = ctx.triggered[0]['prop_id'].split('.')[0]
    try:
        path = json.loads(raw)['path']
        return path, 'tab-explorer'
    except Exception:
        return no_update, no_update


# ── 3. Sync button ────────────────────────────────────────────────────────────

@app.callback(
    Output('btn-sync', 'disabled'),
    Input('btn-sync',  'n_clicks'),
    prevent_initial_call=True,
)
def trigger_sync(n):
    if n:
        threading.Thread(target=run_sync, daemon=True).start()
    return False


# ── 4. Update mission controls when mission changes ───────────────────────────

@app.callback(
    Output('date-range',      'min_date_allowed'),
    Output('date-range',      'max_date_allowed'),
    Output('date-range',      'start_date'),
    Output('date-range',      'end_date'),
    Output('profile-slider',  'max'),
    Output('profile-slider',  'value'),
    Input('mission-select',   'value'),
)
def update_controls(nc_path):
    none6 = (None,) * 4 + (0, 0)
    if not nc_path or not os.path.exists(nc_path):
        return none6
    try:
        with xr.open_dataset(nc_path) as ds:
            times = pd.to_datetime(ds['time'].values)
        d0, d1 = times[0].date(), times[-1].date()
        return d0, d1, d0, d1, len(times) - 1, 0
    except Exception:
        return none6


# ── 5. Update all explorer plots ──────────────────────────────────────────────

@app.callback(
    Output('plot-sal',     'figure'),
    Output('plot-temp',    'figure'),
    Output('plot-map',     'figure'),
    Output('plot-profile', 'figure'),
    Output('profile-info', 'children'),
    Input('mission-select',  'value'),
    Input('date-range',      'start_date'),
    Input('date-range',      'end_date'),
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
                colorbar=dict(title=unit, len=0.6, x=1.01,
                              tickfont=dict(color='#ccc'),
                              titlefont=dict(color='#ccc')),
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

    # ── track map ─────────────────────────────────────────────────────────────
    n = len(lat_f)
    map_fig = _dark_fig()
    map_fig.add_trace(go.Scattermapbox(
        lat=lat_f.tolist(), lon=lon_f.tolist(),
        mode='lines+markers',
        line=dict(width=2, color='rgba(160,160,160,0.4)'),
        marker=dict(size=5, color=list(range(n)), colorscale='Viridis', opacity=0.85),
        hovertext=[t.strftime('%Y-%m-%d %H:%M') for t in times_f],
        hoverinfo='text+lat+lon',
        name='Track',
    ))
    map_fig.add_trace(go.Scattermapbox(
        lat=[lat_f[pi]], lon=[lon_f[pi]],
        mode='markers',
        marker=dict(size=14, color='red'),
        hovertext=times_f[pi].strftime('%Y-%m-%d %H:%M'),
        hoverinfo='text+lat+lon',
        name='Selected',
    ))
    map_fig.update_layout(
        title=dict(text='Track Map', font=dict(size=13, color='#ddd')),
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=float(np.mean(lat_f)), lon=float(np.mean(lon_f))),
            zoom=8,
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


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(run_sync, 'interval', minutes=SYNC_INTERVAL, id='sync')
    scheduler.start()
    logger.info("Scheduler started: sync every %d minutes", SYNC_INTERVAL)

    # Initial sync in background so the app starts immediately
    threading.Thread(target=run_sync, daemon=True).start()

    app.run(
        host=os.environ.get('APP_HOST', '0.0.0.0'),
        port=int(os.environ.get('APP_PORT', '8090')),
        debug=os.environ.get('APP_DEBUG', 'false').lower() == 'true',
    )

# glider2physchem

Automated pipeline for processing Slocum glider binary data and uploading profiles to the [PhysChem](https://physchem-editor.hi.no) ocean database. Includes a Dash web dashboard for interactive data exploration.

## Features

- Syncs binary glider files (`.tcd`/`.tbd`/`.sbd`) from an SFMC server via SSH/SFTP
- Processes raw binaries with [dbdreader](https://github.com/smerckel/dbdreader) into per-mission NetCDF files
- Exports raw CTD profiles as NPC files and uploads them to PhysChem
- Dash web app with interactive salinity/temperature surfaces, track map, and vertical profile viewer
- Fully containerised with Docker; scheduled sync every N minutes

## Requirements

- Docker and Docker Compose
- `.cac` cache files for your glider (required by dbdreader)
- SFMC server credentials (optional — data can also be placed manually in `data/raw/`)
- PhysChem S3 credentials (optional — upload can be disabled)

## Quick start

```bash
cp .env.example .env
# Edit .env with your credentials
docker compose up -d
```

The web app is available at `http://localhost:8090`.

## Configuration

All settings are in `.env`. Key variables:

| Variable | Description |
|---|---|
| `SFMC_HOST` | SFMC server hostname |
| `SFMC_USER` | SFMC SSH username |
| `SFMC_PASSWORD` | SFMC SSH password (or use `SFMC_KEY`) |
| `GLIDER_NAMES` | Comma-separated glider names, e.g. `var,fulla` |
| `SFMC_BASE_PATH` | Base path on the SFMC server |
| `SYNC_INTERVAL_MINUTES` | How often to sync and reprocess (default `30`) |
| `GLIDER_PLATFORM_ID` | PhysChem platform IDs, e.g. `{var:666,fulla:667}` |
| `PHYSCHEM_BASE_URL` | PhysChem API base URL |
| `PHYSCHEM_EDITOR_URL` | PhysChem editor base URL |
| `AWS_ACCESS_KEY_ID` | S3 access key for PhysChem upload |
| `AWS_SECRET_ACCESS_KEY` | S3 secret key for PhysChem upload |
| `DATA_DIR` | Data directory inside the container (default `/app/data`) |
| `CAC_DIR` | Directory containing `.cac` cache files |
| `APP_PORT` | Web app port (default `8090`) |

## Data layout

```
data/
  raw/
    <glider_name>/
      from-glider/        # binary .tcd/.tbd/.sbd files
      .archived-deployments/
        <deployment>/
          archive/        # .tar.gz archives of past deployments
  cac/                    # .cac cache files (one per glider firmware version)
  processed/
    <glider_name>/
      <glider_name>_<date>.nc       # per-mission NetCDF
      <glider_name>_<date>.json     # lightweight sidecar (used by dashboard)
      <glider_name>_<date>_npc/     # per-profile NPC files for PhysChem
      archived/
        <deployment>/
          ...
    physchem_uploaded.json          # upload record (tracks confirmed PhysChem IDs)
```

## Optional: SFTP ingestion server

To let gliders push files directly to this host instead of pulling from SFMC:

```bash
docker compose -f docker-compose.yml -f docker-compose.sftp.yml up -d
python sftp_server_info.py   # prints connection details
```

## Missing .cac files

If the log reports `Missing cache file: XXXXXXXX.cac`, copy that file into `data/cac/` and restart.

## Development

```bash
pip install -r requirements.txt
cp .env.example .env
python app.py
```

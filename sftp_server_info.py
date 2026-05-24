#!/usr/bin/env python3
"""
Print SFTP server connection details.

Run from the project directory:
    python sftp_server_info.py

Reads credentials from .env (or environment variables).
"""
import os
import socket
from pathlib import Path


def _load_dotenv(path='.env'):
    """Minimal .env loader — no dependency on python-dotenv."""
    cfg = {}
    p = Path(path)
    if not p.exists():
        return cfg
    for line in p.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, val = line.partition('=')
        val = val.strip().strip('"').strip("'")
        cfg[key.strip()] = val
    return cfg


def _get(cfg, key, default=''):
    return os.environ.get(key) or cfg.get(key) or default


def _local_ips():
    ips = set()
    try:
        # Connect to a public address (no traffic sent) to find the preferred interface
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('8.8.8.8', 80))
            ips.add(s.getsockname()[0])
    except Exception:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith('127.'):
                ips.add(ip)
    except Exception:
        pass
    return sorted(ips)


cfg = _load_dotenv()

user     = _get(cfg, 'SFTP_USER',        'glider')
password = _get(cfg, 'SFTP_PASSWORD',    '(not set — add SFTP_PASSWORD to .env)')
port     = _get(cfg, 'SFTP_LISTEN_PORT', '2222')
hostname = socket.gethostname()
ips      = _local_ips()

W = 56
print()
print('─' * W)
print('  Glider SFTP Server — connection details')
print('─' * W)
print(f'  Username  : {user}')
print(f'  Password  : {password}')
print(f'  Port      : {port}')
print(f'  Hostname  : {hostname}')
for ip in ips:
    print(f'  IP        : {ip}')
print('─' * W)
print()
print('  Connect from glider system:')
ip_hint = ips[0] if ips else '<host-ip>'
print(f'    sftp -P {port} {user}@{ip_hint}')
print()
print('  Expected upload path layout:')
print('    /from-glider/<file>.tbd')
print()
print('  Example session:')
print(f'    sftp -P {port} {user}@{ip_hint}')
print(f'    sftp> mkdir from-glider')
print(f'    sftp> put *.tbd from-glider/')
print()
print('  Files land in:  ./data/raw/alseamar/from-glider/')
print('  Processed on next sync (automatic every 30 min, or Sync Now).')
print('─' * W)
print()

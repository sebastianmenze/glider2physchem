"""
SFMC SFTP downloader.
Downloads glider data from a Webb Research SFMC server,
only fetching files that are new or changed (rsync-like behaviour).
"""
import os
import stat
import logging

import paramiko

logger = logging.getLogger(__name__)


class SSHDirectoryDownloader:
    def __init__(self, hostname, username, password=None, key_filename=None, port=22):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.key_filename = key_filename
        self.port = port
        self._ssh = None
        self._sftp = None

    def connect(self):
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs = dict(hostname=self.hostname, username=self.username, port=self.port)
        if self.key_filename:
            connect_kwargs['key_filename'] = self.key_filename
        else:
            connect_kwargs['password'] = self.password
        self._ssh.connect(**connect_kwargs)
        self._sftp = self._ssh.open_sftp()
        logger.info("Connected to %s", self.hostname)

    def disconnect(self):
        if self._sftp:
            self._sftp.close()
        if self._ssh:
            self._ssh.close()
        logger.info("Disconnected from %s", self.hostname)

    def _needs_download(self, remote_path, local_path):
        if not os.path.exists(local_path):
            return True
        try:
            rs = self._sftp.stat(remote_path)
            ls = os.stat(local_path)
            return rs.st_size != ls.st_size or rs.st_mtime > ls.st_mtime
        except Exception:
            return True

    def _download_file(self, remote_path, local_path):
        if not self._needs_download(remote_path, local_path):
            return False
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self._sftp.get(remote_path, local_path)
        rs = self._sftp.stat(remote_path)
        os.utime(local_path, (rs.st_atime, rs.st_mtime))
        logger.debug("Downloaded %s", remote_path)
        return True

    def download_directory(self, remote_dir, local_dir):
        """Recursively mirror remote_dir to local_dir. Returns count of new files."""
        os.makedirs(local_dir, exist_ok=True)
        count = 0
        try:
            items = self._sftp.listdir_attr(remote_dir)
        except Exception as exc:
            logger.error("Cannot list %s: %s", remote_dir, exc)
            return 0
        for item in items:
            rpath = f"{remote_dir.rstrip('/')}/{item.filename}"
            lpath = os.path.join(local_dir, item.filename)
            if stat.S_ISDIR(item.st_mode):
                count += self.download_directory(rpath, lpath)
            else:
                try:
                    if self._download_file(rpath, lpath):
                        count += 1
                except Exception as exc:
                    logger.error("Failed to download %s: %s", rpath, exc)
        return count


def sync_glider_data(hostname, username, remote_path, local_path,
                     password=None, key_filename=None, port=22):
    """
    Download new/updated glider data from an SFMC server.
    Returns the number of files newly downloaded.
    """
    dl = SSHDirectoryDownloader(
        hostname=hostname, username=username,
        password=password, key_filename=key_filename, port=port,
    )
    try:
        dl.connect()
        count = dl.download_directory(remote_path, local_path)
        logger.info("Sync complete: %d new/updated files", count)
        return count
    except Exception as exc:
        logger.error("Sync failed: %s", exc)
        return 0
    finally:
        dl.disconnect()

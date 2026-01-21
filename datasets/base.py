"""
Reusable download/extract helpers for dataset classes.

Policy
- Prefer external tools (curl/wget) for downloading when available.
- If external tools are not available, the caller must pass
  ``by_internal_downloader=True`` to use an internal requests-based downloader.
  Optional HTTP headers can be supplied via ``headers``.

Derived datasets can subclass ``DownloadMixin`` (no Dataset inheritance) to
reuse ``download_file`` and ``extract_archive``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import typing as T
from pathlib import Path


class DownloadError(RuntimeError):
    pass


class DownloadMixin:
    """Shared download/extract utilities for dataset classes."""

    @staticmethod
    def _which_tool() -> T.Optional[str]:
        """Return 'curl' or 'wget' if available, else None."""
        for tool in ("curl", "wget"):
            if shutil.which(tool):
                return tool
        return None

    @staticmethod
    def download_file(
        url: str,
        output: T.Union[str, Path],
        *,
        by_internal_downloader: bool = False,
        headers: T.Optional[T.Dict[str, str]] = None,
        tool_preference: T.Optional[str] = None,
        quiet: bool = False,
        chunk_size: int = 1024 * 1024,
    ) -> Path:
        """Download a file to ``output``.

        Prefers external tools (curl/wget). If none found, requires
        ``by_internal_downloader=True`` to use requests-based fallback.
        """
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tool = DownloadMixin._which_tool() if tool_preference is None else tool_preference

        if tool in {"curl", "wget"} and not by_internal_downloader:
            if tool == "curl":
                cmd = [
                    "curl",
                    "-L",  # follow redirects
                    "-f",  # fail on HTTP errors
                    "-o", str(output_path),
                ]
                if quiet:
                    cmd.append("-sS")
                if headers:
                    for k, v in headers.items():
                        cmd.extend(["-H", f"{k}: {v}"])
                cmd.append(url)
            else:  # wget
                cmd = [
                    "wget",
                    "-O", str(output_path),
                ]
                if quiet:
                    cmd.append("-q")
                if headers:
                    for k, v in headers.items():
                        cmd.extend(["--header", f"{k}: {v}"])
                cmd.append(url)

            try:
                subprocess.run(cmd, check=True)
                return output_path
            except subprocess.CalledProcessError as e:
                raise DownloadError(f"External download failed with {tool}: {e}")

        # Internal downloader path requires explicit opt-in
        if not by_internal_downloader:
            raise DownloadError(
                "No external downloader (curl/wget) found. "
                "Pass by_internal_downloader=True to use requests-based download."
            )

        # lazy import to avoid hard dependency when not used
        try:
            import requests
        except Exception as exc:
            raise DownloadError(
                "requests is required for internal download. Install it or use curl/wget."
            ) from exc

        with requests.get(url, stream=True, headers=headers or {}) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
        return output_path

    @staticmethod
    def extract_archive(archive_path: T.Union[str, Path], dest_dir: T.Union[str, Path]) -> Path:
        """Extract common archive formats into ``dest_dir`` and return dest Path."""
        import zipfile
        import tarfile

        archive_path = Path(archive_path)
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)

        lower = archive_path.name.lower()
        if lower.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(dest_dir)
            return dest_dir
        if lower.endswith((".tar.gz", ".tgz", ".tar", ".tar.xz", ".tar.bz2")):
            mode = "r:gz" if lower.endswith((".tar.gz", ".tgz")) else "r"
            with tarfile.open(archive_path, mode) as tf:
                tf.extractall(dest_dir)
            return dest_dir
        # Fallback: unsupported formats
        raise DownloadError(f"Unsupported archive format: {archive_path}")


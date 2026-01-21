import io
import os
import sys
import tarfile
import zipfile
import types
import tempfile
import unittest
from unittest import mock

from datasets.base import DownloadMixin, DownloadError


class TestDownloadMixin(unittest.TestCase):

    @mock.patch("shutil.which", return_value="/usr/bin/curl")
    @mock.patch("subprocess.run")
    def test_external_curl_usage(self, m_run, m_which):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "file.bin")
            DownloadMixin.download_file("http://example.com/file", out)
            self.assertTrue(m_run.called)
            args = m_run.call_args[0][0]
            self.assertIn("curl", args[0])
            self.assertIn("-o", args)
            self.assertIn(out, args)

    @mock.patch("shutil.which", return_value=None)
    def test_internal_requires_opt_in(self, _):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "file.bin")
            with self.assertRaises(DownloadError):
                DownloadMixin.download_file("http://example.com/file", out)

    @mock.patch("shutil.which", return_value=None)
    def test_internal_requests_download(self, _):
        # Inject a fake requests into sys.modules
        fake_requests = types.SimpleNamespace()

        class Resp:
            def __init__(self):
                self.status_code = 200
            def raise_for_status(self):
                return None
            def iter_content(self, chunk_size=8192):
                yield b"hello"
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False

        def fake_get(url, stream=True, headers=None):
            return Resp()

        fake_requests.get = fake_get
        sys.modules['requests'] = fake_requests
        try:
            with tempfile.TemporaryDirectory() as td:
                out = os.path.join(td, "file.bin")
                DownloadMixin.download_file(
                    "http://example.com/file", out, by_internal_downloader=True
                )
                self.assertTrue(os.path.exists(out))
                with open(out, 'rb') as f:
                    self.assertEqual(f.read(), b"hello")
        finally:
            del sys.modules['requests']

    def test_extract_archive_zip(self):
        with tempfile.TemporaryDirectory() as td:
            zip_path = os.path.join(td, "archive.zip")
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("a.txt", "content")
            out_dir = os.path.join(td, "out")
            DownloadMixin.extract_archive(zip_path, out_dir)
            self.assertTrue(os.path.exists(os.path.join(out_dir, "a.txt")))


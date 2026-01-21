import os
import sys
import tempfile
import types
import unittest
from unittest import mock

from datasets.jvs import JVSDataset


class TestJVSDatasetDownload(unittest.TestCase):

    def test_download_requires_gdown(self):
        # Ensure gdown not present
        if 'gdown' in sys.modules:
            del sys.modules['gdown']
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ImportError):
                JVSDataset.download(td)

    def test_download_with_gdown(self):
        # Provide a fake gdown that writes a minimal zip
        import zipfile

        def fake_download(url, output, quiet=False):
            with zipfile.ZipFile(output, 'w') as zf:
                zf.writestr('jvs001/dummy.txt', 'ok')
            return output

        fake = types.SimpleNamespace(download=fake_download)
        sys.modules['gdown'] = fake
        try:
            with tempfile.TemporaryDirectory() as td:
                JVSDataset.download(td)
                self.assertTrue(os.path.exists(os.path.join(td, 'jvs001')))
        finally:
            del sys.modules['gdown']


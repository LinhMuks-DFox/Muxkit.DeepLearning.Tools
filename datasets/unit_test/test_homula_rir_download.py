import os
import tempfile
import zipfile
import unittest
from unittest import mock

from datasets.homula_rir import HomulaRIR


class TestHomulaDownload(unittest.TestCase):

    @mock.patch("datasets.homula_rir.HomulaRIR.download_file")
    def test_download_and_extract(self, m_download_file):
        with tempfile.TemporaryDirectory() as td:
            # Arrange: create a dummy zip in the expected output path
            zip_path = os.path.join(td, HomulaRIR.ZIP_NAME)
            m_download_file.side_effect = lambda url, out, **kwargs: self._make_zip(out)

            # Act
            HomulaRIR.download(td, by_internal_downloader=True)

            # Assert extraction
            self.assertTrue(os.path.isdir(td))

    def _make_zip(self, output):
        with zipfile.ZipFile(output, 'w') as zf:
            zf.writestr('hom/row1/pos-R1-HOM1.csv', '0,0,0')
            zf.writestr('ula/pos-ULA.csv', '0,0,0')
        return output


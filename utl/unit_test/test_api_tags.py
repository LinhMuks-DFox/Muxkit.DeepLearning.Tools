import unittest
import warnings

from utl import api_tags


class TestApiTags(unittest.TestCase):

    def test_deprecated_emits_warning(self):
        @api_tags.deprecated
        def f():
            return 1
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(f(), 1)
            self.assertTrue(
                any(issubclass(i.category, DeprecationWarning) for i in w))

    def test_untested_emits_warning(self):
        @api_tags.untested
        def f():
            return 2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(f(), 2)
            self.assertTrue(
                any(issubclass(i.category, api_tags.UntestedAPIWarning) for i in w))

    def test_bug_api_raises(self):
        @api_tags.bug_api
        def f():
            pass
        with self.assertRaises(NotImplementedError):
            f()

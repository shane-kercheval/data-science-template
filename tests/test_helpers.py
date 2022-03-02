import time
import unittest

from source.helpers.utilities import Timer, get_logger


class TestHelpers(unittest.TestCase):

    def test__open_dict_like_file(self):
        self.assertTrue(True)

    def test__timer(self):
        with Timer("scoring", get_logger()):
            time.sleep(0.1)

        with Timer("scoring", get_logger()) as timer:
            time.sleep(0.1)

        self.assertIsNotNone(timer.interval)

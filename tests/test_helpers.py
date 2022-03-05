import time
import unittest
import pandas as pd
import source.helpers.classification_search_space as css
from source.helpers.utilities import Timer, get_logger
from tests.helpers import get_test_path


class TestHelpers(unittest.TestCase):

    @staticmethod
    def to_string(obj):
        return str(obj). \
            replace(", '", ",\n'"). \
            replace('{', '{\n'). \
            replace('}', '\n}'). \
            replace(', ({', ',\n({')

    def test__open_dict_like_file(self):
        self.assertTrue(True)

    def test__timer(self):
        with Timer("scoring", get_logger()):
            time.sleep(0.1)

        with Timer("scoring", get_logger()) as timer:
            time.sleep(0.1)

        self.assertIsNotNone(timer.interval)

    def test__classification_search_space(self):
        pipeline = css.create_pipeline(pd.DataFrame({
            'a': [1, 2],
            'b': [True, False],
            'c': ['Value1', 'Value2'],
        }))
        # this test mainly makes sure that the function runs, and outputs the result to a file so we can
        # track changes in github
        self.assertIsNotNone(pipeline)
        with open(get_test_path('classification_create_pipeline.txt'), 'w') as file:
            file.write(TestHelpers.to_string(pipeline))

        search_space = css.create_search_space(iterations=23, random_state=99)
        # this test mainly makes sure that the function runs, and outputs the result to a file so we can
        # track changes in github
        self.assertIsNotNone(search_space)
        with open(get_test_path('classification_create_search_space.txt'), 'w') as file:
            file.write(TestHelpers.to_string(search_space))

        mappings = css.get_search_space_mappings()
        # this test mainly makes sure that the function runs, and outputs the result to a file so we can
        # track changes in github
        self.assertIsNotNone(mappings)
        with open(get_test_path('classification_get_search_space_mappings.txt'), 'w') as file:
            file.write(TestHelpers.to_string(mappings))

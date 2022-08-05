import unittest
import pandas as pd
import source.library.classification_search_space as css
from source.tests.helpers import get_test_file_path


class TestHelpers(unittest.TestCase):

    @staticmethod
    def to_string(obj):
        return str(obj). \
            replace(", '", ",\n'"). \
            replace('{', '{\n'). \
            replace('}', '\n}'). \
            replace(', ({', ',\n({')

    def test__classification_search_space(self):
        pipeline = css.create_pipeline(pd.DataFrame({
            'a': [1, 2],
            'b': [True, False],
            'c': ['Value1', 'Value2'],
        }))
        # this test mainly makes sure that the function runs, and outputs the result to a file so
        # we can track changes in github
        self.assertIsNotNone(pipeline)

        with open(get_test_file_path('classification_create_pipeline.txt'), 'w') as file:
            file.write(TestHelpers.to_string(pipeline))

        search_space = css.create_search_space(iterations=23, random_state=99)
        # this test mainly makes sure that the function runs, and outputs the result to a file so
        # we can track changes in github
        self.assertIsNotNone(search_space)
        with open(get_test_file_path('classification_create_search_space.txt'), 'w') as file:
            file.write(TestHelpers.to_string(search_space))

        mappings = css.get_search_space_mappings()
        # this test mainly makes sure that the function runs, and outputs the result to a file so
        # we can track changes in github
        self.assertIsNotNone(mappings)
        with open(get_test_file_path('classification_get_search_space_mappings.txt'), 'w') as file:
            file.write(TestHelpers.to_string(mappings))


if __name__ == '__main__':
    unittest.main()

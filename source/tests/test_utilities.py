import logging.config
import os
import time
import unittest
import pandas as pd
import source.library.classification_search_space as css
from source.library.utilities import Timer, dataframe_to_pickle, dataframe_to_csv, object_to_pickle
from source.tests.helpers import get_test_file_path

logging.config.fileConfig(get_test_file_path("test_logging.conf"),
                          defaults={'logfilename': get_test_file_path("log.log")},
                          disable_existing_loggers=False)


class TestHelpers(unittest.TestCase):

    @staticmethod
    def to_string(obj):
        return str(obj). \
            replace(", '", ",\n'"). \
            replace('{', '{\n'). \
            replace('}', '\n}'). \
            replace(', ({', ',\n({')

    def test__timer(self):
        print('\n')

        with Timer("testing timer"):
            time.sleep(0.05)

        with Timer("testing another timer") as timer:
            time.sleep(0.1)

        self.assertIsNotNone(timer._interval)

    def test__dataframe_to_pickle(self):
        df = pd.DataFrame(['test'])
        output_directory = 'temp_test'
        file_name = 'test.pkl'
        self.assertFalse(os.path.isdir(output_directory))
        file_path = dataframe_to_pickle(df=df, output_directory=output_directory, file_name=file_name)
        self.assertTrue(os.path.isfile(file_path))
        self.assertEqual(file_path, 'temp_test/test.pkl')
        df_unpickled = pd.read_pickle(file_path)
        self.assertTrue((df_unpickled.iloc[0] == df.iloc[0]).all())  # noqa
        os.remove(file_path)
        os.removedirs(output_directory)
        self.assertFalse(os.path.isdir(output_directory))

    def test__dataframe_to_csv(self):
        df = pd.DataFrame(['test'])
        output_directory = 'temp_test'
        file_name = 'test.csv'
        self.assertFalse(os.path.isdir(output_directory))
        file_path = dataframe_to_csv(df=df, output_directory=output_directory, file_name=file_name)
        self.assertTrue(os.path.isfile(file_path))
        self.assertEqual(file_path, 'temp_test/test.csv')
        df_unpickled = pd.read_csv(file_path)
        self.assertEqual(df_unpickled.loc[0, '0'], df.iloc[0, 0])
        os.remove(file_path)
        os.removedirs(output_directory)
        self.assertFalse(os.path.isdir(output_directory))

    def test__object_to_pickle(self):
        df = pd.DataFrame(['test'])
        output_directory = 'temp_test'
        file_name = 'test.pkl'
        self.assertFalse(os.path.isdir(output_directory))
        file_path = object_to_pickle(obj=df, output_directory=output_directory, file_name=file_name)
        self.assertTrue(os.path.isfile(file_path))
        self.assertEqual(file_path, 'temp_test/test.pkl')
        df_unpickled = pd.read_pickle(file_path)
        self.assertTrue((df_unpickled.iloc[0] == df.iloc[0]).all())  # noqa
        os.remove(file_path)
        os.removedirs(output_directory)
        self.assertFalse(os.path.isdir(output_directory))

    def test__classification_search_space(self):
        pipeline = css.create_pipeline(pd.DataFrame({
            'a': [1, 2],
            'b': [True, False],
            'c': ['Value1', 'Value2'],
        }))
        # this test mainly makes sure that the function runs, and outputs the result to a file so we can
        # track changes in github
        self.assertIsNotNone(pipeline)

        with open(get_test_file_path('classification_create_pipeline.txt'), 'w') as file:
            file.write(TestHelpers.to_string(pipeline))

        search_space = css.create_search_space(iterations=23, random_state=99)
        # this test mainly makes sure that the function runs, and outputs the result to a file so we can
        # track changes in github
        self.assertIsNotNone(search_space)
        with open(get_test_file_path('classification_create_search_space.txt'), 'w') as file:
            file.write(TestHelpers.to_string(search_space))

        mappings = css.get_search_space_mappings()
        # this test mainly makes sure that the function runs, and outputs the result to a file so we can
        # track changes in github
        self.assertIsNotNone(mappings)
        with open(get_test_file_path('classification_get_search_space_mappings.txt'), 'w') as file:
            file.write(TestHelpers.to_string(mappings))

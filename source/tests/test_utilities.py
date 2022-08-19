import pandas as pd
import source.domain.classification_search_space as css
from source.tests.helpers import get_test_file_path


def to_string(obj):
    return str(obj). \
        replace(", '", ",\n'"). \
        replace('{', '{\n'). \
        replace('}', '\n}'). \
        replace(', ({', ',\n({')


def test_classification_search_space():
    pipeline = css.create_pipeline(pd.DataFrame({
        'a': [1, 2],
        'b': [True, False],
        'c': ['Value1', 'Value2'],
    }))
    # this test mainly makes sure that the function runs, and outputs the result to a file so
    # we can track changes in github
    assert pipeline is not None

    with open(get_test_file_path('classification_create_pipeline.txt'), 'w') as file:
        file.write(to_string(pipeline))

    search_space = css.create_search_space(iterations=23, random_state=99)
    # this test mainly makes sure that the function runs, and outputs the result to a file so
    # we can track changes in github
    assert search_space is not None
    with open(get_test_file_path('classification_create_search_space.txt'), 'w') as file:
        file.write(to_string(search_space))

    mappings = css.get_search_space_mappings()
    # this test mainly makes sure that the function runs, and outputs the result to a file so
    # we can track changes in github
    assert mappings is not None
    with open(get_test_file_path('classification_get_search_space_mappings.txt'), 'w') as file:
        file.write(to_string(mappings))

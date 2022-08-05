import os


def get_test_file_path(file_path) -> str:
    """Returns the path to /tests folder, adjusting for the difference in the current working
    directory when debugging vs running from command line.
    """
    path = os.getcwd()
    return os.path.join(path, 'source/tests/test_files', file_path)

from helpsk.utility import open_yaml


def get_config():
    return open_yaml('source/config/config.yaml')

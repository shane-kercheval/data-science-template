import datetime
import logging
import logging.config
import yaml
import xmltodict


def get_logger(config="config/logging/local.conf", logger_name='app', leg_level="DEBUG"):
    logging.config.fileConfig(config, disable_existing_loggers=False)
    logger = logging.getLogger(logger_name)
    logger.setLevel(leg_level)

    return logger


def open_dict_like_file(file_name):
    with open(file_name, "r") as f:
        if file_name.endswith("json"):
            result = yaml.load(f)
        elif file_name.endswith("yaml") or file_name.endswith("yml"):
            result = yaml.load(f)
        elif file_name.endswith("xml"):
            result = xmltodict.parse(f.read())
        else:
            logging.warning("%s not a known dictionary-like file type", file_name)
    return result


class Timer:
    def __init__(self, description, logger):
        self.logger = logger
        self.description = description

    def __enter__(self):
        self.start = datetime.datetime.now()

        return self

    def __exit__(self, *args):
        self.end = datetime.datetime.now()
        self.interval = self.end - self.start
        self.logger.info("%s took %0.2f seconds", self.description, self.interval.total_seconds())

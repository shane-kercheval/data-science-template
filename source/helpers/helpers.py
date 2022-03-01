import datetime


def open_dictlike_file(fname):
    with open(fname, "r") as f:
        if fname.endswith("json"):
            fdict = yaml.load(f)
        elif fname.endswith("yaml") or fname.endswith("yml"):
            fdict = yaml.load(f)
        elif fname.endswith("xml"):
            fdict = xmltodict.parse(f.read())
        else:
            logging.warning("%s not a known dictionary-like file type", fname)
    return fdict


class Timer:
    def __init__(self, function, logger):
        self.logger = logger
        self.function = function

    def __enter__(self):
        self.start = datetime.datetime.now()

        return self

    def __exit__(self, *args):
        self.end = datetime.datetime.now()
        self.interval = self.end - self.start
        self.logger.info("%s took %0.2f seconds", self.function, self.interval.total_seconds())
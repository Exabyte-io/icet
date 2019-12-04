import logging
import sys

# This is the root logger of icet
logger = logging.getLogger('icet')

# Set the format which is simpler for the console
FORMAT = '%(name)s: %(levelname)s  %(message)s'
formatter = logging.Formatter(FORMAT)

# Will process all levels of DEBUG or higher
logger.setLevel(logging.DEBUG)

# If you know what you are doing you may set this to True
logger.propagate = False

# The logger will collect events from children and the default
# behaviour is to print it directly to stdout
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(formatter)
sh.setLevel(logging.WARNING)
logger.addHandler(sh)


class MyFilter(logging.Filter):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno == self.__level


def set_log_config(filename=None, level=None, restricted=False):

    # If a filename is provided a logfile will be created
    if filename is not None:
        fh = logging.FileHandler(filename)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Restrict file handler to only write out at the level specified
        if restricted:
            lvl = logging.getLevelName(level)
            fh.addFilter(MyFilter(lvl))

        if level is not None:
            fh.setLevel(level)

    # If only level is provided, the stream handler will be reset to this level
    elif level is not None:
        sh.setLevel(level)

import logging
import sys


# This is the root logger of icet
logger = logging.getLogger('icet')

# Will process all levels of INFO or higher
logger.setLevel(logging.INFO)

# If you know what you are doing you may set this to True
logger.propagate = False

# The logger will collect events from children and the default
# behaviour is to print it directly to stdout
ch = logging.StreamHandler(sys.stdout)
logger.addHandler(ch)


# continuous = True

def set_config(filename=None, level=None, continuous=None):

    # If a filename is provided a logfile will be created
    if filename is not None:
        fh = logging.FileHandler(filename)
        logger.addHandler(fh)

    if level is not None:
        logger.setLevel(level)

#    if continuous is not None:
#        sys.modules[__name__].continuous = continuous

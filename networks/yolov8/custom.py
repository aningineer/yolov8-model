
import logging

########################################### LOGGER ###########################################
LOGGING_NAME = "PyContrast"
LOGGING_LEVEL = logging.DEBUG

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    lightpink = "\x1b[37;20m"
    cyan = "\x1b[36;20m"
    purple = "\x1b[35;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "%(message)s"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# initialise logger
LOGGER = logging.getLogger(LOGGING_NAME)
LOGGER.setLevel(LOGGING_LEVEL)

# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(LOGGING_LEVEL)
ch.setFormatter(CustomFormatter())
LOGGER.addHandler(ch)
########################################### LOGGER ###########################################
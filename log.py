import time
import logging
import os


def configure_logger(data_path):
    """
    Sets up a logger that prints to the console and to a file
    Returns:
        logger object
    """
    logdatetime = time.strftime("%Y_%m_%d__%H_%M_%S")

    # Stop matplotlib filling up logs
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        level=logging.DEBUG,
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )
    logFormatter = logging.Formatter(
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    )
    logger = logging.getLogger(__name__)

    fileHandler = logging.FileHandler(
        os.path.join(data_path, "output", "{}_{}_run.log".format(logdatetime))
    )
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)
    return logger

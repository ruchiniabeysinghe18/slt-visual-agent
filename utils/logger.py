import logging
import logging.handlers
from pathlib import Path

"""
Helper script which invoker logger object
"""


def get_debug_logger(name, file_name):
    """
    description:Initialize the logger object
    parameter  :
                name      : name of the logger instance
                file_name : path to the logger file
    return     :
                logger object
    """
    _logger = logging.getLogger(name)
    if not _logger.hasHandlers():
        _logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # FileHandler setup
        file_handler = logging.FileHandler(file_name, encoding='utf-8')
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

        # StreamHandler setup
        file_path = str(file_name)
        if file_path.endswith("server.log"):
            import sys
            stream_handler = logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False))
            color_formatter = ColorFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            stream_handler.setFormatter(color_formatter)
            _logger.addHandler(stream_handler)

    return _logger


# Custom formatter with color support


class ColorFormatter(logging.Formatter):

    ASCTIME_COLOR = "\033[92m"  # Green for asctime
    LEVEL_COLORS = {
        "WARNING": "\033[93m",  # Yellow for WARNING level
        "INFO": "\033[94m",  # Blue for INFO level
        "DEBUG": "\033[96m",  # Cyan for DEBUG level
        "CRITICAL": "\033[95m",  # Purple for CRITICAL level
        "ERROR": "\033[91m",  # Red for ERROR level
    }
    RESET = "\033[0m"  # Reset color

    def format(self, record):
        log_fmt = self._fmt
        asctime_fmt = self.ASCTIME_COLOR + "%(asctime)s" + self.RESET
        level_fmt = self.LEVEL_COLORS.get(record.levelname, self.RESET) + "%(levelname)s" + self.RESET
        log_fmt = log_fmt.replace("%(asctime)s", asctime_fmt).replace("%(levelname)s", level_fmt)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
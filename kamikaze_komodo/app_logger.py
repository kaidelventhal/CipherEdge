# kamikaze_komodo/app_logger.py
import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file_path = os.path.join(LOG_DIR, "kamikaze_komodo.log")

# Configure logging
logger = logging.getLogger("KamikazeKomodo")
logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels of messages

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Console logs info and above

file_handler = RotatingFileHandler(
    log_file_path, maxBytes=10*1024*1024, backupCount=5  # 10MB per file, 5 backups
)
file_handler.setLevel(logging.DEBUG)  # File logs debug and above

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s')
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a logger instance for a specific module.
    """
    return logging.getLogger(f"KamikazeKomodo.{module_name}")
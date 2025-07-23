import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file_path = os.path.join(LOG_DIR, "cipher_edge.log")

logger = logging.getLogger("CipherEdge")
logger.setLevel(logging.DEBUG)  

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) 

file_handler = RotatingFileHandler(
    log_file_path, maxBytes=10*1024*1024, backupCount=5 
)
file_handler.setLevel(logging.DEBUG) 

log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s - %(message)s')
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a logger instance for a specific module.
    """
    return logging.getLogger(f"CipherEdge.{module_name}")
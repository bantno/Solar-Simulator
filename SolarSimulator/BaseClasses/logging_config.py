# logging_config.py
import logging

def setup_logging(log_level=logging.INFO):
    """
    Configure logging for the application.
    
    Parameters:
        log_level: Logging level (default: INFO).
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

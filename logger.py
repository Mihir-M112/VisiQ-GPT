import logging
import os
from datetime import datetime

def get_logger(name):
    """
    Configures and returns a logger that writes logs to `log.txt` file and console.
    Adds a separator line for each new execution.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Ensure the logs directory exists
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    # File where logs will be saved
    log_file = os.path.join(log_dir, "log.txt")

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Log all levels (DEBUG and above)

    # Formatter for logs
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # File handler (writes logs to a file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(log_format)

    # Console handler (prints logs to the console)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Show INFO level logs in the console
    console_handler.setFormatter(log_format)

    # Add handlers to the logger
    if not logger.hasHandlers():  # Avoid duplicate handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    # Add a separator line for each new execution
    separator = f"\n{'=' * 50}\n=== New Execution: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n{'=' * 50}\n"
    with open(log_file, "a") as f:
        f.write(separator)

    return logger

if __name__ == "__main__":
    # Example usage
    logger = get_logger(__name__)
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

import logging
import sys


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures application-wide structured logging."""
    logger = logging.getLogger("document_qa_rag")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()

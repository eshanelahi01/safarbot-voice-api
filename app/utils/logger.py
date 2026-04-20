import logging
import os


_CONFIGURED = False


def _ensure_logging():
    global _CONFIGURED

    if _CONFIGURED:
        return

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _ensure_logging()
    return logging.getLogger(name)

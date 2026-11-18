import logging
from pathlib import Path
from typing import Optional, Union

stream_logging_format = "\033[36m%(asctime)s\033[0m - \033[32m%(name)s\033[0m - \033[1;33m%(levelname)s\033[0m - %(message)s"
file_logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S %Z"

# Marks the handlers THIS function attached, so a repeat call replaces its own and never
# touches one a caller added themselves.
_OWNED_BY_CREATE_LOGGER = "_bioengine_created"


def create_logger(
    name: str, level: int = logging.INFO, log_file: Optional[Union[str, Path]] = None
) -> logging.Logger:
    """Configure and return the logger for ``name``. Idempotent per name.

    ``logging.getLogger(name)`` returns the SAME object for the same name, so attaching a
    handler unconditionally made the handler list grow by one per call -- two with ``log_file``
    -- and every line was then emitted once per handler. Measured on 0.16.1 and still on
    0.16.115: five calls left five handlers, one ``warning()`` printed three lines after three
    calls, and the ``log_file`` path leaked one file descriptor per call (4 -> 9 fds over five
    calls). A log where one event appears 259 times is not a noisy log, it is an unusable
    record: it looks exactly like work done 259 times (github aicell-lab/bioengine#232, myco
    #0644).

    The name says *create* and every call site reads it as an idempotent accessor -- including
    six inside this package, in ``utils/artifact_utils.py`` alone, each in a function that can
    be called repeatedly.

    Handlers this function previously attached are removed AND CLOSED before new ones are
    attached. Closing is the part that matters for ``log_file``: ``removeHandler`` alone
    detaches the handler and leaves its descriptor open.
    """
    logger = logging.getLogger(name)
    # Applied on every call, so a later call that raises the level still takes effect -- an
    # early-return implementation would silently ignore it, and worker.py passes DEBUG or INFO
    # depending on its --debug flag.
    logger.setLevel(level)
    logger.propagate = False

    for handler in [
        h for h in logger.handlers if getattr(h, _OWNED_BY_CREATE_LOGGER, False)
    ]:
        logger.removeHandler(handler)
        handler.close()

    # Create console handler with a custom format
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(fmt=stream_logging_format, datefmt=date_format)
    console_handler.setFormatter(formatter)
    setattr(console_handler, _OWNED_BY_CREATE_LOGGER, True)
    logger.addHandler(console_handler)

    if log_file:
        log_dir = Path(log_file).resolve().parent
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(fmt=file_logging_format, datefmt=date_format)
        file_handler.setFormatter(formatter)
        setattr(file_handler, _OWNED_BY_CREATE_LOGGER, True)
        logger.addHandler(file_handler)

    return logger

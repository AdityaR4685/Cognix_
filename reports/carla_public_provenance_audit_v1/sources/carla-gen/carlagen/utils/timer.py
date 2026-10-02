import logging
from time import perf_counter
from types import TracebackType
from typing import Optional, Type


class log_timer:  # pylint: disable=invalid-name
    """
    Context manager that measures elapsed wall-clock time and logs it.

    Parameters
    ----------
    logger : logging.Logger
        Destination for the timing message. Falls back to the
        root logger if None.
    level : int, optional
        Logging level (e.g., logging.INFO or logging.DEBUG). Default is INFO.
    msg : str, optional
        Message template; may contain one `{elapsed:.{precision}f}` placeholder.
    precision : int, optional
        Number of decimal places in the formatted seconds. Default is 3.

    Example
    -------
    >>> import logging, time
    >>> logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    >>> with log_timer(logging.getLogger(__name__), msg="FFT took {elapsed:.4f}s"):
    ...     time.sleep(0.15)
    INFO: FFT took 0.1502s
    """

    __slots__ = ("_logger", "_level", "_msg", "_precision", "_start")

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        *,
        level: int = logging.DEBUG,
        msg: str = "Elapsed time: {elapsed:.{precision}f}s",
        precision: int = 3,
    ) -> None:
        self._logger = logger or logging.getLogger()
        self._level = level
        self._msg = msg
        self._precision = precision
        self._start: float = 0.0

    # ——— context-manager protocol ———
    def __enter__(self) -> "log_timer":  # noqa: D401
        self._start = perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        elapsed = perf_counter() - self._start
        self._logger.log(
            self._level,
            f"{self._msg} in {elapsed:.{self._precision}f}s",
        )
        # Propagate any exception that occurred inside the with-block
        return False

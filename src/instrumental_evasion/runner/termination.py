"""Let the runner finalize an interrupted attempt before the worker exits."""

import signal
from contextlib import contextmanager


class TerminationRequested(BaseException):
    """Stop agent execution without treating an external stop as task failure."""


@contextmanager
def finalize_on_termination():
    """Handle SIGINT and SIGTERM once; allow bounded scoring and cleanup.

    A synchronous handler also interrupts the blocking CLI capture loop.
    The scheduler must allow a cleanup grace period before SIGKILL.
    """
    previous = {number: signal.getsignal(number) for number in (signal.SIGINT, signal.SIGTERM)}

    def stop(number, _frame):
        for value in previous:
            signal.signal(value, signal.SIG_IGN)
        raise TerminationRequested(signal.Signals(number).name)

    try:
        for number in previous:
            signal.signal(number, stop)
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)

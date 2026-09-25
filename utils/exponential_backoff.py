import time

_RETRYABLE_ERROR_TOKENS = (
    "rate limit",
    "429",
    "500",
    "internal",
    "502",
    "503",
    "504",
    "service unavailable",
    "deadline exceeded",
)


def _is_retryable_error(error: Exception) -> bool:
    message = str(error).lower().replace("_", " ")
    return any(token in message for token in _RETRYABLE_ERROR_TOKENS)


def retry_with_exponential_backoff(
    func,
    max_retries: int = 20,
    initial_sleep_time: float = 1.0,
    backoff_factor: float = 1.5,
):
    """
    Retry a function with exponential backoff.

    This decorator retries the wrapped function in case of rate limit errors, using an exponential
    backoff strategy to increase the wait time between retries.

    Args:
        func (callable): The function to be retried.
        max_retries (int): Maximum number of retry attempts.
        initial_sleep_time (float): Initial sleep time in seconds.
        backoff_factor (float): Factor by which the sleep time increases after each retry.

    Returns:
        callable: A wrapped version of the input function with retry logic.

    Raises:
        Exception: If the maximum number of retries is exceeded.
        Any other exception raised by the function that is not a rate limit error.

    Note:
        This function specifically handles rate limit errors. All other exceptions
        are re-raised immediately.
    """

    def wrapper(*args, **kwargs):
        sleep_time = initial_sleep_time

        for _ in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if not _is_retryable_error(e):
                    raise e

                time.sleep(sleep_time)
                sleep_time *= backoff_factor

        raise Exception(f"Exceeded {max_retries} of tries")

    return wrapper

import time


def retry_with_exponential_backoff(
    func,
    max_retries=10,
    initial_sleep_time=1.0,
    backoff_factor=2.0,
):

    def wrapper(*args, **kwargs):
        delay = initial_sleep_time

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)

            except Exception as e:
                code = str(getattr(e, "code", ""))

                retryable = (
                    code in {"429", "500", "502", "503", "504"}
                    or isinstance(e, (TimeoutError, ConnectionError))
                )

                if not retryable or attempt == max_retries - 1:
                    raise

                time.sleep(delay)
                delay *= backoff_factor

    return wrapper
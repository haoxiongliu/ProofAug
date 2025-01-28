import threading
import time
from functools import wraps
import multiprocessing
import io
from contextlib import redirect_stdout
import logging
import datetime
import functools

time_logger = logging.getLogger('time_logger')
time_logger.addHandler(logging.FileHandler('time.log'))
time_logger.setLevel(logging.DEBUG)
time_logger.handlers[0].setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s'))


def terminate_process(process: multiprocessing.Process, t_ter: int = 10, t_kill: int = 60):
    """Terminates the given process and ensures it is killed if it doesn't terminate."""
    process.terminate()
    process.join(t_ter)
    if process.is_alive():
        process.kill()
        process.join(t_kill)
    process.close()

# let the decorated function return the printed rather than original return value 'result'
def timeout_catchio_process(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def target(q, *args, **kwargs):
                program_io = io.StringIO()
                with redirect_stdout(program_io):
                    try:
                        result = func(*args, **kwargs)
                        q.put((program_io.getvalue(), None))
                    except Exception as e:
                        q.put((None, e))

            q = multiprocessing.Queue()
            process = multiprocessing.Process(target=target, args=(q,) + args, kwargs=kwargs)
            process.start()
            try:
                result, exception = q.get(timeout=seconds)
            except:
                terminate_process(process)
                raise TimeoutError("Function execution timed out")

            terminate_process(process)

            if exception:
                raise exception
            return result
        return wrapper
    return decorator


def with_timeout_process(timeout):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            queue = multiprocessing.Queue()

            def target(queue, *args, **kwargs):
                result = func(*args, **kwargs)
                queue.put(result)

            p = multiprocessing.Process(target=target, args=(queue, *args), kwargs=kwargs)
            p.start()
            try:
                result = queue.get(timeout=timeout)
            except multiprocessing.queues.Empty:
                print(f'{timeout=} when handling {func.__name__} with {args=} and {kwargs=}')
                result = None  # Indicates that a timeout occurred.
            finally:
                terminate_process(p)
            return result
        return wrapper
    return decorator


def with_timeout(timeout: float, raise_error: bool = False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e

            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                msg = f'{timeout=} when handling {func.__name__} with {args=} and {kwargs=}'
                if raise_error:
                    raise TimeoutError(msg)
                else:
                    print(msg)
                    return None  # Indicates that a timeout occurred.
            if exception[0]:
                raise exception[0]
            return result[0]
        return wrapper
    return decorator


def timeit_repeated(k):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            for _ in range(k):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            avg_time = sum(times) / k
            print(f"Average execution time over {k} runs: {avg_time:.8f} seconds")
            return result
        return wrapper
    return decorator

def log_func_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.datetime.now()
        print(f"{start_time} - start {func.__name__}")
        result = func(*args, **kwargs)
        end_time = datetime.datetime.now()
        print(f"{end_time} - end {func.__name__}")
        return result

    return wrapper
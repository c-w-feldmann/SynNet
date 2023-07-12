"""parallel.py
"""
import logging
from typing import Any, Callable, Optional, TypeVar

from tqdm import tqdm

from synnet.config import MAX_PROCESSES

T = TypeVar("T")
T2 = TypeVar("T2")


def compute_chunksize(input_list: list[Any], cpus: int) -> int:
    """Source: https://github.com/python/cpython/blob/816066f497ab89abcdb3c4f2d34462c750d23713/Lib/multiprocessing/pool.py#L477"""
    chunksize, extra = divmod(len(input_list), cpus * 4)
    if extra:
        chunksize += 1
    if len(input_list) == 0:
        chunksize = 0
    return chunksize


def simple_parallel(
    input_list: list[T],
    function: Callable[[T], T2],
    max_cpu: int = MAX_PROCESSES,
    timeout: int = 4000,
    max_retries: int = 3,
    verbose: bool = False,
) -> list[T2]:
    """Use map async and retries in case we get odd stalling behavior"""
    # originally from: https://github.com/samgoldman97
    from multiprocess.context import TimeoutError
    from pathos import multiprocessing as mp

    def setup_pool() -> tuple[mp.Pool, list[Any]]:
        pool = mp.Pool(processes=max_cpu)
        async_results = [pool.apply_async(function, args=(i,)) for i in input_list]
        # Note from the docs:
        #   "func is only executed in one of the workers of the pool",
        #   -> so we call apply_async for each input in the list
        pool.close()
        return pool, async_results

    pool, async_results = setup_pool()

    retries = 0
    while True:
        try:
            if verbose:
                async_results = tqdm(async_results, total=len(input_list))
            list_outputs = [async_result.get(timeout) for async_result in async_results]
            break
        except TimeoutError:
            retries += 1
            logging.info(f"Timeout Error (s > {timeout})")
            if retries <= max_retries:
                pool, async_results = setup_pool()
                logging.info(f"Retry attempt: {retries}")
            else:
                raise ValueError()

    return list_outputs


def chunked_parallel(
    input_list: list[T],
    function: Callable[[T], T2],
    chunks: Optional[int] = None,
    max_cpu: int = MAX_PROCESSES,
    timeout: int = 4000,
    max_retries: int = 3,
    verbose: bool = False,
) -> list[T2]:
    """chunked_parallel.
    Args:
        input_list : list of objects to apply function
        function : Callable with 1 input and returning a single value
        chunks: number of chunks
        max_cpu: Max num cpus
        timeout: Length of timeout
        max_retries: Num times to retry this

    Example:

    ```python
    input_list = [1,2,3,4,5]
    func = lambda x: x**10
    res = chunked_parallel(input_list,func,verbose=True,max_cpu=4)
    ```

    """
    # originally from: https://github.com/samgoldman97

    # Run plain list comp when no mp is necessary.
    # Note: Keeping this here to have a single interface.
    if max_cpu == 1:
        if verbose:
            input_list = tqdm(input_list)
        return [function(i) for i in input_list]

    # Adding it here fixes some setting disrupted elsewhere
    def batch_func(list_inputs: list[Any]) -> list[Any]:
        return [function(i) for i in list_inputs]

    num_chunks = compute_chunksize(input_list, max_cpu) if chunks is None else chunks
    step_size = len(input_list) // num_chunks

    chunked_list = [input_list[i : i + step_size] for i in range(0, len(input_list), step_size)]
    logging.debug(
        f"{max_cpu=}, {len(input_list)=}, {num_chunks=}, {step_size=}, {len(chunked_list)=}"
    )

    list_outputs = simple_parallel(
        chunked_list,
        batch_func,
        max_cpu=max_cpu,
        timeout=timeout,
        max_retries=max_retries,
        verbose=verbose,
    )
    # Unroll
    full_output = [item for sublist in list_outputs for item in sublist]

    return full_output

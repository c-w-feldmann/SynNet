"""parallel.py"""

from typing import Any, Callable, Optional, TypeVar

from loguru import logger
from tqdm import tqdm

try:
    from pathos import multiprocessing as mp
except ImportError:
    logger.warning("Pathos not found, using multiprocessing instead")
    import multiprocessing as mp

from synnet.config import MAX_PROCESSES

T = TypeVar("T")
V = TypeVar("V")


def compute_chunksize(input_list: list[Any], cpus: int) -> int:
    """Compute the size of a chunk of input list.

    References
    ----------
    https://github.com/python/cpython/blob/816066f497ab89abcdb3c4f2d34462c750d23713/Lib/multiprocessing/pool.py#L477

    Parameters
    ----------
    input_list : list
        List of inputs
    cpus : int
        Number of cpus

    Returns
    -------
    int
        Size of chunk
    """
    chunksize, extra = divmod(len(input_list), cpus * 4)
    if extra:
        chunksize += 1
    if len(input_list) == 0:
        chunksize = 0
    return chunksize


def simple_parallel(
    input_list: list[T],
    function: Callable[[T], V],
    max_cpu: int = MAX_PROCESSES,
    timeout: int = 4000,
    max_retries: int = 3,
    verbose: bool = False,
) -> list[V]:
    """Use map async and retries in case we get odd stalling behavior.

    Parameters
    ----------
    input_list : list
        List of inputs
    function : Callable[[T], T2]
        Function to apply to each input
    max_cpu : int, optional
        Max number of cpus, by default MAX_PROCESSES
    timeout : int, optional
        Timeout, by default 4000
    max_retries : int, optional
        Max number of retries, by default 3
    verbose : bool, optional
        Verbose, by default False

    Returns
    -------
    list
        List of outputs
    """
    # originally from: https://github.com/samgoldman97

    def setup_pool() -> tuple[mp.Pool, list[Any]]:
        """Setup the pool and async results.

        Returns
        -------
        mp.Pool
            Pool object
        list[mp.AsyncResult]
            List of async results.
        """
        pool = mp.Pool(processes=max_cpu)
        async_results = [pool.apply_async(function, args=(i,)) for i in input_list]
        # Note from the docs:
        #   "func is only executed in one of the workers of the pool",
        #   -> so we call apply_async for each input in the list
        pool.close()
        return pool, async_results

    _, async_results = setup_pool()

    retries = 0
    while True:
        try:
            list_outputs = []
            iterator = tqdm(async_results, total=len(input_list), disable=not verbose)
            for async_r in iterator:
                list_outputs.append(async_r.get(timeout))
            break
        except TimeoutError as e:
            retries += 1
            logger.info("Timeout Error (s > {})", timeout)
            if retries <= max_retries:
                _, async_results = setup_pool()
                logger.info(f"Retry attempt: {retries}")
            else:
                raise TimeoutError(f"Max retries exceeded: {max_retries}") from e

    return list_outputs


def chunked_parallel(
    input_list: list[T],
    function: Callable[[T], V],
    chunks: Optional[int] = None,
    max_cpu: int = MAX_PROCESSES,
    timeout: int = 4000,
    max_retries: int = 3,
    verbose: bool = False,
) -> list[V]:
    """Apply a function to a list of objects in parallel using chunks.

    Examples
    --------
    ```python
    input_list = [1,2,3,4,5]
    func = lambda x: x**10
    res = chunked_parallel(input_list,func,verbose=True,max_cpu=4)
    ```

    Parameters
    ----------
    input_list: list[T]
        list of objects to apply function
    function: Callable[[T], T2]
        Callable with 1 input and returning a single value
    chunks: Optional[int]
        number of chunks
    max_cpu: int
        Max num cpus to use.
        Default is MAX_PROCESSES
    timeout: int
        Length of timeout in seconds
    max_retries: int
        Num times to retry this

    Returns
    -------
    list[T2]
        List of outputs
    """
    # originally from: https://github.com/samgoldman97

    # Run plain list comp when no mp is necessary.
    # Note: Keeping this here to have a single interface.
    if max_cpu == 1:
        return [function(i) for i in tqdm(input_list, disable=not verbose)]

    # Adding it here fixes some setting disrupted elsewhere
    def batch_func(list_inputs: list[Any]) -> list[Any]:
        """Apply function to a list of inputs.

        Parameters
        ----------
        list_inputs : list
            List of inputs

        Returns
        -------
        list
            List of outputs
        """
        return [function(i) for i in list_inputs]

    num_chunks = compute_chunksize(input_list, max_cpu) if chunks is None else chunks
    step_size = len(input_list) // num_chunks

    chunked_list = [
        input_list[i : i + step_size] for i in range(0, len(input_list), step_size)
    ]
    logger.debug(
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

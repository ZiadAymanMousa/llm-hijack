"""Run independent jobs on a thread pool; results come back as they finish.

Episodes and guard calls are independent and spend their time waiting on the vLLM
servers, so threads are enough. Callers write results from the main thread only.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed


def run_parallel(fn, items, workers: int):
    if workers <= 1:
        yield from map(fn, items)
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
        for future in as_completed(futures):
            yield future.result()

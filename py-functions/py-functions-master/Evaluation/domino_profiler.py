import time
import tracemalloc
import psutil
import pandas as pd
import polars as pl

def estimate_cost_difference(original_function, original_data, new_function, new_data, N):
    """Estimates the cost difference between two functions for different instance types on a cloud provider.

    Parameters
    ----------
    original_function : callable
        The original function to compare.
    original_data : pd.DataFrame or pl.DataFrame
        The data to pass to the original function. Must be either a pandas or polars DataFrame.
    new_function : callable
        The new function to compare.
    new_data : pd.DataFrame or pl.DataFrame
        The data to pass to the new function. Must be either a pandas or polars DataFrame.
    N : int
        The number of times to run each function to estimate their execution time and memory usage.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
            - Instance Type: The type of cloud instance used for the benchmark.
            - Original Execution Time (s): The average execution time of the original function in seconds.
            - New Execution Time (s): The average execution time of the new function in seconds.
            - Time Difference (%): The percentage difference in execution time between the new and original functions.
            - Original Memory Usage (MiB): The peak memory usage of the original function in mebibytes (MiB).
            - New Memory Usage (MiB): The peak memory usage of the new function in mebibytes (MiB).
            - Memory Difference (%): The percentage difference in memory usage between the new and original functions.
            - Original Cost per N runs: The estimated cost of running the original function N times.
            - New Cost per N runs: The estimated cost of running the new function N times.
            - Cost Difference (%): The percentage difference in cost between the new and original functions.

    Raises
    ------
    ValueError
        If the data provided is not a pandas or polars DataFrame.

    Notes
    -----
    This function estimates the cost difference between the original and new functions for different instance types on a cloud provider.
    It assumes that the functions are stateless and can be run independently. It also assumes that the peak memory usage occurs during
    the execution of the function, and not during the setup or teardown phases.
    """
    instances = {
        "Small 1 core - 4 GiB RAM": 0.048,
        "Medium 4 cores - 15 GiB RAM": 0.192,
        "Large 6 cores - 27 GiB RAM": 0.288,
        "GPU 7 cores - 32 GiB RAM": 1.62
    }

    results = []
    for instance_type, cost_per_hour in instances.items():
        instance_memory = psutil.virtual_memory().available / (1024 * 1024)

        # Make copies of the data for each function
        if isinstance(original_data, pd.DataFrame):
            original_df_copy = original_data.copy()
        elif isinstance(original_data, pl.DataFrame):
            original_df_copy = original_data.clone()
        else:
            raise ValueError("Invalid dataframe type, must be either pandas or polars")

        if isinstance(new_data, pd.DataFrame):
            new_df_copy = new_data.copy()
        elif isinstance(new_data, pl.DataFrame):
            new_df_copy = new_data.clone()
        else:
            raise ValueError("Invalid dataframe type, must be either pandas or polars")

        # Time the original function
        try:
            tracemalloc.start()
            start_time = time.time()
            for i in range(N):
                original_function(original_df_copy)
            end_time = time.time()
            original_execution_time = end_time - start_time
            original_memory_usage = tracemalloc.get_traced_memory()
            original_cost = (cost_per_hour / 60) * (original_execution_time / 60) # assuming we run the function for 1 hour
        finally:
            tracemalloc.stop()

        # Time the new function
        try:
            tracemalloc.start()
            start_time = time.time()
            for i in range(N):
                new_function(new_df_copy)
            end_time = time.time()
            new_execution_time = end_time - start_time
            new_memory_usage = tracemalloc.get_traced_memory()
            new_cost = (cost_per_hour / 60) * (new_execution_time / 60) # assuming we run the function for 1 hour
        finally:
            tracemalloc.stop()

        # Calculate the differences between the two functions
        time_difference = ((new_execution_time / original_execution_time) - 1) * 100
        memory_difference = ((new_memory_usage[1] / original_memory_usage[1]) - 1) * 100
        cost_difference = ((new_cost / original_cost) - 1) * 100
        
        if new_execution_time < original_execution_time:
            time_difference = abs(time_difference)
        else:
            time_difference = -1 * abs(time_difference)
        
        if new_memory_usage[1] < original_memory_usage[1]:
            memory_difference = abs(memory_difference)
        else:
            memory_difference = -1 * abs(memory_difference)
        
        if new_cost < original_cost:
            cost_difference = abs(cost_difference)
        else:
            cost_difference = -1 * abs(cost_difference)
        
        original_time = round(original_execution_time, 2)
        new_time = round(new_execution_time, 2)
        original_memory = round(original_memory_usage[1] / (1024 * 1024), 2)
        new_memory = round(new_memory_usage[1] / (1024 * 1024), 2)
        original_cost_per_n = '${:.2f}/{}'.format(original_cost, N)
        new_cost_per_n = '${:.2f}/{}'.format(new_cost, N)

        
        # Append results to the list of results
        results.append({
            "Instance Type": instance_type,
            "Original Execution Time (s)": original_time,
            "New Execution Time (s)": new_time,
            "Time Difference (%)": '{:.2f}%'.format(time_difference),
            "Original Memory Usage (MiB)": original_memory,
            "New Memory Usage (MiB)": format(round(new_memory_usage[1] / (1024 * 1024), 2)),
            "Memory Difference (%)": '{:.2f}%'.format(memory_difference),
            "Original Cost per {} runs".format(N): "${:.10f}".format(original_cost),
            "New Cost per {} runs".format(N): "${:.10f}".format(new_cost),
            "Cost Difference (%)": '{:.2f}%'.format(cost_difference)
        })

    # Create a dataframe from the results and return it
    results_df = pd.DataFrame(results)
    return results_df

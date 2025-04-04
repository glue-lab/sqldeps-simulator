"""Simulation engine for SQLDeps cost and time analysis.

This module provides functions for simulating and comparing the time, cost,
and efficiency of manual SQL dependency analysis versus automated approaches
using SQLDeps with different configurations.
"""

from .models import SimulationParams, SimulationResult


def calculate_manual_analysis_time(
    num_queries: int, avg_human_analysis_time_per_query: float
) -> float:
    """Calculate the total time for manual analysis.

    Args:
        num_queries: Number of SQL queries to analyze.
        avg_human_analysis_time_per_query: Average time in minutes for human analysis per query.

    Returns:
        Total time in minutes.
    """
    return num_queries * avg_human_analysis_time_per_query


def calculate_sequential_processing_time(
    num_queries: int, avg_api_time_per_query: float, api_rate_limit: int | None = None
) -> float:
    """Calculate the total time for sequential processing with SQLDeps.

    Args:
        num_queries: Number of SQL queries to analyze.
        avg_api_time_per_query: Average time in minutes for API processing per query.
        api_rate_limit: Requests per minute limit. None means no limit.

    Returns:
        Total time in minutes.
    """
    # Calculate theoretical time
    theoretical_time = num_queries * avg_api_time_per_query

    # If rate limit is specified, apply rate limiting
    if api_rate_limit is not None:
        rate_limited_time = num_queries / api_rate_limit
        return max(theoretical_time, rate_limited_time)

    # No rate limit
    return theoretical_time


def calculate_worker_processing_times(
    num_queries: int,
    avg_api_time_per_query: float,
    max_workers: int,
    api_rate_limit: int | None = None,
) -> list[float]:
    """Calculate processing times for different numbers of workers.

    Args:
        num_queries: Number of SQL queries to analyze.
        avg_api_time_per_query: Average time in minutes for API processing per query.
        max_workers: Maximum number of parallel workers.
        api_rate_limit: Requests per minute limit. None means no limit.

    Returns:
        List of processing times for worker counts from 1 to max_workers.
    """
    worker_times = []
    for workers in range(1, max_workers + 1):
        # Theoretical time with perfect parallelism
        theoretical_time = (num_queries * avg_api_time_per_query) / workers

        # Rate-limited time (shared limiter across workers)
        rate_limited_time = num_queries / api_rate_limit if api_rate_limit is not None else 0

        # The actual processing time is constrained by both limits
        worker_times.append(max(theoretical_time, rate_limited_time))

    return worker_times


def calculate_parallel_processing_time(
    num_queries: int,
    avg_api_time_per_query: float,
    max_workers: int,
    api_rate_limit: int | None = None,
) -> float:
    """Calculate the total time for parallel processing with SQLDeps.

    Args:
        num_queries: Number of SQL queries to analyze.
        avg_api_time_per_query: Average time in minutes for API processing per query.
        max_workers: Maximum number of parallel workers.
        api_rate_limit: Requests per minute limit. None means no limit.

    Returns:
        Total time in minutes with optimal parallelism.
    """
    worker_times = calculate_worker_processing_times(
        num_queries, avg_api_time_per_query, max_workers, api_rate_limit
    )
    return min(worker_times) if worker_times else float("inf")


def calculate_manual_analysis_cost(
    num_queries: int, avg_human_analysis_time_per_query: float, hourly_salary: float
) -> float:
    """Calculate the cost of manual analysis.

    Args:
        num_queries: Number of SQL queries to analyze.
        avg_human_analysis_time_per_query: Average time in minutes for human analysis per query.
        hourly_salary: Hourly salary in USD for a data engineer.

    Returns:
        Total cost in USD.
    """
    # Convert minutes to hours and multiply by hourly salary
    hours = (num_queries * avg_human_analysis_time_per_query) / 60
    return hours * hourly_salary


def calculate_api_costs(
    num_queries: int,
    avg_input_prompt_ntokens: int,
    avg_output_prompt_ntokens: int,
    api_input_price: float,
    api_output_price: float,
) -> tuple[float, float, float]:
    """Calculate the API costs.

    Args:
        num_queries: Number of SQL queries to analyze.
        avg_input_prompt_ntokens: Average number of input tokens per query.
        avg_output_prompt_ntokens: Average number of output tokens per query.
        api_input_price: Price per 1M tokens for API input in USD.
        api_output_price: Price per 1M tokens for API output in USD.

    Returns:
        Tuple of (input_cost, output_cost, total_cost) in USD.
    """
    total_input_tokens = num_queries * avg_input_prompt_ntokens
    total_output_tokens = num_queries * avg_output_prompt_ntokens

    input_cost = (total_input_tokens / 1_000_000) * api_input_price
    output_cost = (total_output_tokens / 1_000_000) * api_output_price
    total_cost = input_cost + output_cost

    return input_cost, output_cost, total_cost


def calculate_processing_capacity(
    manual_time: float, sequential_time: float, parallel_time: float, num_queries: int
) -> tuple[float, float, float, float, float, float]:
    """Calculate processing capacity metrics.

    Args:
        manual_time: Total manual processing time in minutes.
        sequential_time: Total sequential processing time in minutes.
        parallel_time: Total parallel processing time in minutes.
        num_queries: Number of queries processed.

    Returns:
        Tuple of (manual_per_hour, sequential_per_hour, parallel_per_hour,
                manual_per_day, sequential_per_day, parallel_per_day).
    """
    # Queries per hour
    manual_per_hour = (num_queries / manual_time) * 60 if manual_time > 0 else 0
    sequential_per_hour = (num_queries / sequential_time) * 60 if sequential_time > 0 else 0
    parallel_per_hour = (num_queries / parallel_time) * 60 if parallel_time > 0 else 0

    # Queries per day (8-hour workday)
    manual_per_day = manual_per_hour * 8
    sequential_per_day = sequential_per_hour * 8
    parallel_per_day = parallel_per_hour * 8

    return (
        manual_per_hour,
        sequential_per_hour,
        parallel_per_hour,
        manual_per_day,
        sequential_per_day,
        parallel_per_day,
    )


def run_simulation(params: SimulationParams) -> SimulationResult:
    """Run a complete simulation with the given parameters.

    Args:
        params: Simulation parameters.

    Returns:
        SimulationResult object with all calculated metrics.
    """
    # Calculate time metrics
    manual_time = calculate_manual_analysis_time(
        params.num_queries, params.avg_human_analysis_time_per_query
    )

    sequential_time = calculate_sequential_processing_time(
        params.num_queries, params.avg_api_time_per_query, params.api_rate_limit
    )

    worker_times = calculate_worker_processing_times(
        params.num_queries,
        params.avg_api_time_per_query,
        params.max_workers,
        params.api_rate_limit,
    )

    parallel_time = min(worker_times) if worker_times else float("inf")

    # Calculate cost metrics
    manual_cost = calculate_manual_analysis_cost(
        params.num_queries, params.avg_human_analysis_time_per_query, params.hourly_salary
    )

    api_input_cost, api_output_cost, api_total_cost = calculate_api_costs(
        params.num_queries,
        params.avg_input_prompt_ntokens,
        params.avg_output_prompt_ntokens,
        params.api_input_price,
        params.api_output_price,
    )

    # Calculate processing capacity
    (
        manual_per_hour,
        sequential_per_hour,
        parallel_per_hour,
        manual_per_day,
        sequential_per_day,
        parallel_per_day,
    ) = calculate_processing_capacity(
        manual_time, sequential_time, parallel_time, params.num_queries
    )

    # Calculate number of queries
    num_queries = params.num_queries

    return SimulationResult(
        manual_analysis_time=manual_time,
        sequential_processing_time=sequential_time,
        parallel_processing_time=parallel_time,
        manual_analysis_cost=manual_cost,
        api_total_cost=api_total_cost,
        api_input_cost=api_input_cost,
        api_output_cost=api_output_cost,
        manual_per_hour=manual_per_hour,
        sequential_per_hour=sequential_per_hour,
        parallel_per_hour=parallel_per_hour,
        manual_per_day=manual_per_day,
        sequential_per_day=sequential_per_day,
        parallel_per_day=parallel_per_day,
        worker_times=worker_times,
        avg_human_analysis_time_per_query=params.avg_human_analysis_time_per_query,
        num_queries=num_queries,
        params=params,
    )

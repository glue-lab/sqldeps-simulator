"""Data models for simulation parameters and results.

This module defines the core data structures used by the SQLDeps Simulator
for representing simulation parameters and results.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class SimulationParams:
    """Parameters for SQL dependency simulation.

    Attributes:
        num_queries: Number of SQL queries to analyze.
        avg_human_analysis_time_per_query: Average time in minutes for human analysis per query.
        avg_api_time_per_query: Average time in minutes for API processing per query.
        monthly_query_workload: Number of queries to process per month for ROI calculation.
        hourly_salary: Hourly salary in USD for a data engineer.
        api_input_price: Price per 1M tokens for API input in USD.
        api_output_price: Price per 1M tokens for API output in USD.
        api_rate_limit: Requests per minute limit. None means no limit.
        max_workers: Maximum number of parallel workers.
        avg_input_prompt_ntokens: Average number of input tokens per query.
        avg_output_prompt_ntokens: Average number of output tokens per query.
    """

    # Workload parameters
    num_queries: int = 200
    avg_human_analysis_time_per_query: float = 5.0
    avg_api_time_per_query: float = 0.5
    monthly_query_workload: int = 1000

    # Cost parameters
    hourly_salary: float = 60.0
    api_input_price: float = 2.5
    api_output_price: float = 10.0

    # API/Technical parameters
    api_rate_limit: int | None = 40
    max_workers: int = 10
    avg_input_prompt_ntokens: int = 5000
    avg_output_prompt_ntokens: int = 500

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SimulationResult:
    """Results from SQL dependency simulation.

    Attributes:
        manual_analysis_time: Total time in minutes for manual analysis.
        sequential_processing_time: Total time in minutes for sequential API processing.
        parallel_processing_time: Total time in minutes for parallel API processing.
        manual_analysis_cost: Total cost in USD for manual analysis.
        api_total_cost: Total cost in USD for API processing.
        api_input_cost: Cost in USD for API input tokens.
        api_output_cost: Cost in USD for API output tokens.
        manual_per_hour: Manual analysis capacity in queries per hour.
        sequential_per_hour: Sequential API processing capacity in queries per hour.
        parallel_per_hour: Parallel API processing capacity in queries per hour.
        manual_per_day: Manual analysis capacity in queries per 8-hour day.
        sequential_per_day: Sequential API processing capacity in queries per 8-hour day.
        parallel_per_day: Parallel API processing capacity in queries per 8-hour day.
        worker_times: List of processing times for different worker counts.
        avg_human_analysis_time_per_query: Average time in minutes for human analysis per query.
        num_queries: Number of SQL queries analyzed.
        params: Parameters used in the simulation.
        time_saved_vs_manual: Dictionary of time saved compared to manual analysis.
        time_saved_percentage_vs_manual: Dictionary of percentage time saved compared to manual.
        cost_saved_vs_manual: Cost saved compared to manual analysis in USD.
        cost_saved_percentage_vs_manual: Percentage cost saved compared to manual analysis.
        cost_per_query: Average cost per query in USD.
    """

    # Time metrics (minutes)
    manual_analysis_time: float
    sequential_processing_time: float
    parallel_processing_time: float

    # Cost metrics (USD)
    manual_analysis_cost: float
    api_total_cost: float
    api_input_cost: float
    api_output_cost: float

    # Processing capacity
    manual_per_hour: float
    sequential_per_hour: float
    parallel_per_hour: float
    manual_per_day: float
    sequential_per_day: float
    parallel_per_day: float

    # Workers data for parallelism chart
    worker_times: list[float]

    # Reference metrics
    avg_human_analysis_time_per_query: float
    num_queries: int

    # Parameters used in the simulation
    params: SimulationParams

    # Derived metrics (calculated at post_init)
    time_saved_vs_manual: dict[str, float] = field(default_factory=dict)
    time_saved_percentage_vs_manual: dict[str, float] = field(default_factory=dict)
    cost_saved_vs_manual: float = 0.0
    cost_saved_percentage_vs_manual: float = 0.0
    cost_per_query: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self.time_saved_vs_manual = {
            "sequential": self.manual_analysis_time - self.sequential_processing_time,
            "parallel": self.manual_analysis_time - self.parallel_processing_time,
        }

        if self.manual_analysis_time > 0:
            self.time_saved_percentage_vs_manual = {
                "sequential": 100
                * (self.manual_analysis_time - self.sequential_processing_time)
                / self.manual_analysis_time,
                "parallel": 100
                * (self.manual_analysis_time - self.parallel_processing_time)
                / self.manual_analysis_time,
            }

        self.cost_saved_vs_manual = self.manual_analysis_cost - self.api_total_cost

        if self.manual_analysis_cost > 0:
            self.cost_saved_percentage_vs_manual = (
                100 * (self.manual_analysis_cost - self.api_total_cost) / self.manual_analysis_cost
            )

        self.cost_per_query = self.api_total_cost / self.num_queries if self.num_queries > 0 else 0

    def to_dict(self) -> dict:
        """Convert to dictionary, including nested params."""
        result = asdict(self)
        return result

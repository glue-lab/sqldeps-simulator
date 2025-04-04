"""Chart generation functions for visualizing simulation results.

This module provides functions for creating interactive visualizations
that compare manual SQL dependency analysis versus automated approaches
using SQLDeps in terms of time, cost, capacity, and ROI.
"""

import plotly.graph_objects as go

from .models import SimulationParams, SimulationResult
from .utils import custom_round


def create_processing_time_chart(result: SimulationResult) -> go.Figure:
    """Create a bar chart comparing processing times.

    Args:
        result: Simulation result.

    Returns:
        Plotly figure displaying processing time comparison between
        manual analysis, sequential processing, and parallel processing.
    """
    fig = go.Figure()

    # Add bars
    n_workers = result.params.max_workers
    fig.add_trace(
        go.Bar(
            x=["Manual Analysis", "SQLDeps (Sequential)", f"SQLDeps ({n_workers} workers)"],
            y=[
                result.manual_analysis_time,
                result.sequential_processing_time,
                result.parallel_processing_time,
            ],
            text=[
                f"{custom_round(result.manual_analysis_time)}m",
                f"{custom_round(result.sequential_processing_time)}m",
                f"{custom_round(result.parallel_processing_time)}m",
            ],
            textposition="auto",
            marker_color=[
                "rgba(255, 99, 132, 0.8)",
                "rgba(54, 162, 235, 0.8)",
                "rgba(75, 192, 192, 0.8)",
            ],
        )
    )

    # Update layout
    saved_minutes = result.manual_analysis_time - result.parallel_processing_time
    saved_percentage = result.time_saved_percentage_vs_manual["parallel"]

    fig.update_layout(
        title="Time to Process SQL Queries",
        xaxis_title="Method",
        yaxis_title="Total Time (minutes)",
        yaxis=dict(gridcolor="rgba(230, 230, 230, 0.8)"),
        annotations=[
            dict(
                x=0.5,
                y=-0.28,
                showarrow=False,
                text=(
                    f"Time saved: {custom_round(saved_minutes)} minutes "
                    f"({custom_round(saved_percentage)}%)"
                ),
                xref="paper",
                yref="paper",
                font=dict(size=14),
            )
        ],
    )

    return fig


def create_parallelism_chart(params: SimulationParams, result: SimulationResult) -> go.Figure:
    """Create a line chart showing impact of parallelism.

    Args:
        params: Simulation parameters.
        result: Simulation result.

    Returns:
        Plotly figure displaying the relationship between number of workers
        and processing time, with rate limit constraints visualized.
    """
    fig = go.Figure()

    # Worker counts (x-axis)
    workers = list(range(1, params.max_workers + 1))

    # Add the processing time line
    fig.add_trace(
        go.Scatter(
            x=workers,
            y=result.worker_times,
            mode="lines+markers",
            name="Processing Time",
            marker=dict(size=8),
            line=dict(width=3, color="rgba(54, 162, 235, 0.8)"),
        )
    )

    # Highlight optimal number of workers
    optimal_workers = result.worker_times.index(min(result.worker_times)) + 1
    fig.add_trace(
        go.Scatter(
            x=[optimal_workers],
            y=[min(result.worker_times)],
            mode="markers",
            marker=dict(color="green", size=15, line=dict(width=2, color="darkgreen")),
            name=f"Optimal: {optimal_workers} workers",
        )
    )

    # Add rate limit line
    if params.api_rate_limit is not None:
        fig.add_trace(
            go.Scatter(
                x=[1, params.max_workers],
                y=[
                    params.num_queries / params.api_rate_limit,
                    params.num_queries / params.api_rate_limit,
                ],
                mode="lines",
                name=f"Rate Limit: {params.api_rate_limit} req/min",
                line=dict(dash="dash", color="red", width=2),
            )
        )

    # Update layout
    fig.update_layout(
        title="Impact of Parallelism with Rate Limiting",
        xaxis_title="Number of Workers",
        yaxis_title="Total Time (minutes)",
        xaxis=dict(tickmode="array", tickvals=workers, gridcolor="rgba(230, 230, 230, 0.8)"),
        yaxis=dict(gridcolor="rgba(230, 230, 230, 0.8)"),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.8)"),
        hovermode="closest",
    )

    # Add optimal point annotation
    fig.add_annotation(
        x=optimal_workers,
        y=min(result.worker_times),
        text=f"Optimal: {optimal_workers} workers<br>{min(result.worker_times):.1f}m",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="green",
        ax=50,
        ay=-40,
    )

    return fig


def create_processing_capacity_chart(result: SimulationResult) -> go.Figure:
    """Create a bar chart comparing processing capacities.

    Args:
        result: Simulation result.

    Returns:
        Plotly figure displaying processing capacity comparison in
        queries per hour and per day for different processing methods.
    """
    fig = go.Figure()

    # Add hourly capacity bars
    fig.add_trace(
        go.Bar(
            x=["Manual Analysis", "SQLDeps Sequential", "SQLDeps (10 workers)"],
            y=[
                result.manual_per_hour,
                result.sequential_per_hour,
                result.parallel_per_hour,
            ],
            text=[
                f"{result.manual_per_hour:.0f}",
                f"{result.sequential_per_hour:.0f}",
                f"{result.parallel_per_hour:.0f}",
            ],
            textposition="auto",
            name="Per Hour",
            marker_color="rgba(54, 162, 235, 0.8)",
        )
    )

    # Add daily capacity bars
    fig.add_trace(
        go.Bar(
            x=["Manual Analysis", "SQLDeps Sequential", "SQLDeps (10 workers)"],
            y=[
                result.manual_per_day,
                result.sequential_per_day,
                result.parallel_per_day,
            ],
            text=[
                f"{result.manual_per_day:.0f}",
                f"{result.sequential_per_day:.0f}",
                f"{result.parallel_per_day:.0f}",
            ],
            textposition="auto",
            name="Per Day (8h)",
            marker_color="rgba(255, 159, 64, 0.8)",
        )
    )

    # Update layout
    fig.update_layout(
        title="Processing Capacity Comparison",
        xaxis_title="Processing Method",
        yaxis_title="Processed Queries",
        barmode="group",
        legend=dict(x=0.01, y=0.99),
        yaxis=dict(gridcolor="rgba(230, 230, 230, 0.8)"),
    )

    return fig


def create_cost_comparison_chart(result: SimulationResult, params: SimulationParams) -> go.Figure:
    """Create a bar chart comparing costs.

    Args:
        result: Simulation result.
        params: Simulation parameters.

    Returns:
        Plotly figure displaying cost comparison between manual analysis,
        total API cost, and breakdown of API input and output costs.
    """
    fig = go.Figure()

    # Add bars
    fig.add_trace(
        go.Bar(
            x=[
                "Manual Analysis",
                "Total API Cost",
                "API Input Cost",
                "API Output Cost",
            ],
            y=[
                result.manual_analysis_cost,
                result.api_total_cost,
                result.api_input_cost,
                result.api_output_cost,
            ],
            text=[
                f"${result.manual_analysis_cost:,.2f}",
                f"${result.api_total_cost:,.2f}",
                f"${result.api_input_cost:,.2f}",
                f"${result.api_output_cost:,.2f}",
            ],
            textposition="auto",
            marker_color=[
                "rgba(255, 99, 132, 0.8)",
                "rgba(54, 162, 235, 0.8)",
                "rgba(75, 192, 192, 0.8)",
                "rgba(153, 102, 255, 0.8)",
            ],
        )
    )

    # Update layout
    cost_saved = result.cost_saved_vs_manual
    cost_saved_percentage = result.cost_saved_percentage_vs_manual
    cost_per_query = result.cost_per_query

    # Change the rate_limit message based on whether rate_limit is None
    rate_limit_text = (
        f"rate-limited at {params.api_rate_limit} req/min"
        if params.api_rate_limit is not None
        else "not rate-limited"
    )

    fig.update_layout(
        title="Cost Comparison",
        xaxis_title="Category",
        yaxis_title="Cost (USD)",
        yaxis=dict(gridcolor="rgba(230, 230, 230, 0.8)"),
        annotations=[
            dict(
                x=0.5,
                y=-0.3,
                showarrow=False,
                text=(
                    f"Note: SQLDeps with {params.max_workers} workers is "
                    f"{rate_limit_text}. "
                    f"Cost saved: ${cost_saved:.2f} ({cost_saved_percentage:.1f}%) or "
                    f"${cost_per_query:.2f}/query"
                ),
                xref="paper",
                yref="paper",
                font=dict(size=14),
            )
        ],
    )

    return fig


def create_roi_chart(result: SimulationResult, params: SimulationParams) -> go.Figure:
    """Create a chart showing return on investment over time.

    This updated version compares costs for processing the SAME number of queries
    per month, rather than comparing what can be processed in the same time period.

    Args:
        result: Simulation result.
        params: Simulation parameters.

    Returns:
        Plotly figure displaying cumulative costs over time and
        identifying the break-even point if applicable.
    """
    # Use the monthly_query_workload from params
    monthly_query_workload = params.monthly_query_workload

    # Calculate monthly costs for processing the SAME number of queries
    manual_cost_per_month = (
        monthly_query_workload * params.avg_human_analysis_time_per_query / 60
    ) * params.hourly_salary

    sqldeps_cost_per_month = (monthly_query_workload * result.api_total_cost) / params.num_queries

    # Calculate cumulative costs over 12 months
    months = list(range(1, 13))
    manual_cumulative_costs = [month * manual_cost_per_month for month in months]
    sqldeps_cumulative_costs = [month * sqldeps_cost_per_month for month in months]

    # Calculate savings over time (manual cost - sqldeps cost)
    savings = [
        m - s for m, s in zip(manual_cumulative_costs, sqldeps_cumulative_costs, strict=False)
    ]

    # Create figure
    fig = go.Figure()

    # Add traces
    fig.add_trace(
        go.Scatter(
            x=months,
            y=manual_cumulative_costs,
            mode="lines+markers",
            name="Manual Analysis",
            line=dict(width=3, color="rgba(255, 99, 132, 0.8)"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=months,
            y=sqldeps_cumulative_costs,
            mode="lines+markers",
            name="SQLDeps",
            line=dict(width=3, color="rgba(54, 162, 235, 0.8)"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=months,
            y=savings,
            mode="lines+markers",
            name="Savings",
            line=dict(width=3, color="rgba(75, 192, 192, 0.8)"),
        )
    )

    # Find break-even point (if any)
    break_even_month = None
    for i, saving in enumerate(savings):
        if saving > 0:
            # Linear interpolation to get more precise break-even point
            if i > 0:
                prev_saving = savings[i - 1]
                if prev_saving <= 0:
                    # Interpolate
                    t = -prev_saving / (saving - prev_saving)
                    break_even_month = i + t
            else:
                break_even_month = i + 1
            break

    # Calculate annual savings
    annual_savings = savings[-1] if savings else 0
    monthly_savings = annual_savings / 12 if annual_savings else 0

    # Update layout
    fig.update_layout(
        title=(
            f"ROI: Cumulative Cost Over Time (Processing {monthly_query_workload:,} queries/month)"
        ),
        xaxis_title="Months",
        yaxis_title="Cumulative Cost (USD)",
        xaxis=dict(tickmode="array", tickvals=months, gridcolor="rgba(230, 230, 230, 0.8)"),
        yaxis=dict(gridcolor="rgba(230, 230, 230, 0.8)"),
        legend=dict(x=0.01, y=0.99),
        annotations=[
            dict(
                x=0.5,
                y=-0.28,
                showarrow=False,
                text=(
                    f"Annual savings: ${custom_round(annual_savings)} "
                    f"(${custom_round(monthly_savings)}/month) when processing "
                    f"{monthly_query_workload:,} queries/month"
                ),
                xref="paper",
                yref="paper",
                font=dict(size=14),
            )
        ],
    )

    # Add break-even annotation if applicable
    if break_even_month:
        fig.add_annotation(
            x=break_even_month,
            y=0,
            text=f"Break-even at {break_even_month:.1f} months",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="green",
            ax=0,
            ay=-40,
        )

    return fig

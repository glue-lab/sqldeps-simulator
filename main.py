"""SQLDeps Time & Cost Analyzer for evaluating dependency mapping benefits.

This module provides a Streamlit-based web application for simulating and
visualizing the benefits of using SQLDeps for SQL dependency analysis compared
to manual approaches. It includes interactive visualizations of time savings,
cost efficiency, and return on investment metrics.

The application allows users to:
1. Configure simulation parameters through an interactive sidebar
2. View key performance metrics in an easily digestible dashboard
3. Explore detailed visualizations across multiple dimensions
4. Access tabular data for more in-depth analysis
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.models import SimulationParams, SimulationResult
from src.simulator import run_simulation
from src.utils import custom_round
from src.visualization import (
    create_cost_comparison_chart,
    create_parallelism_chart,
    create_processing_capacity_chart,
    create_processing_time_chart,
    create_roi_chart,
)


def setup_page():
    """Configure the Streamlit page with title, icon and descriptive content.

    Sets the page configuration, adds the title header, and provides
    an explanatory introduction describing SQLDeps and the simulator's purpose.
    """
    st.set_page_config(
        page_title="SQLDeps Simulator",
        page_icon=Path("assets/images/icon.png"),
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("SQLDeps Time & Cost Analyzer")

    st.markdown("""
    ### Streamline SQL Dependency Mapping with Automated Intelligence

    SQLDeps leverages Large Language Models (LLMs) to automatically identify table and column
    dependencies in your SQL codebase, significantly **reducing analysis time** while
    **enhancing accuracy**, **increasing processing capacity**, and **lowering operational
    costs**—ultimately freeing engineers to focus on high-value strategic tasks.

    This simulator allows you to:
    - **Quantify time savings** compared to manual SQL dependency analysis
    - **Measure throughput capacity** to process more SQL queries per day
    - **Assess cost benefits** based on your team's actual parameters

    Adjust sidebar parameters to see how SQLDeps can streamline your database operations.
    """)


def sidebar_parameters() -> SimulationParams:
    """Create sidebar inputs for simulation parameters.

    Creates interactive input widgets in the Streamlit sidebar for users
    to configure simulation parameters across three categories:
    1. Workload Parameters - query volumes and processing times
    2. Cost Parameters - salary and API pricing factors
    3. API Parameters - technical constraints and token estimates

    Returns:
        SimulationParams object with user-specified values.
    """
    # Setup sidebar header and logo
    st.sidebar.image(Path("assets/images/sqldeps_gray.png"))
    st.sidebar.header("Simulation Parameters")

    # Workload parameter section
    with st.sidebar.expander("Workload Parameters", expanded=True):
        num_queries = st.number_input(
            "Number of SQL queries to analyze",
            min_value=1,
            max_value=10_000,
            value=300,
            step=50,
            help="Total number of SQL queries to process in the simulation",
        )

        avg_human_time = st.number_input(
            "Average human analysis time per query (minutes)",
            min_value=1.0,
            max_value=60.0,
            value=5.0,
            step=0.5,
            help="Average time a data engineer spends analyzing a single SQL query",
        )

        avg_api_time = st.number_input(
            "Average API processing time per query (minutes)",
            min_value=0.1,
            max_value=10.0,
            value=0.25,
            step=0.1,
            help="Average time for SQLDeps (API) to process a single SQL query",
        )

        monthly_query_workload = st.number_input(
            "Monthly query workload",
            min_value=10,
            max_value=10_000,
            value=1_000,
            step=100,
            help="Number of queries to process per month for ROI calculation",
        )

    # Cost parameter section
    with st.sidebar.expander("Cost Parameters", expanded=True):
        hourly_salary = st.number_input(
            "Data Engineer hourly salary (USD)",
            min_value=10.0,
            max_value=200.0,
            value=60.0,
            step=5.0,
            help="Hourly salary for a data engineer performing manual analysis",
        )

        api_input_price = st.number_input(
            "API input price per 1M tokens (USD)",
            min_value=0.1,
            max_value=50.0,
            value=2.5,
            step=0.1,
            help="Cost per 1M tokens for API input (e.g., GPT-4o input cost)",
        )

        api_output_price = st.number_input(
            "API output price per 1M tokens (USD)",
            min_value=0.1,
            max_value=50.0,
            value=10.0,
            step=0.1,
            help="Cost per 1M tokens for API output (e.g., GPT-4o output cost)",
        )

    # API parameter section
    with st.sidebar.expander("API Parameters", expanded=True):
        api_rate_limit = st.number_input(
            "API rate limit (requests per minute)",
            min_value=1,
            max_value=1_000,
            value=30,
            step=10,
            help="Maximum API requests allowed per minute",
        )

        max_workers = st.number_input(
            "Maximum number of parallel workers",
            min_value=1,
            max_value=64,
            value=12,
            step=1,
            help="Maximum number of parallel workers for distributed processing",
        )

        avg_input_tokens = st.number_input(
            "Average input prompt tokens per query",
            min_value=100,
            max_value=10_000,
            value=1800,
            step=100,
            help="Average number of tokens in the input prompt for each query",
        )

        avg_output_tokens = st.number_input(
            "Average output tokens per query",
            min_value=100,
            max_value=1000,
            value=300,
            step=100,
            help="Average number of tokens in the output response for each query",
        )

    st.sidebar.markdown("---")

    # GitHub call-to-action
    st.sidebar.markdown(
        """
    <div style="text-align: center">
        <h3>SQLDeps is 100% open-source</h3>
        <a href="https://github.com/glue-lab/sqldeps" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-Star_SQLDeps-BF050C?logo=github&style=for-the-badge" alt="Star on GitHub">
        </a>
        <p style="margin-top: 10px; font-size: 0.9em;">✨ Support us by starring the project!</p>
    </div>
    """,  # noqa: E501
        unsafe_allow_html=True,
    )

    # Create and return SimulationParams with all collected values
    return SimulationParams(
        num_queries=num_queries,
        hourly_salary=hourly_salary,
        avg_human_analysis_time_per_query=avg_human_time,
        avg_api_time_per_query=avg_api_time,
        api_rate_limit=api_rate_limit,
        max_workers=max_workers,
        api_input_price=api_input_price,
        api_output_price=api_output_price,
        avg_input_prompt_ntokens=avg_input_tokens,
        avg_output_prompt_ntokens=avg_output_tokens,
        monthly_query_workload=monthly_query_workload,
    )


def display_metrics(result: SimulationResult):
    """Display key metrics as a dashboard with visual indicators.

    Shows the key benefits of SQLDeps in a metrics dashboard with visual
    indicators of improvement or regression compared to manual analysis.
    For each metric, appropriate comparisons and delta indicators are
    provided to highlight the relative performance.

    Args:
        result: SimulationResult containing calculated metrics.
    """
    # Create a container with a light blue background for key metrics
    metrics_container = st.container()
    metrics_container.markdown(
        """
    <div style="padding: 10px; background-color: #f0f8ff; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="text-align: center;">Key Benefits of SQLDeps</h3>
    </div>
    """,  # noqa: E501
        unsafe_allow_html=True,
    )

    # Create metrics in multiple columns
    col1, col2, col3, col4 = st.columns(4)

    # Productivity Boost metric (Time Saved)
    with col1:
        if round(result.time_saved_vs_manual["parallel"], 2) == 0:
            # SQLDeps takes the same time than manual analysis
            delta_text = None
            delta_color = "off"
        elif result.time_saved_vs_manual["parallel"] > 0:
            # SQLDeps is faster - show how much faster
            speed_improvement = result.manual_analysis_time / result.parallel_processing_time
            delta_text = f"{custom_round(speed_improvement)}x faster"
            delta_color = "normal"
        else:
            # SQLDeps is slower - show how much slower
            speed_ratio = result.parallel_processing_time / result.manual_analysis_time
            delta_text = f"{custom_round(speed_ratio)}x slower"
            delta_color = "inverse"

        st.metric(
            label="⏱️ Productivity Boost",
            value=f"{custom_round(result.time_saved_vs_manual['parallel'])} min",
            delta=delta_text,
            delta_color=delta_color,
        )

    # Daily Throughput metric
    with col2:
        # Show daily throughput comparison more clearly
        throughput_increase = result.parallel_per_day - result.manual_per_day

        if throughput_increase > 0:
            delta_text = f"+{custom_round(throughput_increase)} queries/day"
            delta_color = "normal"
        else:
            delta_text = f"{custom_round(throughput_increase)} queries/day"
            delta_color = "normal"

        st.metric(
            label="📊 Daily Throughput",
            value=f"{result.parallel_per_day:.0f} vs {result.manual_per_day:.0f}",
            delta=delta_text,
            delta_color=delta_color,
        )

    # Cost Savings metric
    with col3:
        if abs(round(result.cost_saved_vs_manual, 2)) == 0:
            # Same costs between manual and API
            delta_text = None
            delta_color = "off"
        elif result.cost_saved_vs_manual > 0:
            # There are savings - show how much more cost-effective
            cost_effectiveness = result.manual_analysis_cost / result.api_total_cost
            delta_text = f"{custom_round(cost_effectiveness)}x more cost-effective"
            delta_color = "normal"
        else:
            # SQLDeps costs more - show how much more expensive
            cost_ratio = result.api_total_cost / result.manual_analysis_cost
            delta_text = f"{custom_round(cost_ratio)}x more expensive"
            delta_color = "inverse"

        st.metric(
            label="💰 Cost Savings",
            value=f"${result.cost_saved_vs_manual:.2f}",
            delta=delta_text,
            delta_color=delta_color,
        )

    # Cost Per Query metric
    with col4:
        manual_cost_per_query = result.manual_analysis_cost / result.num_queries
        api_cost_per_query = result.cost_per_query

        # Calculate the cost difference percentage
        if manual_cost_per_query > 0:
            cost_diff_pct = (
                (manual_cost_per_query - api_cost_per_query) / manual_cost_per_query * 100
            )
            delta_text = (
                f"{custom_round(cost_diff_pct)}% cheaper"
                if cost_diff_pct > 0
                else f"Costs {custom_round(-cost_diff_pct)}% more"
            )
            delta_color = "normal" if cost_diff_pct > 0 else "inverse"
        else:
            delta_text = None
            delta_color = "off"

        st.metric(
            label="💸 Cost Per Query",
            value=f"${api_cost_per_query:.2f}",
            delta=delta_text,
            delta_color=delta_color,
            help="Average cost to process one SQL query with SQLDeps",
        )


def display_charts(params: SimulationParams, result: SimulationResult):
    """Display visualization charts in organized tabs with contextual insights.

    Creates a tabbed interface with various charts for analyzing different
    aspects of the simulation results. Each tab contains a chart and
    additional contextual information to help interpret the results.
    The tabs include:
    - Time Analysis: Compares processing times between methods
    - Capacity Analysis: Shows throughput capabilities
    - Cost Analysis: Compares costs between methods
    - ROI Analysis: Shows return on investment over time
    - Parallelism Impact: Shows how parallelism affects processing time

    Args:
        params: SimulationParams used to generate the results.
        result: SimulationResult containing calculated metrics.
    """
    # Create tabs for different chart categories
    tab_titles = [
        "Time Analysis",
        "Capacity Analysis",
        "Cost Analysis",
        "ROI Analysis",
        "Parallelism Impact",
    ]
    tabs = st.tabs(tab_titles)

    # Time Analysis tab
    with tabs[0]:  # Time Analysis
        st.subheader("Processing Time Comparison")
        st.plotly_chart(create_processing_time_chart(result), use_container_width=True)

        # Add contextual insights
        workers_text = f"{params.max_workers} workers" if params.max_workers > 1 else "1 worker"

        if result.time_saved_vs_manual["parallel"] > 0:
            time_saved_pct = result.time_saved_percentage_vs_manual["parallel"]
            time_saved_per_month = (
                params.monthly_query_workload
                * result.time_saved_vs_manual["parallel"]
                / params.num_queries
            )
            st.markdown("### 🚀 Key Insights")
            st.success(f"""
            **SQLDeps reduces analysis time by {custom_round(time_saved_pct)}%**
            compared to manual analysis when using {workers_text}.

            - For your {custom_round(params.num_queries)} queries, this means saving
              approximately **{custom_round(result.time_saved_vs_manual["parallel"] / 60)} hours**
              of engineering time.
            - For your {custom_round(params.monthly_query_workload)} queries as monthly workload,
            this means saving approximately **{custom_round(time_saved_per_month / 60)} hours** of
            engineering time.
            """)
        else:
            st.markdown("### ⚠️ Key Insights")
            st.warning("""
            In this scenario, manual analysis is faster than SQLDeps.

            Try adjusting your parameters to find scenarios where SQLDeps provides time benefits
            for your workflow.
            """)

    # Capacity Analysis tab
    with tabs[1]:  # Capacity Analysis
        st.subheader("Processing Capacity")
        st.plotly_chart(create_processing_capacity_chart(result), use_container_width=True)

        # Display additional capacity insights
        st.markdown("### Capacity Insights")

        # Calculate the actual speedup factor correctly
        speedup_factor = (
            result.parallel_per_day / result.manual_per_day if result.manual_per_day > 0 else 0
        )

        # Determine comparative language based on actual values
        if speedup_factor > 1:
            comparative = f"**{custom_round(speedup_factor)}x faster** than manual analysis"
        elif speedup_factor < 1:
            slowdown_factor = 1 / speedup_factor if speedup_factor > 0 else 0
            comparative = f"**{custom_round(slowdown_factor)}x slower** than manual analysis"
        else:
            comparative = "**equally fast** as manual analysis"

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            #### Manual Analysis
            Can process approximately **{result.manual_per_hour:.0f} queries per hour** or
            **{result.manual_per_day:.0f} queries per workday** (8 hours).

            Monthly capacity (160 hours): **{custom_round(result.manual_per_hour * 160)} queries**
            """)

        with col2:
            workers_text = (
                f"with {params.max_workers} workers"
                if params.max_workers > 1
                else "in sequential mode"
            )
            st.success(f"""
            #### SQLDeps {workers_text}
            Can process approximately **{custom_round(result.parallel_per_hour)} queries per hour**
            or **{custom_round(result.parallel_per_day)} queries per workday** (8 hours).
            This throughput is {comparative}.

            Monthly capacity (160 hours):
            **{custom_round(result.parallel_per_hour * 160)} queries**
            """)

    # Cost Analysis tab
    with tabs[2]:  # Cost Analysis
        st.subheader("Cost Comparison")
        st.plotly_chart(create_cost_comparison_chart(result, params), use_container_width=True)

        # Display additional cost insights
        st.markdown("### Cost Insights")

        col1, col2 = st.columns(2)

        # Calculate cost per query and total cost for manual analysis
        manual_cost_per_query = result.manual_analysis_cost / result.num_queries

        with col1:
            st.info(f"""
            #### Manual Analysis Cost
            - Cost per query: **${custom_round(manual_cost_per_query)}**
            - Total cost for {params.num_queries} queries:
            **${custom_round(result.manual_analysis_cost)}**
            - Total cost for {params.monthly_query_workload} queries:
            **${custom_round(result.manual_analysis_cost * manual_cost_per_query)}**
            - Based on hourly rate: **${custom_round(params.hourly_salary)}**
            """)

        with col2:
            if result.cost_saved_vs_manual > 0:
                cost_ratio = result.manual_analysis_cost / result.api_total_cost
                st.success(f"""
                #### SQLDeps Cost
                - Cost per query: **${custom_round(result.cost_per_query)}**
                - Total cost for {params.num_queries} queries:
                **${custom_round(result.api_total_cost)}**
                - Total cost for {params.monthly_query_workload} queries:
                **${custom_round(result.cost_per_query * params.monthly_query_workload)}**
                - **{custom_round(cost_ratio)}x more cost-effective** than manual analysis
                """)
            else:
                st.warning(f"""
                #### SQLDeps Cost
                - Cost per query: **${result.cost_per_query:.2f}**
                - Total cost for {params.num_queries} queries: **${result.api_total_cost:.2f}**
                - In this scenario, SQLDeps costs more than manual analysis
                - Try adjusting parameters to find cost-effective configurations
                """)

    # ROI Analysis tab
    with tabs[3]:  # ROI Analysis
        st.subheader("Return on Investment Over Time")
        st.plotly_chart(create_roi_chart(result, params), use_container_width=True)

        # Display ROI insights
        # Calculate costs for the specified monthly workload
        monthly_query_workload = params.monthly_query_workload
        manual_monthly_cost = (
            monthly_query_workload * params.avg_human_analysis_time_per_query / 60
        ) * params.hourly_salary

        sqldeps_monthly_cost = (
            monthly_query_workload * result.api_total_cost
        ) / params.num_queries

        monthly_savings = manual_monthly_cost - sqldeps_monthly_cost
        annual_savings = monthly_savings * 12

        st.markdown("### ROI Projection")

        # Calculate time savings
        manual_monthly_time = monthly_query_workload * params.avg_human_analysis_time_per_query
        sqldeps_monthly_time = monthly_query_workload * (
            result.parallel_processing_time / params.num_queries
        )
        monthly_time_saved = (manual_monthly_time - sqldeps_monthly_time) / 60  # Convert to hours

        # Only show positive savings message if there are actually savings
        if monthly_savings > 0:
            st.success(f"""
            Based on processing **{monthly_query_workload:,} queries per month**:

            #### Financial Impact
            - Monthly savings: **${custom_round(monthly_savings)}**
            - Annual savings: **${custom_round(annual_savings)}**
            - 5-year savings: **${custom_round(annual_savings * 5)}**

            #### Time Impact
            - Monthly time saved: **{custom_round(monthly_time_saved)} hours**
            - Annual time saved: **{custom_round(monthly_time_saved * 12)} hours**

            These resources can be redirected to higher-value activities like performance
            optimization, feature development, or data quality improvements.
            """)
        else:
            st.warning(f"""
            Based on processing **{monthly_query_workload:,} queries per month**, manual analysis
            would be **${-monthly_savings:.2f} cheaper per month** than using SQLDeps in this
            specific scenario.

            Consider adjusting these parameters to find optimal conditions for SQLDeps adoption:
            - Increase the number of queries processed
            - Choose a more cost-effective LLM provider
            - Use a higher number of workers for parallelization
            - Evaluate non-financial benefits like:
              - Reduced errors in schema changes
              - Improved accuracy in dependency tracking
              - Better documentation for database structures
            """)

    # Parallelism Impact tab
    with tabs[4]:  # Parallelism Impact
        st.subheader("Impact of Parallelism with Rate Limiting")

        # Show chart only if max_workers > 1, otherwise show informative message
        if params.max_workers > 1:
            st.plotly_chart(create_parallelism_chart(params, result), use_container_width=True)

            # Add explanatory text about parallelism
            st.markdown("""
            ### Parallelism Insights

            This chart shows how increasing the number of workers affects the total
            processing time. Some key points to understand:

            - **Theoretical speedup**:
                In ideal conditions, adding more workers should reduce processing time
                proportionally
            - **Rate limiting**:
                If the API has a rate limit, adding more workers may not improve performance beyond
                 a certain point
            - **Optimal point**:
                The green marker shows the optimal number of workers for your current parameters
            - **Diminishing returns**:
                Notice how the curve flattens when adding more workers beyond the optimal point
            """)

            # Show details about the optimal workers configuration
            optimal_workers = result.worker_times.index(min(result.worker_times)) + 1
            optimal_time = min(result.worker_times)

            col1, col2 = st.columns(2)
            with col1:
                throughput = params.num_queries / optimal_time * 60
                st.info(f"""
                - **Optimal configuration**: {optimal_workers} workers
                - **Processing time**: {custom_round(optimal_time)} minutes
                - **Throughput**: {custom_round(throughput)} queries per hour
                """)

            with col2:
                if params.api_rate_limit > 0:
                    st.warning(
                        f"**Rate limit impact**: Your current API rate limit is "
                        f"{params.api_rate_limit} requests per minute.\n\n"
                        f"This means SQLDeps can process at most "
                        f"{custom_round(params.api_rate_limit * 60)} "
                        f"queries per hour, regardless of how many workers you add."
                    )
                else:
                    st.success(
                        "You have no API rate limit configured, so performance will scale with "
                        "the number of workers up to the limitations of your hardware and network."
                    )
        else:
            st.info("""
            ### Parallelism not applicable

            You currently have only 1 worker configured, so parallelism analysis is not applicable.

            To see the impact of parallelism on processing time and to find the optimal number of
            workers for your workload, increase the 'Maximum number of parallel workers' setting
            in the sidebar to at least 2.

            Parallelism can significantly reduce processing time for large workloads, especially
            when the API doesn't have strict rate limits.
            """)


def display_data_tables(params: SimulationParams, result: SimulationResult):
    """Display detailed data tables with results for deeper analysis.

    Creates expandable data tables showing detailed metrics from the
    simulation for more in-depth analysis. The tables include:
    - Time metrics (comparing processing times)
    - Cost metrics (breaking down costs by category)
    - Capacity metrics (showing throughput capabilities)
    - Parallelism data (showing impact of different worker counts)

    Args:
        params: SimulationParams used to generate the results.
        result: SimulationResult containing calculated metrics.
    """
    with st.expander("Detailed Results Data", expanded=False):
        # Time metrics table
        st.markdown("### Time Metrics (minutes)")
        time_data = {
            "Method": ["Manual Analysis", "SQLDeps (Sequential)", "SQLDeps (Parallel)"],
            "Total Time": [
                result.manual_analysis_time,
                result.sequential_processing_time,
                result.parallel_processing_time,
            ],
            "Time per Query": [
                params.avg_human_analysis_time_per_query,
                result.sequential_processing_time / params.num_queries,
                result.parallel_processing_time / params.num_queries,
            ],
            "Time Saved vs Manual": [
                0,
                result.time_saved_vs_manual["sequential"],
                result.time_saved_vs_manual["parallel"],
            ],
            "Time Saved (%)": [
                0,
                result.time_saved_percentage_vs_manual["sequential"],
                result.time_saved_percentage_vs_manual["parallel"],
            ],
        }
        st.dataframe(pd.DataFrame(time_data))

        # Cost metrics table
        st.markdown("### Cost Metrics (USD)")
        cost_data = {
            "Category": ["Manual Analysis", "API Total", "API Input", "API Output"],
            "Cost": [
                result.manual_analysis_cost,
                result.api_total_cost,
                result.api_input_cost,
                result.api_output_cost,
            ],
            "Cost per Query": [
                result.manual_analysis_cost / params.num_queries,
                result.api_total_cost / params.num_queries,
                result.api_input_cost / params.num_queries,
                result.api_output_cost / params.num_queries,
            ],
        }
        st.dataframe(pd.DataFrame(cost_data))

        # Capacity metrics table
        st.markdown("### Capacity Metrics (queries)")
        capacity_data = {
            "Method": ["Manual Analysis", "SQLDeps (Sequential)", "SQLDeps (Parallel)"],
            "Per Hour": [
                result.manual_per_hour,
                result.sequential_per_hour,
                result.parallel_per_hour,
            ],
            "Per Day (8h)": [
                result.manual_per_day,
                result.sequential_per_day,
                result.parallel_per_day,
            ],
            "Per Week (40h)": [
                result.manual_per_hour * 40,
                result.sequential_per_hour * 40,
                result.parallel_per_hour * 40,
            ],
            "Per Month (160h)": [
                result.manual_per_hour * 160,
                result.sequential_per_hour * 160,
                result.parallel_per_hour * 160,
            ],
        }
        st.dataframe(pd.DataFrame(capacity_data))

        # Parallelism data table
        st.markdown("### Parallelism Data")
        workers_data = {
            "Workers": list(range(1, params.max_workers + 1)),
            "Processing Time (min)": result.worker_times,
            "Queries per Hour": [(params.num_queries / time) * 60 for time in result.worker_times],
            "Queries per Day (8h)": [
                (params.num_queries / time) * 60 * 8 for time in result.worker_times
            ],
        }
        st.dataframe(pd.DataFrame(workers_data))

def display_comparison_section():
    """Display a side-by-side comparison of SQLDeps benefits versus manual analysis.

    Creates two columns with styled content to highlight the advantages of using SQLDeps
    compared to traditional manual SQL dependency analysis.
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background-color: #f9f3e6; border: 1px solid #e0d5c1; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 100%;">
            <h3 style="color: #4a4a4a; text-align: center; border-bottom: 2px solid #e0d5c1; padding-bottom: 10px;">⏱️ Without SQLDeps</h3>
            <ul style="list-style-type: none; padding-top: 20px; padding-left: 0;">
                <li style="margin-bottom: 10px; padding: 10px; background-color: #fff4e6; border-radius: 8px; border-left: 4px solid #ff6b6b;">
                    <strong style="color: #d32f2f;">Time-consuming</strong>: Engineers spend hours manually tracing dependencies
                </li>
                <li style="margin-bottom: 10px; padding: 10px; background-color: #fff4e6; border-radius: 8px; border-left: 4px solid #ff6b6b;">
                    <strong style="color: #d32f2f;">Expensive</strong>: Skilled engineers trapped in manual tracing work
                </li>
                <li style="margin-bottom: 10px; padding: 10px; background-color: #fff4e6; border-radius: 8px; border-left: 4px solid #ff6b6b;">
                    <strong style="color: #d32f2f;">Error-prone</strong>: Easy to miss complex dependencies
                </li>
                <li style="margin-bottom: 10px; padding: 10px; background-color: #fff4e6; border-radius: 8px; border-left: 4px solid #ff6b6b;">
                    <strong style="color: #d32f2f;">Inconsistent</strong>: Different engineers, different approaches
                </li>
                <li style="margin-bottom: 10px; padding: 10px; background-color: #fff4e6; border-radius: 8px; border-left: 4px solid #ff6b6b;">
                    <strong style="color: #d32f2f;">Limited scalability</strong>: Analysis capacity tied to team size
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)  # noqa: E501

    with col2:
        st.markdown("""
        <div style="background-color: #f0f8ff; border: 1px solid #b3d9ff; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 100%;">
            <h3 style="color: #2c3e50; text-align: center; border-bottom: 2px solid #b3d9ff; padding-bottom: 10px;">🚀 With SQLDeps </h3>
            <ul style="list-style-type: none; padding-top: 20px; padding-left: 0;">
                <li style="margin-bottom: 10px; padding: 10px; background-color: #e6f3ff; border-radius: 8px; border-left: 4px solid #4a90e2;">
                    <strong style="color: #2980b9;">Fast analysis</strong>: Process hundreds of queries in minutes
                </li>
                <li style="margin-bottom: 10px; padding: 10px; background-color: #e6f3ff; border-radius: 8px; border-left: 4px solid #4a90e2;">
                    <strong style="color: #2980b9;">Cost-effective</strong>: Fraction of the cost of manual analysis
                </li>
                <li style="margin-bottom: 10px; padding: 10px; background-color: #e6f3ff; border-radius: 8px; border-left: 4px solid #4a90e2;">
                    <strong style="color: #2980b9;">Comprehensive</strong>: Systematic detection of all dependencies
                </li>
                <li style="margin-bottom: 10px; padding: 10px; background-color: #e6f3ff; border-radius: 8px; border-left: 4px solid #4a90e2;">
                    <strong style="color: #2980b9;">Standardized</strong>: Consistent approach for all queries
                </li>
                <li style="margin-bottom: 10px; padding: 10px; background-color: #e6f3ff; border-radius: 8px; border-left: 4px solid #4a90e2;">
                    <strong style="color: #2980b9;">Scalable</strong>: Add workers to increase throughput
                </li>
            </ul>
        </div>
        """, unsafe_allow_html=True)  # noqa: E501

def main():
    """Main application function.

    Orchestrates the entire application flow, including:
    1. Setting up the page layout and introduction
    2. Collecting simulation parameters from the sidebar
    3. Running the simulation with provided parameters
    4. Displaying results through metrics, charts, and data tables
    5. Showing a footer with links to resources and disclaimers
    """
    # Initialize the page with title and description
    setup_page()

    # Get parameters from sidebar
    params = sidebar_parameters()

    # Run simulation
    with st.spinner("Running simulation..."):
        result = run_simulation(params)

    # Display results in organized sections
    display_metrics(result)
    display_charts(params, result)
    display_data_tables(params, result)
    display_comparison_section()
    # App footer with additional resources and disclaimers
    st.markdown("---")
    st.markdown("""
    ### Get Started with SQLDeps
    SQLDeps is available as an open-source Python package. Try it out!

    ```bash
    pip install sqldeps
    ```

    [⭐ GitHub](https://github.com/glue-lab/sqldeps) | 
    [📚 Documentation](https://sqldeps.readthedocs.io/) |
    [💬 Discussions](https://github.com/glue-lab/sqldeps/discussions) |
    [🌐 Web App](https://sqldeps.streamlit.app/)
    """)  # noqa: W291

    st.caption("""
    &nbsp;

    Disclaimer: Simulation results are estimates based on your inputs.  
    Actual performance may vary based on SQL complexity, LLM performance, and other factors.
    """)  # noqa: W291


if __name__ == "__main__":
    main()

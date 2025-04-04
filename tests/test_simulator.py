"""Test suite for calculator functions in the simulator module."""

import unittest

from src.models import SimulationParams
from src.simulator import (
    calculate_api_costs,
    calculate_manual_analysis_cost,
    calculate_manual_analysis_time,
    calculate_parallel_processing_time,
    calculate_processing_capacity,
    calculate_sequential_processing_time,
    calculate_worker_processing_times,
    run_simulation,
)


class TestCalculator(unittest.TestCase):
    """Test suite for calculator functions."""

    def setUp(self):
        """Set up test parameters."""
        self.params = SimulationParams(
            num_queries=100,
            hourly_salary=50.0,
            avg_human_analysis_time_per_query=5.0,
            avg_api_time_per_query=0.5,
            api_rate_limit=40,
            max_workers=4,
            api_input_price=2.5,
            api_output_price=10.0,
            avg_input_prompt_ntokens=5000,
            avg_output_prompt_ntokens=500,
        )

    def test_calculate_manual_analysis_time(self):
        """Test calculation of manual analysis time."""
        # Using direct parameters
        result = calculate_manual_analysis_time(
            num_queries=100, avg_human_analysis_time_per_query=5.0
        )
        expected = 100 * 5.0
        self.assertEqual(result, expected)

    def test_calculate_sequential_processing_time(self):
        """Test calculation of sequential processing time."""
        # Test with rate limiting - using direct parameters
        result = calculate_sequential_processing_time(
            num_queries=100, avg_api_time_per_query=0.5, api_rate_limit=40
        )
        # Expected: max(100*0.5, 100/40) = max(50, 2.5) = 50
        expected = max(100 * 0.5, 100 / 40)
        self.assertEqual(result, expected)

        # Test without rate limiting
        result = calculate_sequential_processing_time(
            num_queries=100, avg_api_time_per_query=0.5, api_rate_limit=None
        )
        expected = 100 * 0.5
        self.assertEqual(result, expected)

    def test_calculate_worker_processing_times(self):
        """Test calculation of worker processing times."""
        result = calculate_worker_processing_times(
            num_queries=100, avg_api_time_per_query=0.5, max_workers=4, api_rate_limit=40
        )

        # Should have one entry per worker
        self.assertEqual(len(result), 4)

        # First entry (1 worker) should match sequential time
        self.assertEqual(
            result[0],
            calculate_sequential_processing_time(
                num_queries=100, avg_api_time_per_query=0.5, api_rate_limit=40
            ),
        )

        # Check that times generally decrease with more workers
        self.assertTrue(result[0] >= result[-1])

    def test_calculate_parallel_processing_time(self):
        """Test calculation of parallel processing time."""
        result = calculate_parallel_processing_time(
            num_queries=100, avg_api_time_per_query=0.5, max_workers=4, api_rate_limit=40
        )
        worker_times = calculate_worker_processing_times(
            num_queries=100, avg_api_time_per_query=0.5, max_workers=4, api_rate_limit=40
        )
        expected = min(worker_times)
        self.assertEqual(result, expected)

    def test_calculate_manual_analysis_cost(self):
        """Test calculation of manual analysis cost."""
        result = calculate_manual_analysis_cost(
            num_queries=100, avg_human_analysis_time_per_query=5.0, hourly_salary=50.0
        )
        # Expected: (100 * 5 / 60) * 50 = 8.33 * 50 = 416.67
        expected = (100 * 5.0 / 60) * 50.0
        self.assertAlmostEqual(result, expected, places=2)

    def test_calculate_api_costs(self):
        """Test calculation of API costs."""
        input_cost, output_cost, total_cost = calculate_api_costs(
            num_queries=100,
            avg_input_prompt_ntokens=5000,
            avg_output_prompt_ntokens=500,
            api_input_price=2.5,
            api_output_price=10.0,
        )

        # Expected input cost: (100 * 5000 / 1_000_000) * 2.5 = 0.5 * 2.5 = 1.25
        expected_input = (100 * 5000 / 1_000_000) * 2.5

        # Expected output cost: (100 * 500 / 1_000_000) * 10 = 0.05 * 10 = 0.5
        expected_output = (100 * 500 / 1_000_000) * 10.0

        # Total: 1.25 + 0.5 = 1.75
        expected_total = expected_input + expected_output

        self.assertAlmostEqual(input_cost, expected_input, places=2)
        self.assertAlmostEqual(output_cost, expected_output, places=2)
        self.assertAlmostEqual(total_cost, expected_total, places=2)

    def test_calculate_processing_capacity(self):
        """Test calculation of processing capacity."""
        manual_time = 500  # 500 minutes
        sequential_time = 50  # 50 minutes
        parallel_time = 12.5  # 12.5 minutes
        num_queries = 100

        result = calculate_processing_capacity(
            manual_time, sequential_time, parallel_time, num_queries
        )

        # Expected: (queries / time) * 60
        expected_manual_per_hour = (num_queries / manual_time) * 60  # 12 per hour
        expected_sequential_per_hour = (num_queries / sequential_time) * 60  # 120 per hour
        expected_parallel_per_hour = (num_queries / parallel_time) * 60  # 480 per hour

        # Per day: per hour * 8
        expected_manual_per_day = expected_manual_per_hour * 8  # 96 per day
        expected_sequential_per_day = expected_sequential_per_hour * 8  # 960 per day
        expected_parallel_per_day = expected_parallel_per_hour * 8  # 3840 per day

        self.assertAlmostEqual(result[0], expected_manual_per_hour, places=2)
        self.assertAlmostEqual(result[1], expected_sequential_per_hour, places=2)
        self.assertAlmostEqual(result[2], expected_parallel_per_hour, places=2)
        self.assertAlmostEqual(result[3], expected_manual_per_day, places=2)
        self.assertAlmostEqual(result[4], expected_sequential_per_day, places=2)
        self.assertAlmostEqual(result[5], expected_parallel_per_day, places=2)

    def test_run_simulation(self):
        """Test the full simulation."""
        result = run_simulation(self.params)

        # Verify the simulation ran and returned a SimulationResult
        self.assertIsNotNone(result)
        self.assertEqual(result.manual_analysis_time, 500)  # 100 queries * 5 min

        # Test derived metrics
        self.assertTrue(result.time_saved_vs_manual["parallel"] > 0)

        # Cost savings
        self.assertTrue(result.cost_saved_vs_manual > 0)

        # Processing capacity
        self.assertTrue(result.parallel_per_hour > result.manual_per_hour)


if __name__ == "__main__":
    unittest.main()

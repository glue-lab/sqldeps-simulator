"""Utility functions for formatting and calculations.

This module provides helper functions for data formatting and calculations
used throughout the SQLDeps simulator.
"""


def custom_round(number):
    """Round a number with appropriate precision and add thousands separator.

    Args:
        number: The number to round.

    Returns:
        A string representing the rounded and formatted number with
        thousands separator. Rounds to 0 decimal places if decimals are zero,
        otherwise rounds to 2 decimal places.
    """
    formatted_number = f"{number:,.2f}"
    formatted_number = f"{number:,.0f}" if formatted_number.endswith(".00") else f"{number:,.2f}"
    return formatted_number

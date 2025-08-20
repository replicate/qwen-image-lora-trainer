"""
Billing metrics module for Cog predictors.

This module provides functionality to record billing metrics for Replicate models.
It validates metric types and values, and uses Cog's experimental metric recording API.
"""

import warnings
from typing import Union

from cog import ExperimentalFeatureWarning, current_scope

# Suppress experimental feature warnings for metric recording
warnings.filterwarnings("ignore", category=ExperimentalFeatureWarning)

# Metric definitions based on Replicate's billing system
# Reference: https://github.com/replicate/web/blob/main/replicate_web/metronome.py#L48-L65
# Reference: https://github.com/replicate/director/blob/fc47af0457a1eead08a2f6574ee06eb75c7f6c43/cog/types.go#L64

# Integer metrics that must have non-negative integer values
INTEGER_METRICS = {
    "audio_output_count",
    "character_input_count",
    "character_output_count",
    "generic_output_count",
    "image_output_count",
    "token_input_count",
    "token_output_count",
    "training_step_count",
    "video_output_count",
    "video_output_total_pixel_count",
}

# Float metrics that must have non-negative numeric values
FLOAT_METRICS = {
    "audio_output_duration_seconds",
    "unspecified_billing_metric",
    "video_output_duration_seconds",
}

# String metrics that accept text values
STRING_METRICS = {
    "model_variant",
    "motion_mode",
    "resolution_target",
    "resolution_upscale_target",
}

# Boolean metrics that must have true/false values
BOOL_METRICS = {
    "with_audio",
}

# Combined set of all valid metric names
ALL_METRICS = INTEGER_METRICS | FLOAT_METRICS | STRING_METRICS | BOOL_METRICS


def record_billing_metric(metric_name: str, value: Union[float, int, str, bool]) -> None:
    """
    Record a billing metric using Cog's metric recording API.
    
    This function validates the metric name and value type before recording.
    It ensures that metrics conform to Replicate's billing system requirements.
    
    Args:
        metric_name: The name of the metric to record. Must be one of the predefined metrics.
        value: The value to record. Type must match the metric's expected type.
    
    Raises:
        ValueError: If the metric name is invalid or the value type/range is incorrect.
    
    Examples:
        >>> record_billing_metric("image_output_count", 1)  # Record single image output
        >>> record_billing_metric("audio_output_duration_seconds", 10.5)  # Record audio duration
        >>> record_billing_metric("model_variant", "high_quality")  # Record model variant
        >>> record_billing_metric("with_audio", True)  # Record audio presence
    """
    # Validate metric name
    if metric_name not in ALL_METRICS:
        raise ValueError(
            f"Invalid metric name: {metric_name}. "
            f"Must be one of: {', '.join(sorted(ALL_METRICS))}"
        )
    
    # Validate value type and range based on metric category
    if metric_name in INTEGER_METRICS:
        if not isinstance(value, int):
            raise ValueError(
                f"Metric {metric_name} requires an integer value, got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(
                f"Metric {metric_name} requires a non-negative value, got {value}"
            )
    
    elif metric_name in FLOAT_METRICS:
        if not isinstance(value, (int, float)):
            raise ValueError(
                f"Metric {metric_name} requires a numeric value, got {type(value).__name__}"
            )
        if value < 0:
            raise ValueError(
                f"Metric {metric_name} requires a non-negative value, got {value}"
            )
    
    elif metric_name in STRING_METRICS:
        if not isinstance(value, str):
            raise ValueError(
                f"Metric {metric_name} requires a string value, got {type(value).__name__}"
            )
        if not value:  # Empty strings are not allowed
            raise ValueError(
                f"Metric {metric_name} requires a non-empty string value"
            )
    
    elif metric_name in BOOL_METRICS:
        if not isinstance(value, bool):
            raise ValueError(
                f"Metric {metric_name} requires a boolean value, got {type(value).__name__}"
            )
    
    # Record the metric using Cog's API
    try:
        current_scope().record_metric(metric_name, value)
    except Exception as e:
        # Log the error but don't fail the prediction
        # This ensures billing metrics don't break the main functionality
        print(f"Warning: Failed to record billing metric {metric_name}: {e}")


# Convenience functions for common metrics
def record_image_output(count: int = 1) -> None:
    """Record image output count for billing."""
    record_billing_metric("image_output_count", count)


def record_token_input(count: int) -> None:
    """Record token input count for billing."""
    record_billing_metric("token_input_count", count)


def record_token_output(count: int) -> None:
    """Record token output count for billing."""
    record_billing_metric("token_output_count", count)


def record_audio_duration(seconds: float) -> None:
    """Record audio output duration for billing."""
    record_billing_metric("audio_output_duration_seconds", seconds)


def record_video_duration(seconds: float) -> None:
    """Record video output duration for billing."""
    record_billing_metric("video_output_duration_seconds", seconds)


def record_training_steps(count: int) -> None:
    """Record training step count for billing."""
    record_billing_metric("training_step_count", count)

MIN_PAIRED_EXAMPLES = 5


class InsufficientDataError(ValueError):
    """Raised when a metric/arm-pair has fewer than MIN_PAIRED_EXAMPLES eligible paired examples."""

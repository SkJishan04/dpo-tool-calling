"""Error handling utilities."""

from typing import Optional, Callable, Any
import traceback
from utils.logger import setup_logger

logger = setup_logger(__name__)


class ErrorHandler:
    """Handle and log errors gracefully."""

    @staticmethod
    def safe_execute(
        func: Callable,
        *args,
        default_value: Any = None,
        log_error: bool = True,
        **kwargs
    ) -> Any:
        """
        Execute function with error handling.
        
        Args:
            func: Function to execute
            default_value: Value to return if error occurs
            log_error: Whether to log the error
        
        Returns:
            Function result or default_value
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if log_error:
                logger.error(f"Error in {func.__name__}: {str(e)}")
                logger.error(traceback.format_exc())
            return default_value

    @staticmethod
    def handle_json_error(func: Callable) -> Callable:
        """Decorator for JSON parsing errors."""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"JSON error in {func.__name__}: {str(e)}")
                return None
        return wrapper

    @staticmethod
    def handle_model_error(func: Callable) -> Callable:
        """Decorator for model inference errors."""
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except RuntimeError as e:
                logger.error(f"Model error: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
                return None
        return wrapper
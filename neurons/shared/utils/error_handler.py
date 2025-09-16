"""
Standardized error handling utilities for consistent logging and error management
"""

import traceback
from functools import wraps
from typing import Any, Callable, Optional

import bittensor as bt


class ErrorHandler:
    """Centralized error handling utilities"""

    @staticmethod
    def log_error(
        operation: str,
        error: Exception,
        context: Optional[dict] = None,
        include_traceback: bool = False,
    ) -> None:
        """
        Standardized error logging with consistent format

        Args:
            operation: Description of the operation that failed
            error: The exception that occurred
            context: Additional context information
            include_traceback: Whether to include full traceback
        """
        error_msg = f"Error in {operation}: {str(error)}"

        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            error_msg += f" (Context: {context_str})"

        if include_traceback:
            bt.logging.exception(error_msg)
        else:
            bt.logging.error(error_msg)

    @staticmethod
    def log_warning(
        operation: str, message: str, context: Optional[dict] = None
    ) -> None:
        """
        Standardized warning logging

        Args:
            operation: Description of the operation
            message: Warning message
            context: Additional context information
        """
        warning_msg = f"Warning in {operation}: {message}"

        if context:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            warning_msg += f" (Context: {context_str})"

        bt.logging.warning(warning_msg)

    @staticmethod
    def safe_execute(
        func: Callable,
        operation_name: str,
        default_return: Any = None,
        context: Optional[dict] = None,
        reraise: bool = False,
    ) -> Any:
        """
        Execute function with standardized error handling

        Args:
            func: Function to execute
            operation_name: Name of operation for logging
            default_return: Default return value on error
            context: Additional context for logging
            reraise: Whether to re-raise the exception after logging

        Returns:
            Function result or default_return on error
        """
        try:
            return func()
        except Exception as e:
            ErrorHandler.log_error(operation_name, e, context, include_traceback=True)
            if reraise:
                raise
            return default_return

    @staticmethod
    def async_error_handler(
        operation_name: str, context: Optional[dict] = None, reraise: bool = True
    ):
        """
        Decorator for standardized async error handling

        Args:
            operation_name: Name of operation for logging
            context: Additional context for logging
            reraise: Whether to re-raise exceptions
        """

        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    ErrorHandler.log_error(
                        operation_name, e, context, include_traceback=True
                    )
                    if reraise:
                        raise
                    return None

            return wrapper

        return decorator

    @staticmethod
    def sync_error_handler(
        operation_name: str,
        context: Optional[dict] = None,
        reraise: bool = True,
        default_return: Any = None,
    ):
        """
        Decorator for standardized sync error handling

        Args:
            operation_name: Name of operation for logging
            context: Additional context for logging
            reraise: Whether to re-raise exceptions
            default_return: Default return value on error
        """

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    ErrorHandler.log_error(
                        operation_name, e, context, include_traceback=True
                    )
                    if reraise:
                        raise
                    return default_return

            return wrapper

        return decorator


class ValidationError(Exception):
    """Custom exception for validation errors"""

    def __init__(self, message: str, context: Optional[dict] = None):
        self.context = context or {}
        super().__init__(message)


class CommunicationError(Exception):
    """Custom exception for communication errors"""

    def __init__(
        self,
        message: str,
        endpoint: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        self.endpoint = endpoint
        self.context = context or {}
        super().__init__(message)


class WorkerError(Exception):
    """Custom exception for worker-related errors"""

    def __init__(
        self,
        message: str,
        worker_id: Optional[str] = None,
        context: Optional[dict] = None,
    ):
        self.worker_id = worker_id
        self.context = context or {}
        super().__init__(message)

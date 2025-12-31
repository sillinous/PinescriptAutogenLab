# backend/middleware/error_handler.py

"""
Comprehensive error handling middleware.

Features:
- Catches all unhandled exceptions
- Logs errors with full stack traces
- Returns consistent error responses
- Tracks error rates
- Prevents sensitive data leakage
"""

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from backend.monitoring.logger import log_error, log_warning
from backend.monitoring.metrics import get_metrics_collector
import traceback
from typing import Union


async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global error handler for all unhandled exceptions.

    Args:
        request: FastAPI request object
        exc: Exception that was raised

    Returns:
        JSON error response
    """
    # Record error in metrics
    metrics = get_metrics_collector()
    metrics.record_error(str(type(exc).__name__))

    # Get request details for logging
    url = str(request.url)
    method = request.method
    client_ip = request.client.host if request.client else "unknown"

    # Determine error type and create appropriate response
    if isinstance(exc, StarletteHTTPException):
        # HTTP exceptions (404, 401, etc.)
        return await http_exception_handler(request, exc)

    elif isinstance(exc, RequestValidationError):
        # Validation errors from Pydantic
        return await validation_exception_handler(request, exc)

    elif isinstance(exc, ValueError):
        # Value errors (usually from business logic)
        log_warning(f"[ERROR] ValueError at {method} {url}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "error": "Invalid Request",
                "message": str(exc),
                "type": "ValueError"
            }
        )

    elif isinstance(exc, PermissionError):
        # Permission denied
        log_warning(f"[ERROR] PermissionError at {method} {url}: {str(exc)} from {client_ip}")
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": "Forbidden",
                "message": "You don't have permission to perform this action",
                "type": "PermissionError"
            }
        )

    elif isinstance(exc, TimeoutError):
        # Timeout errors
        log_error(f"[ERROR] TimeoutError at {method} {url}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": "Gateway Timeout",
                "message": "Request timed out",
                "type": "TimeoutError"
            }
        )

    elif isinstance(exc, ConnectionError):
        # Connection errors (to broker, database, etc.)
        log_error(f"[ERROR] ConnectionError at {method} {url}: {str(exc)}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Service Unavailable",
                "message": "Unable to connect to required service",
                "type": "ConnectionError"
            }
        )

    else:
        # Unexpected errors - log full traceback
        error_trace = traceback.format_exc()
        log_error(
            f"[ERROR] Unhandled exception at {method} {url}\n"
            f"Client: {client_ip}\n"
            f"Error: {str(exc)}\n"
            f"Traceback:\n{error_trace}"
        )

        # In production, don't expose internal error details
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "message": "An unexpected error occurred. Please try again later.",
                "type": type(exc).__name__,
                # Include error ID for support tickets
                "error_id": f"{method}_{url}_{hash(str(exc))}"
            }
        )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """
    Handler for HTTP exceptions (404, 401, etc.).

    Args:
        request: FastAPI request object
        exc: HTTPException that was raised

    Returns:
        JSON error response
    """
    # Log warnings for client errors (4xx), errors for server errors (5xx)
    if exc.status_code >= 500:
        log_error(f"[ERROR] HTTP {exc.status_code} at {request.method} {request.url}: {exc.detail}")
    else:
        log_warning(f"[WARN] HTTP {exc.status_code} at {request.method} {request.url}: {exc.detail}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail or "HTTP Error",
            "status_code": exc.status_code
        }
    )


async def validation_exception_handler(request: Request, exc: Union[RequestValidationError, ValidationError]) -> JSONResponse:
    """
    Handler for request validation errors.

    Args:
        request: FastAPI request object
        exc: Validation exception

    Returns:
        JSON error response with validation details
    """
    log_warning(f"[WARN] Validation error at {request.method} {request.url}: {exc.errors()}")

    # Format validation errors in user-friendly way
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error.get("loc", []))
        message = error.get("msg", "Invalid value")
        error_type = error.get("type", "validation_error")

        errors.append({
            "field": field,
            "message": message,
            "type": error_type
        })

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "Request validation failed",
            "details": errors
        }
    )


# Custom exception classes for business logic

class BusinessLogicError(Exception):
    """Base exception for business logic errors."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class InsufficientFundsError(BusinessLogicError):
    """Raised when account has insufficient funds."""
    def __init__(self, required: float, available: float):
        message = f"Insufficient funds. Required: ${required:.2f}, Available: ${available:.2f}"
        super().__init__(message, status_code=402)


class OrderRejectedError(BusinessLogicError):
    """Raised when broker rejects order."""
    def __init__(self, reason: str):
        message = f"Order rejected: {reason}"
        super().__init__(message, status_code=400)


class AuthenticationError(BusinessLogicError):
    """Raised for authentication failures."""
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)


class AuthorizationError(BusinessLogicError):
    """Raised for authorization failures."""
    def __init__(self, message: str = "Access denied"):
        super().__init__(message, status_code=403)


class RateLimitExceeded(BusinessLogicError):
    """Raised when API rate limit is exceeded."""
    def __init__(self, limit: int, window: str):
        message = f"Rate limit exceeded: {limit} requests per {window}"
        super().__init__(message, status_code=429)


class ResourceNotFoundError(BusinessLogicError):
    """Raised when requested resource doesn't exist."""
    def __init__(self, resource_type: str, resource_id: str):
        message = f"{resource_type} with ID {resource_id} not found"
        super().__init__(message, status_code=404)


class DuplicateResourceError(BusinessLogicError):
    """Raised when trying to create duplicate resource."""
    def __init__(self, resource_type: str, identifier: str):
        message = f"{resource_type} with {identifier} already exists"
        super().__init__(message, status_code=409)


# Handler for custom business logic errors
async def business_logic_error_handler(request: Request, exc: BusinessLogicError) -> JSONResponse:
    """Handler for custom business logic errors."""
    log_warning(f"[WARN] {type(exc).__name__} at {request.method} {request.url}: {exc.message}")

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__,
            "message": exc.message
        }
    )


# Middleware to catch all exceptions
class ErrorHandlingMiddleware:
    """Middleware to catch and handle all exceptions."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                # Track response status codes
                metrics = get_metrics_collector()
                if message.get("status", 200) >= 400:
                    metrics.record_error(f"HTTP_{message['status']}")
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as exc:
            # This should rarely be hit since FastAPI handles most exceptions
            # But it's here as a safety net
            log_error(f"[ERROR] Unhandled exception in middleware: {exc}")
            raise

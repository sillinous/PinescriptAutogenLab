# backend/middleware/__init__.py

from .error_handler import (
    global_error_handler,
    http_exception_handler,
    validation_exception_handler,
    business_logic_error_handler,
    ErrorHandlingMiddleware,
    BusinessLogicError,
    InsufficientFundsError,
    OrderRejectedError,
    AuthenticationError,
    AuthorizationError,
    RateLimitExceeded,
    ResourceNotFoundError,
    DuplicateResourceError
)

__all__ = [
    'global_error_handler',
    'http_exception_handler',
    'validation_exception_handler',
    'business_logic_error_handler',
    'ErrorHandlingMiddleware',
    'BusinessLogicError',
    'InsufficientFundsError',
    'OrderRejectedError',
    'AuthenticationError',
    'AuthorizationError',
    'RateLimitExceeded',
    'ResourceNotFoundError',
    'DuplicateResourceError'
]

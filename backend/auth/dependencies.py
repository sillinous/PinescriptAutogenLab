# backend/auth/dependencies.py

from fastapi import Depends, HTTPException, status, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
from backend.auth.auth_service import get_auth_service, AuthService
import time

# Security scheme
security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_service: AuthService = Depends(get_auth_service)
) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token.

    Usage in endpoints:
        @app.get("/protected")
        def protected_route(current_user: dict = Depends(get_current_user)):
            return {"user": current_user}
    """
    token = credentials.credentials

    user_data = auth_service.verify_access_token(token)

    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get full user data
    user = auth_service.get_user_by_id(user_data['user_id'])

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current active user (not suspended)."""
    # In a real implementation, check if user is active
    return current_user


async def get_current_admin_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current user (must be admin)."""
    if not current_user.get('is_admin'):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


async def verify_api_key(
    x_api_key: Optional[str] = Header(None),
    auth_service: AuthService = Depends(get_auth_service)
) -> Optional[Dict[str, Any]]:
    """
    Verify API key from X-API-Key header.

    This is an alternative to JWT tokens for programmatic access.

    Usage:
        @app.get("/api/data")
        def get_data(user: dict = Depends(verify_api_key)):
            return {"user_id": user['id']}
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required in X-API-Key header"
        )

    user = auth_service.verify_api_key(x_api_key)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key or rate limit exceeded"
        )

    return user


async def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    auth_service: AuthService = Depends(get_auth_service)
) -> Optional[Dict[str, Any]]:
    """
    Optional authentication - returns user if authenticated, None otherwise.

    Useful for endpoints that work both with and without auth.
    """
    if not credentials:
        return None

    user_data = auth_service.verify_access_token(credentials.credentials)

    if not user_data:
        return None

    return auth_service.get_user_by_id(user_data['user_id'])


class RateLimitChecker:
    """Rate limiting dependency."""

    def __init__(self, calls: int = 100, period: int = 60):
        """
        Initialize rate limiter.

        Args:
            calls: Number of allowed calls
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.requests = {}

    async def __call__(self, request: Request):
        """Check rate limit for request."""
        client_ip = request.client.host

        now = time.time()
        if client_ip not in self.requests:
            self.requests[client_ip] = []

        # Remove old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if now - req_time < self.period
        ]

        # Check limit
        if len(self.requests[client_ip]) >= self.calls:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {self.calls} requests per {self.period} seconds"
            )

        # Add current request
        self.requests[client_ip].append(now)


# Pre-configured rate limiters
rate_limit_strict = RateLimitChecker(calls=10, period=60)  # 10 calls/minute
rate_limit_normal = RateLimitChecker(calls=100, period=60)  # 100 calls/minute
rate_limit_generous = RateLimitChecker(calls=1000, period=60)  # 1000 calls/minute

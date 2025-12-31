# backend/auth/models.py

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class UserCreate(BaseModel):
    """Request model for user registration."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=100)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    """Request model for user login."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """Response model for login/refresh."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 86400  # 24 hours


class UserResponse(BaseModel):
    """Response model for user data."""
    id: int
    email: str
    username: str
    full_name: Optional[str]
    subscription_tier: str
    is_admin: bool
    api_key: Optional[str] = None


class PasswordChange(BaseModel):
    """Request model for password change."""
    old_password: str
    new_password: str = Field(..., min_length=8, max_length=100)


class APIKeyResponse(BaseModel):
    """Response model for API key generation."""
    api_key: str
    message: str

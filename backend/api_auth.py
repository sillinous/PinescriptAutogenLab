# backend/api_auth.py
"""
API endpoints for user authentication and authorization.
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, EmailStr
from typing import Dict, Any

from backend.auth.auth_service import get_auth_service, AuthService
from backend.auth.email_verification import get_email_verification_service
from backend.auth.password_reset import get_password_reset_service
from backend.auth.dependencies import get_current_user

router = APIRouter()

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    full_name: str = None

class UserLogin(BaseModel):
    username: str # Can be username or email
    password: str

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(user: UserCreate, auth_service: AuthService = Depends(get_auth_service)):
    try:
        created_user = auth_service.create_user(
            email=user.email,
            username=user.username,
            password=user.password,
            full_name=user.full_name
        )
        # In a real app, you'd trigger an email verification here
        return {"message": "User created successfully", "user": created_user}
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Could not create user.")

@router.post("/login")
def login_for_access_token(form_data: UserLogin, auth_service: AuthService = Depends(get_auth_service)):
    user = auth_service.authenticate_user(email=form_data.username, password=form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth_service.create_access_token(
        user_id=user['id'], username=user['username']
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/users/me", response_model=Dict[str, Any])
def read_users_me(current_user: Dict[str, Any] = Depends(get_current_user)):
    return current_user

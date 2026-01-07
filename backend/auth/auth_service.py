# backend/auth/auth_service.py

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
import jwt
from backend.config import Config
from backend.database import get_db
import secrets

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings - require dedicated secret for production security
if not Config.JWT_SECRET:
    import warnings
    warnings.warn(
        "JWT_SECRET not set - generating random key. Sessions will be invalidated on restart. "
        "Set JWT_SECRET environment variable for production.",
        UserWarning
    )
    SECRET_KEY = secrets.token_urlsafe(32)
else:
    SECRET_KEY = Config.JWT_SECRET
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours
REFRESH_TOKEN_EXPIRE_DAYS = 30


class AuthService:
    """User authentication and authorization service."""

    def __init__(self):
        self._init_db_tables()

    def _init_db_tables(self):
        """Initialize user tables if they don't exist."""
        conn = get_db()
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                full_name TEXT,
                is_active BOOLEAN DEFAULT 1,
                is_verified BOOLEAN DEFAULT 0,
                is_admin BOOLEAN DEFAULT 0,
                subscription_tier TEXT DEFAULT 'free',
                subscription_expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                api_key TEXT UNIQUE,
                api_calls_count INTEGER DEFAULT 0,
                api_calls_limit INTEGER DEFAULT 1000
            )
        """)

        # Sessions table for refresh tokens
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                refresh_token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_agent TEXT,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # API usage tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                status_code INTEGER,
                response_time_ms REAL,
                ip_address TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_api_key ON users(api_key)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(refresh_token)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_usage_user ON api_usage_log(user_id)")

        conn.commit()
        conn.close()

    # Password utilities
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    def hash_password(self, password: str) -> str:
        """Hash a password for storage."""
        return pwd_context.hash(password)

    # User management
    def create_user(
        self,
        email: str,
        username: str,
        password: str,
        full_name: Optional[str] = None,
        subscription_tier: str = "free"
    ) -> Dict[str, Any]:
        """
        Create a new user.

        Args:
            email: User email (unique)
            username: Username (unique)
            password: Plain text password (will be hashed)
            full_name: Optional full name
            subscription_tier: 'free', 'starter', 'pro', 'enterprise'

        Returns:
            User data (without password)
        """
        conn = get_db()
        cursor = conn.cursor()

        # Check if user exists
        cursor.execute("SELECT id FROM users WHERE email = ? OR username = ?", (email, username))
        if cursor.fetchone():
            conn.close()
            raise ValueError("User with this email or username already exists")

        # Hash password
        hashed_password = self.hash_password(password)

        # Generate API key
        api_key = f"pk_{secrets.token_urlsafe(32)}"

        # Set API limits based on tier
        api_limits = {
            'free': 1000,
            'starter': 10000,
            'pro': 100000,
            'enterprise': -1  # unlimited
        }

        # Insert user
        cursor.execute("""
            INSERT INTO users (email, username, hashed_password, full_name, subscription_tier, api_key, api_calls_limit)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (email, username, hashed_password, full_name, subscription_tier, api_key, api_limits.get(subscription_tier, 1000)))

        user_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return {
            'id': user_id,
            'email': email,
            'username': username,
            'full_name': full_name,
            'subscription_tier': subscription_tier,
            'api_key': api_key
        }

    def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user with email and password.

        Returns:
            User data if successful, None otherwise
        """
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()

        if not user:
            return None

        if not self.verify_password(password, user['hashed_password']):
            return None

        if not user['is_active']:
            return None

        # Update last login
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user['id'],))
        conn.commit()
        conn.close()

        return {
            'id': user['id'],
            'email': user['email'],
            'username': user['username'],
            'full_name': user['full_name'],
            'is_admin': bool(user['is_admin']),
            'subscription_tier': user['subscription_tier']
        }

    # JWT token management
    def create_access_token(self, user_id: int, username: str, expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

        to_encode = {
            'sub': str(user_id),
            'username': username,
            'exp': expire,
            'type': 'access'
        }

        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def create_refresh_token(
        self,
        user_id: int,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> str:
        """Create and store refresh token."""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO user_sessions (user_id, refresh_token, expires_at, user_agent, ip_address)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, token, expires_at, user_agent, ip_address))

        conn.commit()
        conn.close()

        return token

    def verify_access_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT access token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = int(payload.get('sub'))
            username = payload.get('username')

            if user_id is None or username is None:
                return None

            return {'user_id': user_id, 'username': username}

        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None

    def verify_refresh_token(self, token: str) -> Optional[int]:
        """Verify refresh token and return user_id."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id, expires_at
            FROM user_sessions
            WHERE refresh_token = ?
        """, (token,))

        session = cursor.fetchone()
        conn.close()

        if not session:
            return None

        if datetime.fromisoformat(session['expires_at']) < datetime.utcnow():
            # Token expired, delete it
            self.revoke_refresh_token(token)
            return None

        return session['user_id']

    def revoke_refresh_token(self, token: str):
        """Revoke a refresh token."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_sessions WHERE refresh_token = ?", (token,))
        conn.commit()
        conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        user = cursor.fetchone()
        conn.close()

        if not user:
            return None

        return {
            'id': user['id'],
            'email': user['email'],
            'username': user['username'],
            'full_name': user['full_name'],
            'is_admin': bool(user['is_admin']),
            'subscription_tier': user['subscription_tier'],
            'api_key': user['api_key'],
            'api_calls_count': user['api_calls_count'],
            'api_calls_limit': user['api_calls_limit']
        }

    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return user data."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM users WHERE api_key = ? AND is_active = 1", (api_key,))
        user = cursor.fetchone()
        conn.close()

        if not user:
            return None

        # Check API rate limit
        if user['api_calls_limit'] > 0 and user['api_calls_count'] >= user['api_calls_limit']:
            return None

        # Increment API call count
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET api_calls_count = api_calls_count + 1 WHERE id = ?", (user['id'],))
        conn.commit()
        conn.close()

        return {
            'id': user['id'],
            'email': user['email'],
            'username': user['username'],
            'subscription_tier': user['subscription_tier']
        }

    def reset_api_usage(self, user_id: int):
        """Reset monthly API usage counter."""
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET api_calls_count = 0 WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()

    def log_api_usage(
        self,
        user_id: Optional[int],
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        ip_address: Optional[str] = None
    ):
        """Log API usage for analytics."""
        conn = get_db()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO api_usage_log (user_id, endpoint, method, status_code, response_time_ms, ip_address)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, endpoint, method, status_code, response_time_ms, ip_address))

        conn.commit()
        conn.close()


# Global auth service instance
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get auth service instance."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service

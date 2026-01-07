import pytest
import httpx
from backend.app import app


@pytest.mark.asyncio
async def test_health():
    """Test health endpoint."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        r = await client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

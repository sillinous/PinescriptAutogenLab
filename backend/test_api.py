#!/usr/bin/env python
"""
Quick API test script
Run with: python backend/test_api.py
"""

import httpx
import asyncio

API_BASE = "http://localhost:8080"


async def test_endpoints():
    """Test key API endpoints."""
    print("\n" + "="*60)
    print("TESTING PINELAB API")
    print("="*60)

    async with httpx.AsyncClient() as client:
        # Test health check
        print("\n1. Testing /healthz...")
        try:
            response = await client.get(f"{API_BASE}/healthz")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   [ERROR] {e}")

        # Test P&L summary
        print("\n2. Testing /pnl/summary...")
        try:
            response = await client.get(f"{API_BASE}/pnl/summary")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   [ERROR] {e}")

        # Test orders endpoint
        print("\n3. Testing /journal/orders...")
        try:
            response = await client.get(f"{API_BASE}/journal/orders")
            print(f"   Status: {response.status_code}")
            print(f"   Orders count: {len(response.json())}")
        except Exception as e:
            print(f"   [ERROR] {e}")

        # Test positions
        print("\n4. Testing /positions...")
        try:
            response = await client.get(f"{API_BASE}/positions")
            print(f"   Status: {response.status_code}")
            print(f"   Positions count: {len(response.json())}")
        except Exception as e:
            print(f"   [ERROR] {e}")

        # Test webhook signature helper
        print("\n5. Testing /webhook/test-signature...")
        try:
            response = await client.get(f"{API_BASE}/webhook/test-signature")
            print(f"   Status: {response.status_code}")
            data = response.json()
            print(f"   Example payload: {data.get('example_payload')}")
            print(f"   Signature: {data.get('x_signature_header')[:20]}...")
        except Exception as e:
            print(f"   [ERROR] {e}")

        # Test broker credentials endpoint
        print("\n6. Testing /broker/credentials...")
        try:
            response = await client.get(f"{API_BASE}/broker/credentials")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.json()}")
        except Exception as e:
            print(f"   [ERROR] {e}")

    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60 + "\n")
    print("To start the server:")
    print("  uvicorn backend.app:app --reload --port 8080")
    print("\nThen visit:")
    print("  http://localhost:8080/docs")
    print("  http://localhost:5173 (frontend)")
    print()


if __name__ == "__main__":
    print("\n[INFO] Make sure the backend is running:")
    print("       uvicorn backend.app:app --reload --port 8080\n")

    try:
        asyncio.run(test_endpoints())
    except KeyboardInterrupt:
        print("\n[INFO] Tests cancelled")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        print("\nMake sure the backend is running!")

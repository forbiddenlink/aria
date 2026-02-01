#!/usr/bin/env python3
"""Test API endpoints without starting the full server."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fastapi.testclient import TestClient

from ai_artist.web.app import app


def test_health_endpoint():
    """Test health check endpoint."""
    print("\n🔍 Testing /health endpoint...")
    client = TestClient(app)

    response = client.get("/health")
    print(f"  Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"  ✓ Status: {data.get('status')}")
        print(f"  ✓ Version: {data.get('version')}")
        print(f"  ✓ Services: {list(data.get('services', {}).keys())}")
        return True
    else:
        print(f"  ❌ Failed: {response.text}")
        return False


def test_images_endpoint():
    """Test images list endpoint."""
    print("\n🔍 Testing /api/images endpoint...")
    client = TestClient(app)

    response = client.get("/api/images?limit=5")
    print(f"  Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"  ✓ Total images: {data.get('total', 0)}")
        print(f"  ✓ Images in response: {len(data.get('images', []))}")
        if data.get("images"):
            first_image = data["images"][0]
            print(
                f"  ✓ Sample image path: {first_image.get('image_path', 'N/A')[:50]}..."
            )
        return True
    else:
        print(f"  ❌ Failed: {response.text}")
        return False


def test_aria_state_endpoint():
    """Test Aria state endpoint."""
    print("\n🔍 Testing /api/aria/state endpoint...")
    client = TestClient(app)

    response = client.get("/api/aria/state")
    print(f"  Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"  ✓ Name: {data.get('name')}")
        print(f"  ✓ Mood: {data.get('mood')}")
        print(f"  ✓ Energy: {data.get('energy'):.2f}")
        print(f"  ✓ Paintings created: {data.get('paintings_created')}")
        print(f"  ✓ Personality traits: {len(data.get('personality', {}))}")
        return True
    else:
        print(f"  ❌ Failed: {response.text}")
        return False


def test_aria_statement_endpoint():
    """Test Aria artist statement endpoint."""
    print("\n🔍 Testing /api/aria/statement endpoint...")
    client = TestClient(app)

    response = client.get("/api/aria/statement")
    print(f"  Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"  ✓ Name: {data.get('name')}")
        print(f"  ✓ Statement length: {len(data.get('statement', ''))}")
        print(f"  ✓ Statement preview: {data.get('statement', '')[:80]}...")
        return True
    else:
        print(f"  ❌ Failed: {response.text}")
        return False


def test_homepage():
    """Test homepage rendering."""
    print("\n🔍 Testing / (homepage) endpoint...")
    client = TestClient(app)

    response = client.get("/")
    print(f"  Status: {response.status_code}")

    if response.status_code == 200:
        html = response.text
        print(f"  ✓ Response length: {len(html)} bytes")
        print(f"  ✓ Contains 'AI Artist': {'AI Artist' in html}")
        print(f"  ✓ Contains gallery div: {'gallery' in html.lower()}")
        return True
    else:
        print(f"  ❌ Failed: {response.text[:200]}")
        return False


def test_aria_page():
    """Test Aria page rendering."""
    print("\n🔍 Testing /aria page endpoint...")
    client = TestClient(app)

    response = client.get("/aria")
    print(f"  Status: {response.status_code}")

    if response.status_code == 200:
        html = response.text
        print(f"  ✓ Response length: {len(html)} bytes")
        print(f"  ✓ Contains 'Aria': {'Aria' in html}")
        print(f"  ✓ Contains CREATE button: {'CREATE' in html or 'create' in html}")
        return True
    else:
        print(f"  ❌ Failed: {response.text[:200]}")
        return False


def run_api_tests():
    """Run all API tests."""
    print("\n" + "=" * 60)
    print("🧪 API ENDPOINTS TEST")
    print("=" * 60)

    results = {}

    try:
        results["health"] = test_health_endpoint()
    except Exception as e:
        print(f"  ❌ Health endpoint failed: {e}")
        results["health"] = False

    try:
        results["images"] = test_images_endpoint()
    except Exception as e:
        print(f"  ❌ Images endpoint failed: {e}")
        results["images"] = False

    try:
        results["aria_state"] = test_aria_state_endpoint()
    except Exception as e:
        print(f"  ❌ Aria state endpoint failed: {e}")
        results["aria_state"] = False

    try:
        results["aria_statement"] = test_aria_statement_endpoint()
    except Exception as e:
        print(f"  ❌ Aria statement endpoint failed: {e}")
        results["aria_statement"] = False

    try:
        results["homepage"] = test_homepage()
    except Exception as e:
        print(f"  ❌ Homepage failed: {e}")
        results["homepage"] = False

    try:
        results["aria_page"] = test_aria_page()
    except Exception as e:
        print(f"  ❌ Aria page failed: {e}")
        results["aria_page"] = False

    # Summary
    print("\n" + "=" * 60)
    print("📊 API TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 ALL API TESTS PASSED!")
        return 0
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit_code = run_api_tests()
    sys.exit(exit_code)

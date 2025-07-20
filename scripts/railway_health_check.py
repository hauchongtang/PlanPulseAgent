"""
Simple health check script for Railway deployment.
Use this to verify your supervisor pattern is working correctly after deployment.
"""

import requests
import sys
import json
from typing import Dict, Any


def check_endpoint(url: str, endpoint: str, expected_status: int = 200) -> Dict[str, Any]:
    """Check a specific endpoint and return status info."""
    full_url = f"{url.rstrip('/')}/{endpoint.lstrip('/')}"
    
    try:
        response = requests.get(full_url, timeout=30)
        return {
            "url": full_url,
            "status_code": response.status_code,
            "success": response.status_code == expected_status,
            "response": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text[:200],
            "error": None
        }
    except requests.exceptions.RequestException as e:
        return {
            "url": full_url,
            "status_code": None,
            "success": False,
            "response": None,
            "error": str(e)
        }


def test_agent_routing(base_url: str) -> Dict[str, Any]:
    """Test the supervisor agent routing with a simple request."""
    endpoint = "v1/telegram/"
    full_url = f"{base_url.rstrip('/')}/{endpoint}"
    
    test_payload = {
        "chat_id": "health_check",
        "chat_message": "Calculate 2 + 2"
    }
    
    try:
        response = requests.post(full_url, json=test_payload, timeout=30)
        
        result = {
            "url": full_url,
            "status_code": response.status_code,
            "success": response.status_code == 200,
            "error": None
        }
        
        if response.status_code == 200:
            data = response.json()
            supervisor_meta = data.get("supervisor_metadata", {})
            result["selected_agent"] = supervisor_meta.get("selected_agent")
            result["confidence_scores"] = supervisor_meta.get("confidence_scores", {})
            result["response_preview"] = data.get("response", "")[:100]
        else:
            result["response"] = response.text[:200]
            
        return result
        
    except requests.exceptions.RequestException as e:
        return {
            "url": full_url,
            "status_code": None,
            "success": False,
            "response": None,
            "error": str(e)
        }


def main():
    """Run health checks on Railway deployment."""
    if len(sys.argv) != 2:
        print("Usage: python railway_health_check.py <your-railway-url>")
        print("Example: python railway_health_check.py https://planpulseagent-production.up.railway.app")
        sys.exit(1)
    
    base_url = sys.argv[1]
    
    print("🚀 Railway Deployment Health Check")
    print("=" * 50)
    print(f"Target: {base_url}")
    print()
    
    # Test endpoints
    tests = [
        ("Root Endpoint", ""),
        ("Health Check", "health"),
        ("Agent Health", "v1/telegram/health"),
        ("Agent Info", "v1/telegram/agents"),
        ("API Docs", "docs"),
    ]
    
    results = []
    
    for test_name, endpoint in tests:
        print(f"Testing {test_name}...")
        result = check_endpoint(base_url, endpoint)
        results.append((test_name, result))
        
        if result["success"]:
            print(f"  ✅ Status: {result['status_code']}")
        else:
            print(f"  ❌ Failed: {result['error'] or f'Status {result['status_code']}'}")
    
    # Test agent routing
    print(f"\nTesting Agent Routing...")
    routing_result = test_agent_routing(base_url)
    results.append(("Agent Routing", routing_result))
    
    if routing_result["success"]:
        print(f"  ✅ Status: {routing_result['status_code']}")
        print(f"  🤖 Selected Agent: {routing_result.get('selected_agent', 'unknown')}")
        print(f"  📊 Confidence Scores: {routing_result.get('confidence_scores', {})}")
        print(f"  💬 Response: {routing_result.get('response_preview', 'No preview')}")
    else:
        print(f"  ❌ Failed: {routing_result['error'] or f'Status {routing_result['status_code']}'}")
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Summary:")
    
    passed = sum(1 for _, result in results if result["success"])
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Your supervisor pattern is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        print("\n💡 Common issues:")
        print("   • Environment variables not set (GOOGLE_API_KEY, NOTION_TOKEN, etc.)")
        print("   • Application still starting up (try again in a minute)")
        print("   • Network connectivity issues")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())

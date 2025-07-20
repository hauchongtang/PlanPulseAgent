"""
Test script to demonstrate the supervisor pattern with specialized agents.
Run this to see how different types of requests are routed to appropriate agents.
"""

import asyncio
import aiohttp
import json
from typing import List, Dict


async def test_supervisor_routing():
    """Test the supervisor pattern with various request types."""
    base_url = "http://127.0.0.1:8001/v1/telegram"
    
    # Test cases that should route to different agents
    test_cases = [
        {
            "name": "Math Calculation",
            "chat_message": "Calculate the square root of 144 plus 15% of 200",
            "expected_agent": "math_agent"
        },
        {
            "name": "Unit Conversion", 
            "chat_message": "Convert 75 degrees fahrenheit to celsius",
            "expected_agent": "math_agent"
        },
        {
            "name": "Notion Events",
            "chat_message": "Show me my events for tomorrow",
            "expected_agent": "notion_agent"
        },
        {
            "name": "Calendar Query",
            "chat_message": "What's on my schedule this week?",
            "expected_agent": "notion_agent"
        },
        {
            "name": "Mixed Request",
            "chat_message": "How many days until my next meeting?",
            "expected_agent": "notion_agent"  # Should lean towards notion for calendar info
        },
        {
            "name": "Percentage Calculation",
            "chat_message": "What is 25% of 480?",
            "expected_agent": "math_agent"
        }
    ]
    
    print("🤖 Testing Supervisor Pattern Agent Routing\n")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📋 Test {i}: {test_case['name']}")
            print(f"📝 Message: {test_case['chat_message']}")
            print(f"🎯 Expected Agent: {test_case['expected_agent']}")
            
            try:
                # Make request to supervisor
                payload = {
                    "chat_id": f"test_user_{i}",
                    "chat_message": test_case['chat_message']
                }
                
                async with session.post(f"{base_url}/", json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract supervisor metadata
                        supervisor_meta = result.get("supervisor_metadata", {})
                        selected_agent = supervisor_meta.get("selected_agent", "unknown")
                        confidence_scores = supervisor_meta.get("confidence_scores", {})
                        
                        print(f"✅ Selected Agent: {selected_agent}")
                        print(f"📊 Confidence Scores:")
                        for agent, score in confidence_scores.items():
                            emoji = "🎯" if agent == selected_agent else "  "
                            print(f"   {emoji} {agent}: {score:.2f}")
                        
                        # Check if routing was correct
                        if selected_agent == test_case['expected_agent']:
                            print("✅ Routing: CORRECT")
                        else:
                            print("⚠️  Routing: UNEXPECTED")
                        
                        print(f"💬 Response: {result.get('response', 'No response')[:100]}...")
                        
                    else:
                        print(f"❌ Request failed: {response.status}")
                        
            except Exception as e:
                print(f"❌ Error: {str(e)}")
            
            print("-" * 60)


async def test_agent_info():
    """Test the agent information endpoints."""
    base_url = "http://127.0.0.1:8001/v1/telegram"
    
    print("\n🔍 Testing Agent Information Endpoints\n")
    
    async with aiohttp.ClientSession() as session:
        # Test health endpoint
        print("🏥 Health Check:")
        try:
            async with session.get(f"{base_url}/health") as response:
                if response.status == 200:
                    health = await response.json()
                    print(f"   Status: {health.get('status', 'unknown')}")
                    print(f"   Agents Managed: {health.get('service_info', {}).get('agents_managed', 0)}")
                    
                    agents_health = health.get('agents', {})
                    for agent_name, agent_status in agents_health.items():
                        status = agent_status.get('status', 'unknown')
                        print(f"   {agent_name}: {status}")
                else:
                    print(f"   ❌ Health check failed: {response.status}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
        
        # Test agents info endpoint
        print("\n🤖 Agent Capabilities:")
        try:
            async with session.get(f"{base_url}/agents") as response:
                if response.status == 200:
                    agents_info = await response.json()
                    
                    for agent_name, capabilities in agents_info.get('capabilities', {}).items():
                        print(f"\n   📱 {agent_name}:")
                        for capability in capabilities:
                            print(f"      • {capability}")
                else:
                    print(f"   ❌ Agents info failed: {response.status}")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")


async def main():
    """Run all tests."""
    print("🚀 PlanPulseAgent Supervisor Pattern Test Suite")
    print("=" * 60)
    
    try:
        await test_agent_info()
        await test_supervisor_routing()
        
        print("\n🎉 Test Suite Complete!")
        print("\n💡 Key Observations:")
        print("   • Each request is analyzed by the supervisor")
        print("   • Confidence scores determine agent selection")
        print("   • Specialized agents handle domain-specific tasks")
        print("   • Full transparency in routing decisions")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {str(e)}")
        print("   Make sure the server is running on http://127.0.0.1:8001")


if __name__ == "__main__":
    asyncio.run(main())

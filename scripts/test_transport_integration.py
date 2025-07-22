#!/usr/bin/env python3
"""
Test script to verify transport agent integration with supervisor.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.supervisor_agent import SupervisorAgent
from app.agents.transport_agent import TransportAgent

def test_transport_agent_standalone():
    """Test the transport agent directly."""
    print("=== Testing Transport Agent (Standalone) ===")
    
    transport_agent = TransportAgent()
    
    # Test confidence scoring
    test_queries = [
        "What are the bus stops near Queensway Shopping Centre?",
        "How to get to Marina Bay Sands?",
        "Calculate 2 + 2",
        "What are my events today?",
        "Bus near Orchard Road"
    ]
    
    print("\n--- Confidence Scores ---")
    for query in test_queries:
        confidence = transport_agent.can_handle(query)
        print(f"'{query}' -> {confidence:.2f}")
    
    # Test actual processing
    print("\n--- Processing Transport Query ---")
    test_query = "Bus stops near Orchard Road"
    result = transport_agent.process_task(test_query)
    print(f"Query: {test_query}")
    print(f"Success: {result.get('success')}")
    print(f"Response: {result.get('response', 'No response')[:200]}...")

def test_supervisor_integration():
    """Test transport agent through supervisor."""
    print("\n=== Testing Supervisor Integration ===")
    
    supervisor = SupervisorAgent()
    
    # Test transport query through supervisor
    transport_query = "What are the bus stops near Queensway Shopping Centre?"
    print(f"\nProcessing: {transport_query}")
    
    result = supervisor.process_message(transport_query, user_id="test_user")
    
    print(f"Success: {result.get('success')}")
    print(f"Selected Agent: {result.get('supervisor_metadata', {}).get('selected_agent')}")
    print(f"Confidence Scores: {result.get('supervisor_metadata', {}).get('confidence_scores')}")
    print(f"Response: {result.get('response', 'No response')[:300]}...")

def test_agent_capabilities():
    """Test agent capabilities and health check."""
    print("\n=== Testing Agent Capabilities ===")
    
    supervisor = SupervisorAgent()
    
    # Get capabilities
    capabilities = supervisor.get_agent_capabilities()
    print("\nAgent Capabilities:")
    for agent_name, caps in capabilities.items():
        print(f"\n{agent_name}:")
        for cap in caps:
            print(f"  - {cap}")
    
    # Health check
    health = supervisor.health_check()
    print(f"\nHealth Check: {health.get('status')}")
    print(f"Agents Status:")
    for agent_name, status in health.get('agents', {}).items():
        print(f"  {agent_name}: {status.get('status')}")

if __name__ == "__main__":
    print("🚌 Transport Agent Integration Test")
    print("=" * 50)
    
    try:
        test_transport_agent_standalone()
        test_supervisor_integration()
        test_agent_capabilities()
        print("\n✅ All tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

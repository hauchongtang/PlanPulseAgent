#!/usr/bin/env python3
"""
Test script to demonstrate follow-up query functionality for transport agent.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.supervisor_agent import SupervisorAgent

def test_follow_up_queries():
    """Test the follow-up query functionality with bus arrival times."""
    print("🚌 Testing Follow-up Query Functionality")
    print("=" * 50)
    
    supervisor = SupervisorAgent()
    
    # Simulate a conversation flow
    print("\n📍 **Step 1: Initial Location Query**")
    print("User: \"What are the bus stops near Orchard Road?\"")
    
    result1 = supervisor.process_message("What are the bus stops near Orchard Road?", user_id="test_followup")
    print(f"\nResponse: {result1.get('response', '')[:400]}...")
    print(f"Agent Selected: {result1.get('supervisor_metadata', {}).get('selected_agent')}")
    
    # Follow-up query with specific bus stop code
    print("\n\n🕐 **Step 2: Follow-up Arrival Time Query**")
    print("User: \"Bus arrival times at 09047\"")
    
    result2 = supervisor.process_message("Bus arrival times at 09047", user_id="test_followup")
    print(f"\nResponse: {result2.get('response', '')[:500]}...")
    print(f"Agent Selected: {result2.get('supervisor_metadata', {}).get('selected_agent')}")
    print(f"Confidence Score: {result2.get('supervisor_metadata', {}).get('confidence_scores', {}).get('transport_agent', 0)}")
    
    # Another follow-up query format
    print("\n\n🚌 **Step 3: Alternative Follow-up Format**")
    print("User: \"When is the next bus at 11519?\"")
    
    result3 = supervisor.process_message("When is the next bus at 11519?", user_id="test_followup")
    print(f"\nResponse: {result3.get('response', '')[:500]}...")
    print(f"Agent Selected: {result3.get('supervisor_metadata', {}).get('selected_agent')}")
    print(f"Confidence Score: {result3.get('supervisor_metadata', {}).get('confidence_scores', {}).get('transport_agent', 0)}")

def test_confidence_scoring():
    """Test confidence scoring for different query types."""
    print("\n\n🎯 **Testing Enhanced Confidence Scoring**")
    print("=" * 50)
    
    from app.agents.transport_agent import TransportAgent
    transport_agent = TransportAgent()
    
    test_queries = [
        ("Bus arrival times at 09047", "Arrival query with bus stop code"),
        ("When is the next bus at 11519?", "Next bus query"),
        ("Bus stops near Orchard Road", "Location-based bus stop query"),
        ("How to get to Marina Bay Sands", "Directions query"),
        ("What are my meetings today?", "Non-transport query"),
        ("arrival times near raffles place", "Arrival query with location"),
    ]
    
    print("\nQuery Type Analysis:")
    for query, description in test_queries:
        confidence = transport_agent.can_handle(query)
        print(f"• {description}")
        print(f"  Query: \"{query}\"")
        print(f"  Confidence: {confidence:.2f}")
        print()

if __name__ == "__main__":
    try:
        test_follow_up_queries()
        test_confidence_scoring()
        print("✅ Follow-up query tests completed!")
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

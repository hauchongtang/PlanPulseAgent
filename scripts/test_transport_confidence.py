#!/usr/bin/env python3
"""
Tes        # Lower confidence (0.1-0.4)
        ("How to travel to Jurong East", 0.4),
        ("Directions near Bishan", 0.4),
        ("Bus near me", 0.5),
        ("Where is the shopping mall", 0.1),
        ("Find location", 0.2),
        
        # Very low confidence for non-transport queries (correctly filtered)
        ("What time is it", 0.05),
        ("Calculate 2 + 2", 0.05),
        ("My calendar events tomorrow", 0.05),Transport Agent confidence scoring.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents.transport_agent import TransportAgent

def test_confidence_scoring():
    """Test various queries and their confidence scores."""
    
    agent = TransportAgent()
    
    # Test cases with expected confidence ranges
    test_cases = [
        # Very high confidence (0.9+)
        ("Bus arrival times at 09047", 0.9),
        ("When is the next bus at Marina Bay Sands", 0.9),
        ("Public transport near Orchard Road", 0.9),
        ("How to get to Changi Airport", 0.9),
        ("Bus stop code 11519 timing", 0.9),
        
        # High confidence (0.7-0.9)
        ("Bus stops near Queensway Shopping Centre", 0.7),
        ("Transport options around Tampines", 0.7),
        ("Bus route 174 schedule", 0.6),
        
        # Medium confidence (0.4-0.7)
        ("How to travel to Jurong East", 0.4),
        ("Directions near Bishan", 0.4),
        ("Bus near me", 0.5),
        
        # Lower confidence (0.1-0.4)
        ("Where is the shopping mall", 0.2),
        ("Find location", 0.2),
        ("What time is it", 0.1),
        
        # Math/other queries (should be very low)
        ("Calculate 2 + 2", 0.1),
        ("My calendar events tomorrow", 0.1),
    ]
    
    print("🧪 **Transport Agent Confidence Scoring Test**\n")
    print("=" * 60)
    
    for query, expected_min in test_cases:
        confidence = agent.can_handle(query)
        explanation = agent.get_confidence_explanation(query)
        
        # Determine result
        if confidence >= expected_min:
            result = "✅ PASS"
        else:
            result = "❌ FAIL"
        
        print(f"\n**Query:** {query}")
        print(f"**Confidence:** {confidence} (expected ≥ {expected_min}) {result}")
        
        if explanation['explanations']:
            print(f"**Reasoning:** {'; '.join(explanation['explanations'])}")
        else:
            print("**Reasoning:** Base confidence (no specific matches)")
    
    print("\n" + "=" * 60)
    print("🎯 **Test Summary**")
    
    # Test specific scenarios
    print("\n**Scenario Testing:**")
    
    scenarios = [
        "Bus stops near Marina Bay Sands",
        "When is the next bus 97 arriving at stop 09047?",
        "How to get to Orchard Road using public transport",
        "Transport options around Changi Airport",
        "Bus service 174 arrival times"
    ]
    
    for scenario in scenarios:
        conf = agent.can_handle(scenario)
        print(f"• {scenario}: {conf}")

if __name__ == "__main__":
    test_confidence_scoring()

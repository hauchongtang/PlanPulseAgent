# PlanPulseAgent - Supervisor Pattern Architecture

## Overview
PlanPulseAgent implements a sophisticated **Supervisor Pattern** using LangGraph's orchestration capabilities. The system automatically analyzes incoming requests and routes them to the most appropriate specialized agent for optimal performance and accuracy.

## 🏗️ Architecture

### Supervisor Pattern Design
```
┌─────────────────┐
│  User Request   │
└─────────┬───────┘
          │
    ┌─────▼─────┐
    │ Supervisor │
    │   Agent    │
    └─────┬─────┘
          │ (Analyzes & Routes)
    ┌─────▼─────┐
    │ Confidence │
    │ Scoring    │
    └─────┬─────┘
          │
  ┌───────▼───────┐
  │               │
┌─▼─┐   ┌─▼─┐   ┌─▼─┐
│N.A│   │M.A│   │T.A│
└───┘   └───┘   └───┘
Notion  Math   Transport
Agent   Agent   Agent
```

### 1. **Supervisor Agent** (`app/agents/supervisor_agent.py`)
- **Orchestration Engine**: Analyzes tasks and delegates to specialized agents
- **Confidence Scoring**: Evaluates which agent is best suited for each task
- **LangGraph Workflow**: Uses state management for robust task routing
- **Fallback Handling**: Graceful degradation when confidence is low

### 2. **Specialized Agents**

#### **Notion Agent** (`app/agents/notion_agent.py`)
- **Domain**: Calendar events, schedules, Notion database operations
- **Tools**: `get_events`, `add` (calendar operations)
- **Optimization**: Lower temperature (0.3) for consistent date handling
- **Expertise**: Date parsing, event retrieval, schedule management

#### **Math Agent** (`app/agents/math_agent.py`)
- **Domain**: Mathematical calculations, unit conversions, percentage operations
- **Tools**: `calculate`, `convert_units`, `percentage_calculator`
- **Optimization**: Very low temperature (0.1) for precise calculations
- **Expertise**: Arithmetic, algebra, trigonometry, unit conversions

#### **Transport Agent** (`app/agents/transport_agent.py`)
- **Domain**: Singapore public transport, location search, real-time bus arrivals
- **Tools**: `transport_search`, `google_places`, `lta_datamall`
- **Optimization**: Moderate temperature (0.4) for natural conversation
- **Expertise**: Location discovery, bus stop information, arrival times, follow-up queries

### 3. **Agent Service Layer** (`app/services/agent_service.py`)
- **Supervisor Coordination**: Single entry point for all agent interactions
- **Service Abstraction**: Clean interface for route handlers
- **Health Monitoring**: Comprehensive health checks across all agents
- **Error Handling**: Graceful error propagation and fallback mechanisms

## 🎯 Key Features

### ✅ **Intelligent Task Routing**
- **Dynamic Analysis**: Each request is analyzed for content and intent
- **Confidence-Based Selection**: Agents score their capability for each task
- **Optimal Delegation**: Tasks routed to the most capable agent
- **Transparent Reasoning**: Full visibility into routing decisions

### ✅ **Specialized Expertise**
- **Domain-Specific Optimization**: Each agent fine-tuned for their specialty
- **Appropriate Tooling**: Agents have access to relevant tools only
- **Optimized Parameters**: Temperature and settings tailored per domain
- **Expert Prompting**: Specialized system messages for each agent type

### ✅ **Robust Error Handling**
- **Graceful Degradation**: System continues functioning if one agent fails
- **Comprehensive Logging**: Detailed error reporting and debugging info
- **Fallback Mechanisms**: Alternative routing when primary selection fails
- **Health Monitoring**: Real-time status of all system components

### ✅ **Scalable Design**
- **Easy Agent Addition**: New specialized agents can be added seamlessly
- **Modular Architecture**: Each component is independently testable
- **Resource Efficiency**: Lazy loading and singleton patterns
- **Production Ready**: Optimized for deployment and monitoring

### ✅ **Real-Time Transport Integration** 
- **Google Places API**: Location search with free tier optimization
- **LTA DataMall API**: Live Singapore bus data with arrival predictions
- **Follow-Up Query Support**: Conversational transport assistance
- **Mobile-Optimized Format**: Compact display with emojis and color coding
- **Bus Capacity Indicators**: Real-time crowding information (🟢🟡🔴)

## 📊 Agent Selection Process

### Confidence Scoring Algorithm
Each specialized agent evaluates incoming tasks:

1. **Keyword Analysis**: High/medium confidence keywords detection
2. **Pattern Matching**: Mathematical symbols, date patterns, etc.
3. **Domain Relevance**: Task alignment with agent expertise
4. **Confidence Calculation**: Score between 0.0-1.0

### Example Routing:
- `"What's my schedule for tomorrow?"` → **Notion Agent** (0.9 confidence)
- `"Calculate 15% tip on $42.50"` → **Math Agent** (0.9 confidence)
- `"Bus routes to Marina Bay Sands"` → **Transport Agent** (0.9 confidence)
- `"Convert 10 miles to kilometers"` → **Math Agent** (0.8 confidence)
- `"Show me events this week"` → **Notion Agent** (0.8 confidence)
- `"What time does the next bus arrive?"` → **Transport Agent** (0.9 confidence)

## � Transport System Architecture

### Singapore Public Transport Integration
The Transport Agent provides comprehensive Singapore public transport information through a multi-layered approach:

#### **🗺️ Location Discovery** (`app/tools/transport/google_places.py`)
- **Google Places API**: Search for locations, malls, attractions
- **Free Tier Optimization**: Text search with minimal API usage
- **Address Normalization**: Consistent location formatting
- **Nearby Discovery**: Find transport hubs near destinations

#### **🚌 Real-Time Bus Data** (`app/tools/transport/lta_datamall.py`)
- **LTA DataMall v3 API**: Live Singapore bus arrival times
- **Bus Stop Information**: Complete network of 5,000+ stops
- **Service Details**: Routes, destinations, operational status
- **Capacity Indicators**: Real-time bus crowding levels

#### **🔍 Unified Transport Search** (`app/tools/transport/transport_search.py`)
- **Intelligent Routing**: Combines location search with transport data
- **Multi-Modal Results**: Bus stops, services, and alternatives
- **Follow-Up Query Handling**: Seamless conversation continuity
- **Mobile-Friendly Format**: Compact display with visual indicators

### Follow-Up Query Capabilities
The Transport Agent excels at conversational transport assistance:

```
User: "Bus routes to Marina Bay Sands"
Agent: [Lists nearby bus stops with services]

User: "What time does bus 97 arrive at stop 09047?"
Agent: "🚌 **Bus 97** at Marina Centre/The Shoppes (09047):
• **2 mins** 🟢 (Not Crowded)
• **32 mins** 🟢 (Not Crowded)  
• **43 mins** 🟡 (Standing Available)"
```

## �🔧 API Endpoints

### Core Endpoints

#### **POST** `/v1/telegram/`
Process messages through supervisor orchestration
```json
{
  "chat_id": "user123",
  "chat_message": "Calculate the square root of 144"
}
```

**Enhanced Response:**
```json
{
  "chat_id": "user123",
  "chat_message": "Bus routes to Marina Bay Sands",
  "response": "🚌 **Marina Bay Sands** transport options:\n\n**Bus Stops Nearby:**\n• **Marina Centre/The Shoppes** (09047)\n  📍 Marina Bay Sands Shopping Centre\n  🚌 Services: 36, 97, 97e, 106, 133, 502A\n\n• **Marina Sq/Millenia** (11519)\n  📍 Marina Square Shopping Centre\n  🚌 Services: 7, 36, 97, 97e, 106, 133\n\n💡 *Ask me for live arrival times: \"What time does bus 97 arrive at stop 09047?\"*",
  "success": true,
  "metadata": {},
  "supervisor_metadata": {
    "selected_agent": "transport_agent",
    "confidence_scores": {
      "transport_agent": 0.9,
      "notion_agent": 0.1,
      "math_agent": 0.1
    },
    "reasoning": "High confidence transport-related location search",
    "agent_response": {...}
  }
}
```

#### **GET** `/v1/telegram/agents`
Get detailed agent information
```json
{
  "supervisor_pattern": true,
  "agents": {
    "notion_agent": [
      "Retrieve events from Notion databases",
      "Query calendar schedules",
      "Handle date-based operations"
    ],
    "math_agent": [
      "Perform mathematical calculations",
      "Convert between units",
      "Calculate percentages"
    ],
    "transport_agent": [
      "Search Singapore public transport options",
      "Provide real-time bus arrival information",
      "Location discovery with Google Places API",
      "Handle follow-up queries about transport"
    ]
  },
  "workflow": "Each request is analyzed and routed to the most appropriate specialized agent"
}
```

#### **GET** `/v1/telegram/health`
Comprehensive health monitoring
```json
{
  "status": "healthy",
  "supervisor_model": "gemini-2.5-flash",
  "agents": {
    "notion_agent": {"status": "healthy", "capabilities_count": 5},
    "math_agent": {"status": "healthy", "capabilities_count": 7},
    "transport_agent": {"status": "healthy", "capabilities_count": 8}
  },
  "workflow_compiled": true,
  "service_info": {
    "pattern": "supervisor",
    "agents_managed": 3,
    "workflow_enabled": true
  }
}
```

## 🚀 Development & Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run with hot reload
uvicorn main:app --reload --port 8000

# Access documentation
http://localhost:8000/docs
```

### Production Deployment
```bash
# Optimized for Azure App Service
gunicorn -w 2 -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:8000 --timeout 600 --preload
```

### Adding New Agents
```python
# 1. Create specialized agent
class NewAgent:
    def __init__(self):
        self.name = "new_agent"
        self.tools = [your_tools]
    
    def can_handle(self, task: str) -> float:
        # Implement confidence scoring
        return confidence_score
    
    def process_task(self, task: str) -> Dict[str, Any]:
        # Implement task processing
        return response

# 2. Register with supervisor
# In supervisor_agent.py:
self.agents["new_agent"] = NewAgent()
```

## 🔍 Monitoring & Debugging

### Agent Selection Transparency
- **Confidence Scores**: See why each agent was or wasn't selected
- **Reasoning**: Natural language explanation of routing decisions
- **Execution Metadata**: Detailed information about task processing

### Health Monitoring
- **Individual Agent Status**: Monitor each agent's health independently
- **Workflow Integrity**: Ensure LangGraph workflow compilation
- **Resource Monitoring**: Track API usage and performance

## 🌟 Benefits Achieved

### **🎯 Specialized Excellence**
- Each agent optimized for specific domains
- Better accuracy through domain expertise
- Appropriate tool selection and model parameters

### **📈 Scalable Architecture**
- Easy to add new specialized agents
- Modular, testable components
- Resource-efficient lazy loading

### **🔒 Robust Operations**
- Graceful error handling and fallbacks
- Comprehensive health monitoring
- Production-ready logging and debugging

### **👥 Developer Experience**
- Clear separation of concerns
- Intuitive API design
- Comprehensive documentation and examples

This supervisor pattern implementation provides a robust, scalable foundation for multi-agent AI systems with clear specialization and intelligent task routing. The addition of the Transport Agent demonstrates the system's extensibility, seamlessly integrating real-time Singapore public transport data with conversational AI capabilities and follow-up query support for enhanced user experience.

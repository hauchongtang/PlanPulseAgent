from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.models import TelegramBase
from app.services.agent_service import agent_service


class TelegramMessageRequest(BaseModel):
    """Request model for telegram message processing."""
    chat_id: str
    chat_message: str


class TelegramMessageResponse(BaseModel):
    """Response model for telegram message processing."""
    chat_id: str
    chat_message: str
    response: str
    success: bool
    metadata: dict = None


router = APIRouter(prefix="/telegram", tags=["telegram"])


@router.post("/", response_model=TelegramMessageResponse)
async def pass_telegram_message_to_orchestrator(request: TelegramMessageRequest):
    """
    Process a telegram message through the AI agent orchestrator.
    
    Args:
        request: The telegram message request containing chat_id and message
        
    Returns:
        TelegramMessageResponse: The processed response from the agent
    """
    try:
        # Process the message through the agent service with user-specific memory
        result = agent_service.process_message(request.chat_message, user_id=request.chat_id)
        
        return TelegramMessageResponse(
            chat_id=request.chat_id,
            chat_message=request.chat_message,
            response=result.get("response", ""),
            success=result.get("success", False),
            metadata=result.get("supervisor_metadata", {})
        )
        
    except Exception as e:
        # Log the error (you might want to use proper logging here)
        print(f"Error processing telegram message: {str(e)}")
        
        return TelegramMessageResponse(
            chat_id=request.chat_id,
            chat_message=request.chat_message,
            response=f"Service temporarily unavailable: {str(e)}",
            success=False,
            metadata={"error": str(e)}
        )


@router.get("/health")
async def health_check():
    """Health check endpoint for the telegram service."""
    health_status = agent_service.health_check()
    
    if health_status.get("status") == "healthy":
        return health_status
    else:
        raise HTTPException(status_code=503, detail=health_status)


@router.get("/tools")
async def get_available_tools():
    """Get list of available agents and their capabilities."""
    agents = agent_service.get_available_agents()
    capabilities = agent_service.get_agent_capabilities()
    
    return {
        "available_agents": agents,
        "agent_count": len(agents),
        "capabilities": capabilities
    }


@router.get("/agents")
async def get_agent_info():
    """Get detailed information about all available agents."""
    return {
        "supervisor_pattern": True,
        "agents": agent_service.get_agent_capabilities(),
        "workflow": "Each request is analyzed and routed to the most appropriate specialized agent"
    }
    
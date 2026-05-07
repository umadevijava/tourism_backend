"""API endpoint for multi-agent workflows."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Response
import json
from typing import Optional

from backend.api.deps import ChatHistoryDep, LamaCppClientDep, VectorDatabaseDep
from chatbot.agents.orchestrator import AgentOrchestrator
from chatbot.helpers.log import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator(llm_client, vector_db=None) -> AgentOrchestrator:
    """Get or create the orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator(llm_client, vector_db)
    return _orchestrator


@router.websocket("/agents/workflow")
async def agent_workflow_stream(websocket: WebSocket):
    """
    WebSocket endpoint for streaming multi-agent workflows.
    
    Receives: {"goal": str, "query": str}
    Streams back agent thoughts, actions, and results
    """
    try:
        await websocket.accept()
        logger.info("✓ Agent workflow WebSocket connection accepted")
        
        while True:
            try:
                data = await websocket.receive_json()
                logger.info(f"Workflow request: {data}")
                
                # Get dependencies
                from backend.api.deps import get_llm_client, get_index
                
                llm_gen = get_llm_client()
                llm_client = next(llm_gen)
                
                index_gen = get_index()
                index = next(index_gen)
                
                # Get orchestrator
                orchestrator = get_orchestrator(llm_client, index)
                
                goal = data.get("goal", "")
                query = data.get("query", "")
                
                if not goal or not query:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Missing goal or query",
                    })
                    continue
                
                # Stream workflow execution
                async for update in orchestrator.process_with_feedback(goal, query):
                    logger.info(f"Sending update: {update['type']}")
                    
                    # Send update to client
                    await websocket.send_json({
                        "type": update["type"],
                        "data": update,
                    })
                
            except WebSocketDisconnect:
                logger.info("✓ WebSocket disconnected")
                break
            except json.JSONDecodeError:
                logger.error("Invalid JSON received")
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON format",
                })
            except Exception as e:
                logger.error(f"Workflow error: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e),
                })
    
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")


@router.get("/agents/status")
async def get_agents_status(llm_client: LamaCppClientDep):
    """Get current status of all agents."""
    orchestrator = get_orchestrator(llm_client)
    
    return {
        "status": "operational",
        "agents": orchestrator.get_agent_states(),
        "memory_summary": orchestrator.shared_memory.get_summary(),
        "workflow_history_count": len(orchestrator.workflow_history),
    }


@router.get("/agents/memory")
async def get_agents_memory(llm_client: LamaCppClientDep):
    """Get current shared memory state."""
    orchestrator = get_orchestrator(llm_client)
    return orchestrator.shared_memory.export()


@router.post("/agents/feedback")
async def submit_feedback(
    feedback: str,
    agent_id: Optional[str] = None,
    llm_client: LamaCppClientDep = None,
):
    """Submit feedback to improve agent behavior."""
    orchestrator = get_orchestrator(llm_client)
    orchestrator.update_feedback(feedback, agent_id)
    
    return {
        "status": "feedback_received",
        "message": "Thank you for the feedback",
    }


@router.delete("/agents/reset")
async def reset_orchestrator(llm_client: LamaCppClientDep):
    """Reset the orchestrator and all agents."""
    orchestrator = get_orchestrator(llm_client)
    await orchestrator.reset()
    
    return {
        "status": "reset_complete",
        "message": "All agents have been reset",
    }


@router.get("/agents/workflow-history")
async def get_workflow_history(
    limit: int = 10,
    llm_client: LamaCppClientDep = None,
):
    """Get recent workflow execution history."""
    orchestrator = get_orchestrator(llm_client)
    return {
        "count": len(orchestrator.workflow_history),
        "history": orchestrator.get_workflow_history(limit),
    }

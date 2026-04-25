"""Agent status and override routes."""

from __future__ import annotations

from fastapi import APIRouter

from triage.api.schemas import AgentOverrideRequest, ApiResponse
from triage.api.service import backend_service
from triage.env.state import AgentType


router = APIRouter()


@router.get("/message-bus/stats", response_model=ApiResponse)
async def get_message_bus_stats() -> ApiResponse:
    return ApiResponse(success=True, data=backend_service.get_message_bus_stats())


@router.get("/", response_model=ApiResponse)
async def list_agents() -> ApiResponse:
    return ApiResponse(success=True, data={"agents": backend_service.get_agent_statuses()})


@router.get("/safety/blocks", response_model=ApiResponse)
async def get_safety_blocks() -> ApiResponse:
    return ApiResponse(success=True, data={"blocks": backend_service.get_safety_blocks()})


@router.get("/safety/stats", response_model=ApiResponse)
async def get_safety_stats() -> ApiResponse:
    return ApiResponse(success=True, data=backend_service.get_safety_stats())


@router.get("/infection/status", response_model=ApiResponse)
async def get_infection_status() -> ApiResponse:
    return ApiResponse(success=True, data=backend_service.get_infection_status())


@router.get("/{agent_type}/status", response_model=ApiResponse)
async def get_agent_status(agent_type: AgentType) -> ApiResponse:
    return ApiResponse(success=True, data=backend_service.get_agent_status(agent_type))


@router.get("/{agent_type}/messages", response_model=ApiResponse)
async def get_agent_messages(agent_type: AgentType) -> ApiResponse:
    return ApiResponse(success=True, data={"messages": backend_service.get_agent_messages(agent_type)})


@router.post("/{agent_type}/override", response_model=ApiResponse)
async def override_agent(agent_type: AgentType, request: AgentOverrideRequest) -> ApiResponse:
    return ApiResponse(success=True, data=await backend_service.override_agent(agent_type, request))

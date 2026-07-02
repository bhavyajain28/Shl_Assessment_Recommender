"""FastAPI service: GET /health, POST /chat. Stateless -- every /chat call
carries the full conversation history and no per-conversation state is kept
between requests."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.agent import Agent
from app.config import get_settings
from app.models import ChatRequest, ChatResponse, HealthResponse
from app.utils import setup_logging

logger = logging.getLogger(__name__)

_agent: Agent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _agent
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Loading catalog, embeddings, and vector index...")
    start = time.time()
    _agent = Agent()
    logger.info("Ready in %.1fs (%d assessments indexed)", time.time() - start, len(_agent.retriever.get_all()))
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s: %s", request.url.path, exc)
    if request.url.path == "/chat":
        return JSONResponse(
            status_code=200,
            content=ChatResponse(
                reply="Something went wrong on my end processing that -- could you rephrase or try again?",
                recommendations=[],
                end_of_conversation=False,
            ).model_dump(),
        )
    return JSONResponse(status_code=500, content={"detail": "internal error"})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    global _agent
    if _agent is None:
        _agent = Agent()
    logger.info("chat request: %d messages", len(request.messages))
    try:
        response = _agent.handle_chat(request.messages)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent failed to handle chat turn: %s", exc)
        return ChatResponse(
            reply="Something went wrong on my end processing that -- could you rephrase or try again?",
            recommendations=[],
            end_of_conversation=False,
        )
    logger.info("chat response: %d recommendations, end_of_conversation=%s", len(response.recommendations), response.end_of_conversation)
    return response

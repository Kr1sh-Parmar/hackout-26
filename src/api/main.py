"""FastAPI app factory: CORS, the domain-error handler, and structured
logging set up on startup."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.logging import setup_logging
from .deps import get_settings
from .errors import DomainError
from .routes import actions, backtest, events, explain, forecast, meta, outlook, storage


@asynccontextmanager
async def _lifespan(app: FastAPI):
    setup_logging(get_settings().log_level)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Renewable Forecast Platform API", lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(DomainError)
    async def _domain_error_handler(request: Request, exc: DomainError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": type(exc).__name__, "message": exc.msg, "hint": exc.hint},
        )

    for module in (meta, forecast, outlook, events, actions, storage, backtest, explain):
        app.include_router(module.router)

    return app


app = create_app()

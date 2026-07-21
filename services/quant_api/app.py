from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from services.quant_api.config import ApiSettings
from services.quant_api.events.routes import router as events_router
from services.quant_api.routes import router
from services.quant_api.services import ServiceContainer


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    service_container = ServiceContainer.create(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service_container.start()
        try:
            yield
        finally:
            service_container.stop()

    app = FastAPI(
        title="QuantAgent Research API",
        version="0.1.0",
        description="Read-only-by-default adapter API for QuantAgent runtime research artifacts.",
        lifespan=lifespan,
    )
    app.state.services = service_container
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(events_router)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    frontend_root = app.state.services.settings.project_root / "apps" / "quant-ui" / "dist"
    assets_root = frontend_root / "assets"
    if assets_root.exists():
        app.mount("/assets", StaticFiles(directory=assets_root), name="quant-ui-assets")

    if (frontend_root / "index.html").exists():
        @app.get("/{full_path:path}", include_in_schema=False)
        async def quant_ui_spa(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(404, "API route not found")
            candidate = (frontend_root / full_path).resolve()
            if (
                candidate != frontend_root.resolve()
                and frontend_root.resolve() in candidate.parents
                and candidate.is_file()
            ):
                return FileResponse(candidate)
            return FileResponse(frontend_root / "index.html")

    return app


app = create_app()

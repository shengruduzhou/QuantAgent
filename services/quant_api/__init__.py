"""FastAPI adapter layer for QuantAgent runtime artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

    from services.quant_api.config import ApiSettings


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Create the API lazily so launcher arguments can configure storage first."""
    from services.quant_api.app import create_app as app_factory

    return app_factory(settings)


__all__ = ["create_app"]

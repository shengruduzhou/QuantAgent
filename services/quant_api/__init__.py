"""FastAPI adapter layer for QuantAgent runtime artifacts."""

from services.quant_api.app import create_app

__all__ = ["create_app"]

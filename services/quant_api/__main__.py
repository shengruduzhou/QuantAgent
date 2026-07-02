from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "services.quant_api.app:app",
        host=os.environ.get("QUANT_UI_HOST", "127.0.0.1"),
        port=int(os.environ.get("QUANT_UI_PORT", "8000")),
        reload=os.environ.get("QUANT_UI_RELOAD", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()

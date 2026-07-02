from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.quant_api.app import create_app


async def main() -> int:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    checks = [
        "/health",
        "/api/system/overview",
        "/api/backtests",
        "/api/factors?query=alpha016",
        "/api/models",
        "/api/selection/runs",
        "/api/risk/overview",
        "/api/do-t/sources",
    ]
    results = []
    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://quant-ui.local",
        timeout=120.0,
    ) as client:
        for path in checks:
            response = await client.get(path)
            payload = response.json()
            results.append({
                "path": path,
                "http_status": response.status_code,
                "data_status": payload.get("status", payload.get("status")),
                "passed": response.status_code == 200,
            })
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item["passed"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

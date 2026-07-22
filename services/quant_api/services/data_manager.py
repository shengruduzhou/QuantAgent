from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.util import find_spec
import os
from typing import Any

from services.quant_api.config import ApiSettings


@dataclass(frozen=True)
class DataProviderSpec:
    id: str
    label: str
    module: str | None
    command_id: str | None
    asset_classes: tuple[str, ...]
    intervals: tuple[str, ...]
    operations: tuple[str, ...]
    requires: tuple[str, ...] = ()
    note: str = ""


PROVIDERS: tuple[DataProviderSpec, ...] = (
    DataProviderSpec(
        id="akshare_market",
        label="AkShare A股行情",
        module="akshare",
        command_id="build-akshare-market-panel-v7",
        asset_classes=("A股",),
        intervals=("1d",),
        operations=("download", "update"),
        note="显式股票范围；复权日线；PIT available_at 在下一工作日可用。",
    ),
    DataProviderSpec(
        id="qlib_local",
        label="Qlib 本地历史库",
        module="qlib",
        command_id="build-market-panel-v7",
        asset_classes=("A股",),
        intervals=("1d",),
        operations=("query", "export"),
        note="读取已安装的本地 Qlib provider；网页任务不会隐式下载官方数据包。",
    ),
    DataProviderSpec(
        id="tushare_fundamentals",
        label="TuShare PIT 财务",
        module="tushare",
        command_id="build-fundamentals-v7",
        asset_classes=("A股财务",),
        intervals=("report",),
        operations=("download", "update"),
        requires=("TUSHARE_TOKEN",),
        note="Token 仅从后端环境变量读取，永不回传浏览器。",
    ),
    DataProviderSpec(
        id="runtime_catalog",
        label="QuantAgent Runtime",
        module=None,
        command_id=None,
        asset_classes=("PIT artifact",),
        intervals=("manifest",),
        operations=("preview", "lineage", "cleanup"),
        note="现有 RuntimeIndexer、manifest 和受保护 Cleanup 是唯一数据目录。",
    ),
)


class DataManagerService:
    """Read-only provider registry for the governed data job surface."""

    def __init__(self, settings: ApiSettings) -> None:
        self.settings = settings

    def overview(self) -> dict[str, Any]:
        return {
            "providers": [self._provider_payload(spec) for spec in PROVIDERS],
            "constraints": [
                "network access requires an explicit operator confirmation",
                "downloads require explicit symbols or a project-scoped symbols file",
                "all Web job outputs must remain inside Runtime",
                "provider results must pass their existing PIT and schema gates",
                "successful jobs invalidate the canonical RuntimeIndexer cache",
                "deletion remains available only through backend-approved Runtime Cleanup candidates",
            ],
            "jobEndpoint": "/api/jobs/data",
            "supportsCancellation": True,
            "runtimeRoot": "runtime",
        }

    def _provider_payload(self, spec: DataProviderSpec) -> dict[str, Any]:
        installed = True if spec.module is None else _module_available(spec.module)
        missing_requirements = [name for name in spec.requires if not os.getenv(name)]
        configured = installed and not missing_requirements
        status = "ready" if configured else "needs_configuration" if installed else "unavailable"
        payload = asdict(spec)
        payload.update(
            {
                "commandId": payload.pop("command_id"),
                "assetClasses": list(payload.pop("asset_classes")),
                "installed": installed,
                "configured": configured,
                "status": status,
                "missingRequirements": missing_requirements,
            }
        )
        payload["intervals"] = list(payload["intervals"])
        payload["operations"] = list(payload["operations"])
        payload["requires"] = list(payload["requires"])
        return payload


def _module_available(module: str) -> bool:
    try:
        return find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False

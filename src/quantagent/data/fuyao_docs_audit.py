"""Live Fuyao documentation drift auditor.

The public ``llms-full.txt`` file is the vendor-generated aggregate contract.
Before an exhaustive acquisition run, compare the live documented REST paths,
market-dump paths and MCP tool table against QuantAgent's checked-in registry.
This makes a newly published vendor capability a fail-closed event instead of a
silent data gap.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

import requests

from quantagent.data.fuyao_catalog import DUMP_CAPABILITIES, REST_CAPABILITIES


LLMS_FULL_URL = "https://fuyao.aicubes.cn/llms-full.txt"

_REST_RE = re.compile(r"\bGET (/api/[A-Za-z0-9][A-Za-z0-9/_-]*)")
_DUMP_RE = re.compile(r"\bGET (/dump/market-dumps/[A-Za-z0-9][A-Za-z0-9/_-]*)")
_TOOL_RE = re.compile(r"\b(get_[a-z0-9_]+)\b")


def parse_llms_full_contract(text: str) -> dict[str, set[str]]:
    """Extract unique live data routes and MCP tools from the aggregate docs."""
    rest_paths = set(_REST_RE.findall(text))
    dump_paths = set(_DUMP_RE.findall(text))

    tools_section = ""
    marker = "## 工具一览"
    if marker in text:
        tools_section = text.split(marker, 1)[1]
        for end_marker in ("## AI Agent", "# MCP 工具"):
            if end_marker in tools_section:
                tools_section = tools_section.split(end_marker, 1)[0]
                break
    mcp_tools = set(_TOOL_RE.findall(tools_section))
    return {"rest_paths": rest_paths, "dump_paths": dump_paths, "mcp_tools": mcp_tools}


def compare_live_contract(parsed: dict[str, set[str]]) -> dict[str, Any]:
    registry_rest = {str(cap.rest_path) for cap in REST_CAPABILITIES if cap.rest_path}
    registry_dumps = {str(cap.rest_path) for cap in DUMP_CAPABILITIES if cap.rest_path}
    registry_tools = {str(cap.mcp_tool) for cap in REST_CAPABILITIES if cap.mcp_tool}

    docs_rest = set(parsed.get("rest_paths", set()))
    docs_dumps = set(parsed.get("dump_paths", set()))
    docs_tools = set(parsed.get("mcp_tools", set()))
    diffs = {
        "rest_only_in_docs": sorted(docs_rest - registry_rest),
        "rest_only_in_registry": sorted(registry_rest - docs_rest),
        "dumps_only_in_docs": sorted(docs_dumps - registry_dumps),
        "dumps_only_in_registry": sorted(registry_dumps - docs_dumps),
        "mcp_only_in_docs": sorted(docs_tools - registry_tools),
        "mcp_only_in_registry": sorted(registry_tools - docs_tools),
    }
    return {
        "ok": not any(diffs.values()),
        "counts": {
            "docs_rest": len(docs_rest),
            "registry_rest": len(registry_rest),
            "docs_dumps": len(docs_dumps),
            "registry_dumps": len(registry_dumps),
            "docs_mcp": len(docs_tools),
            "registry_mcp": len(registry_tools),
        },
        "diffs": diffs,
    }


def audit_live_documentation(*, timeout: float = 30.0) -> dict[str, Any]:
    """Download the current official contract and fail visibly on schema drift."""
    response = requests.get(LLMS_FULL_URL, timeout=timeout)
    response.raise_for_status()
    parsed = parse_llms_full_contract(response.text)
    result = compare_live_contract(parsed)
    return {
        "source": LLMS_FULL_URL,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        **result,
    }


__all__ = [
    "LLMS_FULL_URL",
    "audit_live_documentation",
    "compare_live_contract",
    "parse_llms_full_contract",
]

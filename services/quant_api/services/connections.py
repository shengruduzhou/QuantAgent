from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from threading import RLock
from typing import Mapping


@dataclass(frozen=True)
class ConnectionSpec:
    id: str
    label: str
    variables: tuple[str, ...]
    capabilities: tuple[str, ...]
    note: str


CONNECTIONS: tuple[ConnectionSpec, ...] = (
    ConnectionSpec(
        id="tickflow",
        label="TickFlow",
        variables=("TICKFLOW_API_KEY",),
        capabilities=("daily", "minute", "tick", "depth"),
        note="A股行情采集；会话密钥仅注入白名单数据任务。",
    ),
    ConnectionSpec(
        id="tushare",
        label="TuShare",
        variables=("TUSHARE_TOKEN",),
        capabilities=("pit_fundamentals",),
        note="PIT 财务数据；不会回传或写入 Runtime。",
    ),
    ConnectionSpec(
        id="openai",
        label="OpenAI Research",
        variables=("OPENAI_API_KEY",),
        capabilities=("factor_proposal",),
        note="只用于显式开启网络的研究因子提案，不参与订单链。",
    ),
    ConnectionSpec(
        id="alpaca",
        label="Alpaca Market Data",
        variables=("ALPACA_API_KEY", "ALPACA_API_SECRET"),
        capabilities=("market_data", "paper_only"),
        note="预留市场数据/纸面连接；当前没有实盘路由。",
    ),
)

CREDENTIAL_VARIABLES = frozenset(
    variable
    for spec in CONNECTIONS
    for variable in spec.variables
)


class ConnectionManager:
    """Process-memory credential vault for allowlisted research connectors.

    Values are never serialized.  Public payloads contain only source and a
    short one-way fingerprint so the UI can distinguish sessions without
    learning the secret.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._session_values: dict[str, str] = {}

    def list(self) -> list[dict]:
        return [self._public(spec) for spec in CONNECTIONS]

    def connect(self, provider_id: str, credentials: Mapping[str, str]) -> dict:
        spec = self._spec(provider_id)
        unknown = set(credentials) - set(spec.variables)
        if unknown:
            raise ValueError(f"unsupported credential fields: {sorted(unknown)}")
        missing = [name for name in spec.variables if not str(credentials.get(name) or "").strip()]
        if missing:
            raise ValueError(f"missing credential fields: {missing}")
        normalized: dict[str, str] = {}
        for name in spec.variables:
            value = str(credentials[name]).strip()
            if len(value) < 8 or len(value) > 4_096 or any(char in value for char in "\r\n\x00"):
                raise ValueError(f"invalid credential value for {name}")
            normalized[name] = value
        with self._lock:
            self._session_values.update(normalized)
        return self._public(spec)

    def disconnect(self, provider_id: str) -> dict:
        spec = self._spec(provider_id)
        with self._lock:
            for name in spec.variables:
                self._session_values.pop(name, None)
        return self._public(spec)

    def has_variable(self, name: str) -> bool:
        with self._lock:
            return bool(self._session_values.get(name) or os.getenv(name))

    def environment_for(self, provider_ids: set[str] | tuple[str, ...]) -> dict[str, str]:
        allowed = {
            variable
            for spec in CONNECTIONS
            if spec.id in provider_ids
            for variable in spec.variables
        }
        with self._lock:
            session = {name: value for name, value in self._session_values.items() if name in allowed}
        resolved = {
            name: value
            for name in allowed
            if (value := session.get(name) or os.getenv(name))
        }
        return resolved

    def _public(self, spec: ConnectionSpec) -> dict:
        with self._lock:
            session_values = {
                name: self._session_values.get(name)
                for name in spec.variables
                if self._session_values.get(name)
            }
        environment_values = {
            name: os.getenv(name)
            for name in spec.variables
            if os.getenv(name)
        }
        connected = all(name in session_values or name in environment_values for name in spec.variables)
        source = "session" if connected and all(name in session_values for name in spec.variables) else "environment" if connected else "none"
        fingerprints = {
            name: sha256((session_values.get(name) or environment_values.get(name) or "").encode("utf-8")).hexdigest()[:10]
            for name in spec.variables
            if name in session_values or name in environment_values
        }
        payload = asdict(spec)
        payload.update(
            {
                "variables": list(spec.variables),
                "capabilities": list(spec.capabilities),
                "connected": connected,
                "source": source,
                "fingerprints": fingerprints,
                "persistence": "process_memory" if source == "session" else "server_environment" if source == "environment" else "none",
            }
        )
        return payload

    @staticmethod
    def _spec(provider_id: str) -> ConnectionSpec:
        for spec in CONNECTIONS:
            if spec.id == provider_id:
                return spec
        raise KeyError(provider_id)

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path


@dataclass
class KillSwitch:
    """Fail-closed operational kill switch with optional durable state.

    ``state_path=None`` preserves the lightweight in-memory behaviour used by
    research/backtests.  A production caller should provide ``state_path`` so a
    process restart cannot silently clear loss/reconciliation/provider/audit
    blocks.  Persistent instances also require an explicit reason to release a
    block; a bare ``release()`` is rejected instead of clearing everything.
    """

    manual_triggered: bool = False
    reasons: list[str] = field(default_factory=list)
    state_path: str | Path | None = None
    state_version: int = 1

    def __post_init__(self) -> None:
        if self.state_path is not None:
            self.state_path = Path(self.state_path)
            self._restore()

    @property
    def triggered(self) -> bool:
        return self.manual_triggered or bool(self.reasons)

    @property
    def persistent(self) -> bool:
        return self.state_path is not None

    def trigger(self, reason: str) -> None:
        reason = str(reason).strip()
        if not reason:
            raise ValueError("kill-switch trigger reason must be non-empty")
        if reason not in self.reasons:
            self.reasons.append(reason)
        self._persist()

    def manual_trigger(self, reason: str = "manual_operator_trigger") -> None:
        self.manual_triggered = True
        if reason and reason not in self.reasons:
            self.reasons.append(reason)
        self._persist()

    def release(self, reason: str | None = None) -> None:
        """Release one cause; persistent state cannot be globally cleared by accident."""
        if reason is None:
            if self.persistent:
                raise ValueError(
                    "persistent kill switch requires an explicit release reason; "
                    "use release_all(confirm=True) for a deliberate full reset"
                )
            self.manual_triggered = False
            self.reasons.clear()
            return
        target = str(reason).strip()
        if not target:
            raise ValueError("release reason must be non-empty")
        self.reasons = [item for item in self.reasons if item != target]
        if target == "manual_operator_trigger":
            self.manual_triggered = False
        self._persist()

    def release_all(self, *, confirm: bool = False) -> None:
        if not confirm:
            raise ValueError("full kill-switch release requires confirm=True")
        self.manual_triggered = False
        self.reasons.clear()
        self._persist()

    def evaluate(
        self,
        *,
        daily_loss: float = 0.0,
        drawdown: float = 0.0,
        reconciliation_mismatch: bool = False,
        provider_failure: bool = False,
        audit_write_failure: bool = False,
        rejection_rate: float = 0.0,
        turnover: float = 0.0,
        max_daily_loss: float = 0.03,
        max_drawdown: float = 0.15,
        max_rejection_rate: float = 0.50,
        max_turnover: float = 0.50,
    ) -> bool:
        if daily_loss <= -max_daily_loss:
            self.trigger("severe_daily_loss")
        if drawdown <= -max_drawdown:
            self.trigger("severe_drawdown")
        if reconciliation_mismatch:
            self.trigger("severe_reconciliation_mismatch")
        if provider_failure:
            self.trigger("data_provider_failure")
        if audit_write_failure:
            self.trigger("audit_write_failure")
        if rejection_rate > max_rejection_rate:
            self.trigger("excessive_rejection")
        if turnover > max_turnover:
            self.trigger("abnormal_turnover")
        return self.triggered

    def status(self) -> dict[str, object]:
        return {
            "triggered": self.triggered,
            "reasons": tuple(self.reasons),
            "manual": self.manual_triggered,
            "persistent": self.persistent,
            "state_path": str(self.state_path) if self.state_path is not None else None,
            "state_version": self.state_version,
        }

    def _restore(self) -> None:
        path = Path(self.state_path)  # type: ignore[arg-type]
        if not path.exists():
            self._persist()
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or int(payload.get("version", -1)) != self.state_version:
                raise ValueError("unsupported kill-switch state schema")
            restored_reasons = payload.get("reasons")
            if not isinstance(restored_reasons, list):
                raise ValueError("kill-switch reasons must be a list")
            self.reasons = list(dict.fromkeys(str(item) for item in restored_reasons if str(item).strip()))
            self.manual_triggered = bool(payload.get("manual_triggered", False))
        except Exception:
            # Corrupt/mismatched state is itself an operational-risk condition.
            # Do not start green because the state file could not be understood.
            self.manual_triggered = True
            self.reasons = ["kill_switch_state_unreadable"]
            self._persist()

    def _persist(self) -> None:
        if self.state_path is None:
            return
        path = Path(self.state_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self.state_version,
            "manual_triggered": bool(self.manual_triggered),
            "reasons": list(self.reasons),
        }
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

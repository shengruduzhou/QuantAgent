from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KillSwitch:
    manual_triggered: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def triggered(self) -> bool:
        return self.manual_triggered or bool(self.reasons)

    def trigger(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    def release(self, reason: str | None = None) -> None:
        if reason is None:
            self.manual_triggered = False
            self.reasons.clear()
            return
        self.reasons = [item for item in self.reasons if item != reason]

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
        return {"triggered": self.triggered, "reasons": tuple(self.reasons), "manual": self.manual_triggered}


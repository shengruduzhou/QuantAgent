"""Request shapes for the paper order submission path.

`extra="forbid"` matters more here than elsewhere: an unrecognised field on an
economic request is either a client bug or an attempt to reach a capability this
path does not have, and silently dropping it would let a caller believe it had
asked for something.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PaperModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class PaperOrderSubmission(PaperModel):
    """One economic instruction.

    `idempotencyKey` is required with no default. A server-generated default would
    make every retry look like a new order, which is the failure this whole path
    exists to prevent — so the client must state the identity of its own request.
    """

    idempotency_key: str = Field(
        min_length=1, max_length=200, alias="idempotencyKey",
    )
    run_id: str = Field(min_length=1, max_length=120, alias="runId")
    symbol: str = Field(min_length=1, max_length=32)
    side: Literal["BUY", "SELL", "buy", "sell"]
    quantity: int = Field(gt=0, le=100_000_000)
    #: Every order carries a worst price. There is no market-order form: an
    #: unbounded fill on an A-share limit board books profit that was never
    #: obtainable.
    limit_price: float = Field(gt=0, alias="limitPrice")
    trade_date: str = Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$", alias="tradeDate",
    )
    strategy_version_id: str = Field("", max_length=120, alias="strategyVersionId")
    #: Required, and deliberately not defaulted. The signal is *economic* identity:
    #: it is what lets the order-intent guard tell "the same decision, delivered
    #: twice" from "two decisions that happen to want the same trade". Defaulting it
    #: to a constant would collapse two legitimate sleeve orders into one; deriving
    #: it from `idempotencyKey` would make the economic guard depend on delivery
    #: identity, so a client that lost its key could trade the same intent twice.
    signal_id: str = Field(min_length=1, max_length=200, alias="signalId")


class PaperOrderCancellation(PaperModel):
    idempotency_key: str = Field(min_length=1, max_length=200, alias="idempotencyKey")
    run_id: str = Field(min_length=1, max_length=120, alias="runId")


__all__ = ["PaperOrderCancellation", "PaperOrderSubmission"]

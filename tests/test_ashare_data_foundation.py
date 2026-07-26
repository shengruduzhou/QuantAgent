"""Unit coverage for the A-share data foundation.

These tests use fixtures and fakes deliberately — they pin the contracts
(identity, units, provenance, fallback, resume, fail-loud behaviour) that the
live acquisition depends on. Real-network evidence lives in the capability
matrix and validation report produced by the scripts, not here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

pd = pytest.importorskip("pandas")

from quantagent.data.ashare import contracts  # noqa: E402
from quantagent.data.ashare.acquire import (  # noqa: E402
    BarAcquisition,
    ProviderSpec,
    completed_symbols,
    terminal_failures,
    trading_day_cutoff,
)
from quantagent.data.ashare.env import load_repo_env  # noqa: E402
from quantagent.data.ashare.http import (  # noqa: E402
    RETRY_EMPTY,
    RETRY_ENTITLEMENT,
    RETRY_OK,
    RETRY_PERMANENT,
)
from quantagent.data.ashare.sources import SourceResult, TencentSource, TickFlowSource  # noqa: E402
from quantagent.data.ashare.symbols import (  # noqa: E402
    SymbolError,
    board_of,
    canonical_symbol,
    classify_code,
    identify,
    is_bse_legacy_code,
)


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("code,exchange,board", [
    ("600519", "SSE", "SH_Main"),
    ("601318", "SSE", "SH_Main"),
    ("688981", "SSE", "STAR"),
    ("689009", "SSE", "STAR"),
    ("000001", "SZSE", "SZ_Main"),
    ("002594", "SZSE", "SZ_Main"),      # former SME board, merged 2021-04-06
    ("003816", "SZSE", "SZ_Main"),
    ("300750", "SZSE", "ChiNext"),
    ("301618", "SZSE", "ChiNext"),
    ("920079", "BSE", "BSE"),
    ("830799", "BSE", "BSE"),           # legacy NEEQ-select code
    ("871981", "BSE", "BSE"),
])
def test_board_classification_covers_every_board(code, exchange, board):
    assert classify_code(code)[:2] == (exchange, board)
    assert board_of(code) == board


def test_bse_legacy_versus_current_code_ranges_are_distinguished():
    assert is_bse_legacy_code("830799.BJ") is True
    assert is_bse_legacy_code("920079.BJ") is False


@pytest.mark.parametrize("spelling", [
    "600519", "600519.SH", "sh600519", "SH600519", "600519.XSHG", "600519_SH",
])
def test_symbol_spellings_normalise_to_one_canonical_form(spelling):
    assert canonical_symbol(spelling) == "600519.SH"


def test_contradicting_exchange_suffix_raises_instead_of_relocating_the_security():
    # A vendor suffix must never be able to move an SSE code onto SZSE.
    with pytest.raises(SymbolError):
        identify("600519.SZ")


def test_b_shares_are_typed_but_kept_on_their_board():
    ident = identify("900901")
    assert (ident.board, ident.security_type) == ("SH_Main", "B_share")


def test_vendor_code_helpers_match_each_vendor_convention():
    ident = identify("300750.SZ")
    assert ident.tencent_code == "sz300750"
    assert ident.eastmoney_secid == "0.300750"
    assert identify("600519.SH").eastmoney_secid == "1.600519"


# --------------------------------------------------------------------------
# contracts and units
# --------------------------------------------------------------------------
def test_every_contract_declares_provenance_columns():
    for name, contract in contracts.CONTRACTS.items():
        assert set(contracts.PROVENANCE_COLUMNS).issubset(set(contract.columns)), name


def test_daily_bar_contract_declares_units_explicitly():
    contract = contracts.DAILY_BARS
    assert contract.volume_unit == contracts.VOLUME_SHARES
    assert contract.amount_unit == contracts.AMOUNT_CNY
    assert contract.adjustment == contracts.ADJUST_NONE
    assert contract.timezone == contracts.TIMEZONE_CST


def test_quote_contract_is_labelled_level_one_not_level_two():
    assert "NOT order-by-order Level-2" in contracts.QUOTES.notes


# --------------------------------------------------------------------------
# source parsing and unit normalisation
# --------------------------------------------------------------------------
class _FakeOutcome:
    def __init__(self, payload=None, text=None, ok=True, retry_class=RETRY_OK):
        self.payload, self.text, self.ok = payload, text, ok
        self.retry_class = retry_class
        self.endpoint = "fake://endpoint"
        self.retrieved_at = "2026-07-26T00:00:00+00:00"
        self.error = None
        self.latency_s = 0.01
        self.status_code = 200


class _FakeClient:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0

    def get_json(self, url, params=None, headers=None):
        self.calls += 1
        return self._outcomes[min(self.calls - 1, len(self._outcomes) - 1)]

    def get(self, url, params=None, headers=None, encoding=None):
        return self.get_json(url, params, headers)


def test_tencent_daily_bars_converts_lots_to_shares():
    payload = {"code": 0, "data": {"sh600519": {"day": [
        ["2026-07-23", "1290.00", "1292.01", "1295.00", "1288.00", "35699.000"],
    ]}}}
    source = TencentSource(_FakeClient([_FakeOutcome(payload), _FakeOutcome({"code": 0, "data": []})]))
    result = source.daily_bars("600519.SH", "2026-07-01", "2026-07-24")
    assert result.rows == 1
    row = result.frame.iloc[0]
    # vendor reports 手 (lots); the adapter must publish shares
    assert row["volume"] == pytest.approx(35699.0 * 100)
    assert row["symbol"] == "600519.SH"
    assert row["source"] == "tencent"
    assert row["available_at"].endswith("15:00:00")


def test_tencent_empty_payload_is_empty_not_a_silent_success():
    source = TencentSource(_FakeClient([_FakeOutcome({"code": 0, "data": []})]))
    result = source.daily_bars("600519.SH", "2026-07-01", "2026-07-24")
    assert result.rows == 0
    assert result.retry_class == RETRY_EMPTY
    assert result.frame.empty
    assert not result.ok


def test_tencent_vendor_error_code_is_a_permanent_failure():
    source = TencentSource(_FakeClient([_FakeOutcome({"code": -1, "msg": "bad param"})]))
    result = source.daily_bars("600519.SH", "2026-07-01", "2026-07-24")
    assert result.retry_class == RETRY_PERMANENT
    assert "bad param" in (result.error or "")


def test_tickflow_entitlement_error_is_classified_not_swallowed():
    class _Klines:
        def get(self, *_args, **_kwargs):
            raise PermissionError("无市场深度查询权限 (市场: CN)")

    class _Client:
        klines = _Klines()

    result = TickFlowSource(_Client()).daily_bars("600519.SH")
    assert result.retry_class == RETRY_ENTITLEMENT
    assert result.rows == 0


def test_tickflow_client_without_credential_fails_loudly(monkeypatch):
    from quantagent.data.ashare import sources

    monkeypatch.delenv("TICKFLOW_API_KEY", raising=False)
    monkeypatch.setattr(sources, "load_repo_env", lambda *a, **k: [], raising=False)
    monkeypatch.setattr("quantagent.data.ashare.env.load_repo_env", lambda *a, **k: [])
    with pytest.raises(RuntimeError) as excinfo:
        sources.build_tickflow_client()
    message = str(excinfo.value)
    assert "TICKFLOW_API_KEY" in message
    # the error must say we will NOT quietly use another provider
    assert "silently" in message


# --------------------------------------------------------------------------
# env loading (the root cause of the KeyError-killed backfill)
# --------------------------------------------------------------------------
def test_repo_env_loading_does_not_override_an_exported_value(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FOO_TOKEN='from-file'\n# comment\nBAR_TOKEN=bar\n")
    monkeypatch.setenv("FOO_TOKEN", "from-shell")
    monkeypatch.delenv("BAR_TOKEN", raising=False)
    injected = load_repo_env(env_file, force=True)
    import os

    assert os.environ["FOO_TOKEN"] == "from-shell"
    assert os.environ["BAR_TOKEN"] == "bar"
    assert "BAR_TOKEN" in injected and "FOO_TOKEN" not in injected


# --------------------------------------------------------------------------
# acquisition: fallback, provenance, resume, idempotency
# --------------------------------------------------------------------------
def _result(symbol, rows, retry_class=RETRY_OK, source="fake"):
    if rows:
        frame = pd.DataFrame({
            "symbol": [symbol] * rows,
            "trade_date": pd.date_range("2026-07-01", periods=rows, freq="D"),
            "open": 1.0, "high": 1.2, "low": 0.9, "close": 1.1,
            "volume": 100.0, "amount": 110.0, "source": source,
            "source_endpoint": "fake", "retrieved_at": "2026-07-26T00:00:00+00:00",
            "available_at": "2026-07-01 15:00:00", "quality_status": "OK",
        })
    else:
        frame = pd.DataFrame(columns=list(contracts.DAILY_BARS.columns))
    return SourceResult(frame, source, "fake", retry_class, "2026-07-26T00:00:00+00:00", rows)


def test_acquisition_falls_back_and_records_why_the_primary_failed(tmp_path):
    calls = {"primary": [], "secondary": []}

    def primary(symbol):
        calls["primary"].append(symbol)
        return _result(symbol, 0, RETRY_EMPTY, "primary")

    def secondary(symbol):
        calls["secondary"].append(symbol)
        return _result(symbol, 3, RETRY_OK, "secondary")

    staging, ledger = tmp_path / "staging", tmp_path / "ledger.csv"
    worker = BarAcquisition(staging, ledger,
                            [ProviderSpec("primary", primary), ProviderSpec("secondary", secondary)],
                            log=lambda _m: None)
    report = worker.run(["600519.SH"], max_minutes=1)
    assert report.written == 1
    assert report.by_provider == {"secondary": 1}
    assert calls["primary"] == ["600519.SH"] and calls["secondary"] == ["600519.SH"]

    rows = pd.read_csv(ledger)
    # the losing provider's verdict is preserved: "we asked and it was empty"
    assert set(rows["provider"]) == {"primary", "secondary"}
    assert rows.loc[rows["provider"] == "primary", "retry_class"].iloc[0] == RETRY_EMPTY
    written = pd.read_parquet(staging / "sym_600519_SH.parquet")
    assert written["source"].unique().tolist() == ["secondary"]


def test_acquisition_never_blends_two_providers_inside_one_symbol(tmp_path):
    worker = BarAcquisition(
        tmp_path / "s", tmp_path / "l.csv",
        [ProviderSpec("a", lambda s: _result(s, 2, RETRY_OK, "a")),
         ProviderSpec("b", lambda s: _result(s, 5, RETRY_OK, "b"))],
        log=lambda _m: None)
    worker.run(["600519.SH"], max_minutes=1)
    frame = pd.read_parquet(tmp_path / "s" / "sym_600519_SH.parquet")
    assert frame["source"].nunique() == 1, "a symbol partition must come from one provider"


def test_resume_skips_completed_partitions_and_is_idempotent(tmp_path):
    seen: list[str] = []

    def provider(symbol):
        seen.append(symbol)
        return _result(symbol, 2)

    staging, ledger = tmp_path / "s", tmp_path / "l.csv"
    spec = [ProviderSpec("p", provider)]
    first = BarAcquisition(staging, ledger, spec, log=lambda _m: None).run(
        ["600519.SH", "000001.SZ"], max_minutes=1)
    assert first.written == 2
    assert completed_symbols(staging) == {"600519.SH", "000001.SZ"}

    second = BarAcquisition(staging, ledger, spec, log=lambda _m: None).run(
        ["600519.SH", "000001.SZ"], max_minutes=1)
    assert second.written == 0 and second.skipped_existing == 2
    assert seen == ["600519.SH", "000001.SZ"], "a resume must not refetch completed symbols"


def test_permanent_failures_are_not_retried_on_resume(tmp_path):
    attempts: list[str] = []

    def provider(symbol):
        attempts.append(symbol)
        return _result(symbol, 0, RETRY_ENTITLEMENT)

    staging, ledger = tmp_path / "s", tmp_path / "l.csv"
    spec = [ProviderSpec("p", provider)]
    BarAcquisition(staging, ledger, spec, log=lambda _m: None).run(["600519.SH"], max_minutes=1)
    assert terminal_failures(ledger) == {"600519.SH"}
    BarAcquisition(staging, ledger, spec, log=lambda _m: None).run(["600519.SH"], max_minutes=1)
    assert attempts == ["600519.SH"], "an entitlement failure must not be retried forever"


def test_cancel_file_stops_a_run_without_losing_completed_partitions(tmp_path):
    cancel = tmp_path / "job.cancel"
    cancel.write_text("stop")
    worker = BarAcquisition(tmp_path / "s", tmp_path / "l.csv",
                            [ProviderSpec("p", lambda s: _result(s, 1))],
                            cancel_file=cancel, log=lambda _m: None)
    report = worker.run(["600519.SH"], max_minutes=1)
    assert report.stopped_reason == "cancelled" and report.written == 0


def test_provider_exception_is_recorded_and_does_not_abort_the_run(tmp_path):
    def exploding(symbol):
        raise ValueError("vendor exploded")

    staging, ledger = tmp_path / "s", tmp_path / "l.csv"
    worker = BarAcquisition(staging, ledger,
                            [ProviderSpec("boom", exploding),
                             ProviderSpec("ok", lambda s: _result(s, 1))],
                            log=lambda _m: None)
    report = worker.run(["600519.SH", "000001.SZ"], max_minutes=1)
    assert report.written == 2
    rows = pd.read_csv(ledger)
    assert (rows["detail"].astype(str).str.contains("vendor exploded")).any()


def test_current_day_is_not_fetched_before_the_close_is_published():
    before_close = pd.Timestamp("2026-07-24 10:30", tz="Asia/Shanghai")
    after_close = pd.Timestamp("2026-07-24 16:30", tz="Asia/Shanghai")
    assert trading_day_cutoff(before_close) == pd.Timestamp("2026-07-23")
    assert trading_day_cutoff(after_close) == pd.Timestamp("2026-07-24")


# --------------------------------------------------------------------------
# no synthetic fallback anywhere in the production data path
# --------------------------------------------------------------------------
def test_no_module_in_the_foundation_imports_the_mock_provider():
    package = REPO / "src/quantagent/data/ashare"
    offenders = [p.name for p in package.glob("*.py")
                 if "mock_provider" in p.read_text() or "MockProvider" in p.read_text()]
    assert offenders == [], f"synthetic data must not be reachable from {offenders}"


def test_acquisition_scripts_require_explicit_network_approval():
    for name in ("u0_acquire_bars.py", "u0_acquire_intraday.py", "u0_security_master.py",
                 "ashare_capability_probe.py"):
        text = (REPO / "scripts" / name).read_text()
        assert "--allow-network" in text, name
        assert "refusing" in text, f"{name} must refuse to run without network approval"


def test_capability_probe_treats_import_success_as_insufficient_evidence():
    text = (REPO / "scripts/ashare_capability_probe.py").read_text()
    assert "NOT_INSTALLED" in text and "BLOCKED_BY_ENVIRONMENT" in text
    assert "SUPPORTED means a real request returned parsed rows" in text


def test_capability_probe_separates_throttling_from_unsupported():
    text = (REPO / "scripts/ashare_capability_probe.py").read_text()
    # a temporarily blocked public host must not read as a permanent gap
    assert "families_unavailable_only_due_to_throttling" in text
    assert "RATE_LIMITED" in text


def test_capability_probe_paces_the_rate_limited_vendor():
    """Without pacing the probe rate-limits itself and slanders its own vendor."""
    text = (REPO / "scripts/ashare_capability_probe.py").read_text()
    assert "TICKFLOW_PACE_S = 6.5" in text
    assert text.count("time.sleep(TICKFLOW_PACE_S)") >= 2

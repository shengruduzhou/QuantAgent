"""The workstation must have a governed path to the nonlinear question.

Round 21 / R3 (factor) finding: every scheme `search-factor-fusion` enumerates
returns a weight vector over rank-centred factors, so its score is
``Σ wᵢ · xᵢ`` — additive by construction, unable to express an interaction.
The one module that does form interaction terms
(`quantagent.models.interactions`) was reachable only from
`quantagent.research.model_comparison`, and `audit-nonlinear-factors` was not
on the job allowlist. An operator therefore had no way to ask whether letting
factors interact adds anything, while the fusion report's "best scheme" read
as if it had been chosen from a space that included nonlinearity.

The fusion summary already declares `modelClass: rank_weighted_additive` and
`representsInteraction: False`; these tests pin that declaration and the
allowlist entry together, since the declaration is only useful if the operator
has somewhere to go once they read it.
"""

from __future__ import annotations

import inspect

from services.quant_api.services.jobs import COMMANDS


def test_nonlinear_audit_is_allowlisted() -> None:
    assert "audit-nonlinear-factors" in COMMANDS


def test_allowlist_matches_the_cli_signature_exactly() -> None:
    """A drifted allowlist either blocks valid options or forwards invalid ones."""
    from quantagent.cli.nonlinear import audit_nonlinear_factors

    allowed = COMMANDS["audit-nonlinear-factors"]["allowed"]
    cli_params = set(inspect.signature(audit_nonlinear_factors).parameters)

    assert allowed - cli_params == set(), "allowlisted option the CLI does not accept"
    assert cli_params - allowed == set(), "CLI option the operator cannot reach"


def test_required_inputs_and_path_boundaries_are_declared() -> None:
    spec = COMMANDS["audit-nonlinear-factors"]

    assert spec["required"] == {"panel_path", "factor_names", "output_dir"}
    assert spec["path_inputs"] == {"panel_path"}
    assert spec["path_outputs"] == {"output_dir"}


def test_nonlinear_audit_exposes_no_trial_count_knob() -> None:
    """Same rule as fusion: a declarable trial count is a forgeable significance."""
    assert "n_trials" not in COMMANDS["audit-nonlinear-factors"]["allowed"]


def test_fusion_summary_still_declares_its_model_class() -> None:
    """The honesty floor: 'no nonlinearity was searched' must be visible."""
    import quantagent.fusion.search as search

    source = inspect.getsource(search)
    assert '"modelClass": "rank_weighted_additive"' in source
    assert '"representsInteraction": False' in source

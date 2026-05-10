import numpy as np
import pandas as pd

from quantagent.agents.agent_router import AgentRouter
from quantagent.agents.bl_views import posterior_alpha_from_agent_views
from quantagent.agents.commodity_agent import commodity_evidence_records
from quantagent.agents.flow_agent import flow_evidence_records
from quantagent.agents.policy_agent import PolicyEvent, policy_evidence_records
from quantagent.agents.views_schema import write_audit_jsonl
from quantagent.domain.schemas import AgentSignal


def test_v4_agent_evidence_maps_to_views_and_bl_posterior(tmp_path):
    sector_map = pd.Series({"600519.SH": "food", "300750.SZ": "ev"})
    policy = policy_evidence_records([PolicyEvent("2026-01-01", "support", ("food",), 0.6)], sector_map, reference_date=pd.Timestamp("2026-01-02"))
    flow = flow_evidence_records([AgentSignal("flow", "300750.SZ", 5, 0.8, 0.9, 0.8)], "2026-01-02T15:00:00")
    commodity = commodity_evidence_records(pd.Series({"crude_oil": 0.05}), pd.Series({"601857.SH": "oil_gas"}), "2026-01-02T15:00:00")
    evidence = policy + flow + commodity
    routed = AgentRouter().route(evidence, pd.Index(["600519.SH", "300750.SZ", "601857.SH"]))
    assert routed.views
    assert not any(hasattr(view, "order_type") for view in routed.views)
    prior = pd.Series(0.0, index=["600519.SH", "300750.SZ", "601857.SH"])
    cov = pd.DataFrame(np.eye(3) * 0.04, index=prior.index, columns=prior.index)
    posterior = posterior_alpha_from_agent_views(prior, cov, routed.views)
    assert posterior.shape[0] == 3
    audit = write_audit_jsonl(evidence, tmp_path / "audit.jsonl")
    assert audit.read_text(encoding="utf-8")

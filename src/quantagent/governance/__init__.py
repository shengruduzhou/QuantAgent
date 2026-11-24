"""Research governance gates.

``isolated_production_audit`` is the cross-review gate before the main repair
role: a finding author cannot approve its own work.
"""

from quantagent.governance import isolated_production_audit  # noqa: F401

__all__ = ["isolated_production_audit"]

"""Machine-auditable Qlib documentation and integration capability registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass


QLIB_MIN_VERSION = "0.9.7"
QLIB_SUPPORTED_SERIES = "0.9.x"
QLIB_DOCS_ROOT = "https://qlib.org.cn/en/latest/"
QLIB_STABLE_DOCS_ROOT = "https://qlib.readthedocs.io/en/stable/"
QLIB_PYPI_JSON = "https://pypi.org/pypi/pyqlib/json"


@dataclass(frozen=True)
class QlibCapability:
    capability_id: str
    title: str
    url: str
    domain: str
    integration_mode: str
    quantagent_target: str
    runtime_component: bool
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


# This registry intentionally mirrors every reference supplied in the official
# Qlib documentation hierarchy used by QuantAgent.  "governance_reference"
# means the page affects compatibility/development rules rather than providing a
# separately executable runtime component.
QLIB_CAPABILITIES: tuple[QlibCapability, ...] = (
    QlibCapability(
        "introduction",
        "Introduction",
        "https://qlib.org.cn/en/latest/introduction/introduction.html",
        "getting_started",
        "governance_reference",
        "docs/qlib_full_integration.md",
        False,
        "Architecture and design scope.",
    ),
    QlibCapability(
        "quickstart",
        "Quick Start",
        "https://qlib.org.cn/en/latest/introduction/quick.html",
        "getting_started",
        "governance_reference",
        "quantagent qlib-* CLI",
        False,
        "Canonical first-run workflow reference.",
    ),
    QlibCapability(
        "installation",
        "Installation",
        "https://qlib.org.cn/en/latest/start/installation.html",
        "first_use",
        "dependency_contract",
        "pyproject.toml[research]",
        False,
        "QuantAgent pins the tested 0.9.x series and lazy-loads pyqlib.",
    ),
    QlibCapability(
        "initialization",
        "Initialization",
        "https://qlib.org.cn/en/latest/start/initialization.html",
        "first_use",
        "native_bridge",
        "quantagent.qlib.runtime.QlibRuntime.initialize",
        True,
        "Provider URI and region remain explicit runtime configuration.",
    ),
    QlibCapability(
        "data_retrieval",
        "Data Retrieval",
        "https://qlib.org.cn/en/latest/start/getdata.html",
        "first_use",
        "native_bridge",
        "quantagent.qlib.runtime.QlibRuntime",
        True,
        "D.calendar, D.instruments and D.features are exposed through the bridge.",
    ),
    QlibCapability(
        "custom_model_integration",
        "Custom Model Integration",
        "https://qlib.org.cn/en/latest/start/integration.html",
        "first_use",
        "native_config_bridge",
        "quantagent.qlib.workflow + QlibRuntime.instantiate",
        True,
        "Any Qlib Model-compatible config can be used without hard-coding one learner.",
    ),
    QlibCapability(
        "workflow",
        "Workflow Management",
        "https://qlib.org.cn/en/latest/component/workflow.html",
        "component",
        "native_bridge",
        "quantagent.qlib.runtime.QlibRuntime.run_workflow_config",
        True,
        "qrun-style model/dataset/record execution with QuantAgent pre-validation.",
    ),
    QlibCapability(
        "data_layer",
        "Data Framework",
        "https://qlib.org.cn/en/latest/component/data.html",
        "component",
        "native_bridge",
        "quantagent.qlib.parquet + existing QlibProvider",
        True,
        "Qlib 0.9.7 StaticDataLoader reads QuantAgent Parquet directly.",
    ),
    QlibCapability(
        "model",
        "Forecast Model",
        "https://qlib.org.cn/en/latest/component/model.html",
        "component",
        "native_config_bridge",
        "QlibRuntime.instantiate / task_train",
        True,
        "Qlib built-ins and custom Model implementations are both supported.",
    ),
    QlibCapability(
        "strategy_backtest",
        "Portfolio Strategy and Backtest",
        "https://qlib.org.cn/en/latest/component/strategy.html",
        "component",
        "native_bridge",
        "quantagent.qlib.workflow.build_static_parquet_task",
        True,
        "TopkDropoutStrategy and PortAnaRecord are available as baselines, not promotion shortcuts.",
    ),
    QlibCapability(
        "highfreq_nested_decision",
        "Nested Decision Execution / High Frequency",
        "https://qlib.org.cn/en/latest/component/highfreq.html",
        "component",
        "native_config_bridge",
        "QlibRuntime.instantiate + QuantAgent execution layer",
        True,
        "Qlib nested execution remains optional and must respect QuantAgent execution/risk gates.",
    ),
    QlibCapability(
        "meta_controller",
        "Meta Controller",
        "https://qlib.org.cn/en/latest/component/meta.html",
        "component",
        "native_config_bridge",
        "QlibRuntime.instantiate + QuantAgent research layer",
        True,
        "MetaTask/MetaDataset/MetaModel can be instantiated from upstream configs.",
    ),
    QlibCapability(
        "recorder",
        "Qlib Recorder",
        "https://qlib.org.cn/en/latest/component/recorder.html",
        "component",
        "native_bridge",
        "QlibRuntime.recorder_api / task_train",
        True,
        "Experiment/Recorder artifacts supplement, not replace, QuantAgent manifests.",
    ),
    QlibCapability(
        "analysis_report",
        "Analysis and Report",
        "https://qlib.org.cn/en/latest/component/report.html",
        "component",
        "native_bridge",
        "Qlib records + QlibRuntime.evaluate_module",
        True,
        "Signal/backtest analysis is retained alongside QuantAgent OOS gates.",
    ),
    QlibCapability(
        "online",
        "Online Serving",
        "https://qlib.org.cn/en/latest/component/online.html",
        "component",
        "native_config_bridge",
        "QlibRuntime.build_online_manager",
        True,
        "OnlineManager/OnlineStrategy may drive simulation/update, never bypass live safety mode.",
    ),
    QlibCapability(
        "reinforcement_learning",
        "Reinforcement Learning",
        "https://qlib.org.cn/en/latest/component/rl/toctree.html",
        "component",
        "native_config_bridge",
        "QlibRuntime.instantiate + quantagent.rl",
        True,
        "Qlib RL components are usable as execution/research baselines under PIT controls.",
    ),
    QlibCapability(
        "formulaic_alpha",
        "Formulaic Alpha",
        "https://qlib.org.cn/en/latest/advanced/alpha.html",
        "advanced",
        "native_bridge",
        "QlibRuntime.features + QuantAgent factor evaluation",
        True,
        "Qlib expression operators and Alpha158/Alpha360 are benchmark/factor sources.",
    ),
    QlibCapability(
        "online_offline_server",
        "Online and Offline Mode",
        "https://qlib.org.cn/en/latest/advanced/server.html",
        "advanced",
        "native_config_bridge",
        "QlibRuntime.initialize",
        True,
        "Provider/server mode is explicit; no implicit network fallback.",
    ),
    QlibCapability(
        "serialization",
        "Serialization",
        "https://qlib.org.cn/en/latest/advanced/serial.html",
        "advanced",
        "native_config_bridge",
        "Qlib upstream Serializable + Recorder artifacts",
        True,
        "Serialized upstream objects keep version/provenance in QuantAgent manifests.",
    ),
    QlibCapability(
        "task_management",
        "Task Management",
        "https://qlib.org.cn/en/latest/advanced/task_management.html",
        "advanced",
        "native_bridge",
        "QlibRuntime.train_tasks",
        True,
        "TrainerR/DelayTrainerR/TrainerRM; TaskManager/Mongo is opt-in.",
    ),
    QlibCapability(
        "pit_database",
        "Point-in-Time Database",
        "https://qlib.org.cn/en/latest/advanced/PIT.html",
        "advanced",
        "governed_semantic_bridge",
        "QuantAgent available_at/PIT contracts + Qlib PIT semantics",
        True,
        "QuantAgent's disclosure-time and purge/embargo rules remain stricter source of truth.",
    ),
    QlibCapability(
        "code_standard",
        "Code Standard and Developer Guide",
        "https://qlib.org.cn/en/latest/developer/code_standard_and_dev_guide.html",
        "developer",
        "governance_reference",
        "QuantAgent tests/CI/docs",
        False,
        "Upstream extension and contribution conventions are compatibility references.",
    ),
    QlibCapability(
        "development_guidance",
        "Development Guidance",
        "https://qlib.org.cn/en/latest/developer/code_standard_and_dev_guide.html#development-guidance",
        "developer",
        "governance_reference",
        "QuantAgent tests/CI/docs",
        False,
        "Anchor retained separately because it is a supplied reference.",
    ),
    QlibCapability(
        "build_image",
        "Build Image",
        "https://qlib.org.cn/en/latest/developer/how_to_build_image.html",
        "developer",
        "governance_reference",
        "deployment documentation",
        False,
        "Container guidance only; no forced Docker runtime dependency.",
    ),
    QlibCapability(
        "api_reference",
        "API Reference",
        "https://qlib.org.cn/en/latest/reference/api.html",
        "reference",
        "governance_reference",
        "all Qlib bridges",
        False,
        "Primary API compatibility reference.",
    ),
    QlibCapability(
        "faq",
        "FAQ",
        "https://qlib.org.cn/en/latest/FAQ/FAQ.html",
        "reference",
        "governance_reference",
        "docs/qlib_full_integration.md",
        False,
        "Operational caveats and known behavior.",
    ),
    QlibCapability(
        "changelog",
        "Changelog",
        "https://qlib.org.cn/en/latest/changelog/changelog.html",
        "reference",
        "compatibility_watch",
        "quantagent qlib-audit",
        False,
        "Used with PyPI/release metadata to detect upstream drift.",
    ),
)

QLIB_DOC_COUNT = 27

_ids = [item.capability_id for item in QLIB_CAPABILITIES]
if len(_ids) != len(set(_ids)):
    raise RuntimeError("duplicate Qlib capability_id in registry")
if len(QLIB_CAPABILITIES) != QLIB_DOC_COUNT:
    raise RuntimeError(
        f"Qlib capability registry count drift: {len(QLIB_CAPABILITIES)} != {QLIB_DOC_COUNT}"
    )

RUNTIME_CAPABILITIES = tuple(
    item for item in QLIB_CAPABILITIES if item.runtime_component
)
GOVERNANCE_CAPABILITIES = tuple(
    item for item in QLIB_CAPABILITIES if not item.runtime_component
)


def coverage_registry() -> list[dict[str, object]]:
    return [item.to_dict() for item in QLIB_CAPABILITIES]

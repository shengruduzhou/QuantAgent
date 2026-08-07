"""Governed integration layer for Microsoft Qlib.

The package is intentionally optional: importing QuantAgent never imports
``pyqlib``.  Runtime Qlib imports happen only when a Qlib command or bridge is
explicitly used.
"""

from quantagent.qlib.catalog import (
    QLIB_CAPABILITIES,
    QLIB_DOC_COUNT,
    QLIB_MIN_VERSION,
    QLIB_SUPPORTED_SERIES,
    QlibCapability,
)
from quantagent.qlib.parquet import (
    QlibParquetManifest,
    build_qlib_static_frame,
    write_qlib_static_parquet,
)
from quantagent.qlib.runtime import QlibRuntime, QlibUnavailable
from quantagent.qlib.workflow import QlibSegments, build_static_parquet_task

__all__ = [
    "QLIB_CAPABILITIES",
    "QLIB_DOC_COUNT",
    "QLIB_MIN_VERSION",
    "QLIB_SUPPORTED_SERIES",
    "QlibCapability",
    "QlibParquetManifest",
    "QlibRuntime",
    "QlibSegments",
    "QlibUnavailable",
    "build_qlib_static_frame",
    "build_static_parquet_task",
    "write_qlib_static_parquet",
]

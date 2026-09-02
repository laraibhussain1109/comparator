"""Industrial casting inspection prototype and process configuration."""

from __future__ import annotations

import os


def configure_process_runtime() -> None:
    """Configure joblib before scikit-learn initializes its worker backend.

    Joblib otherwise tries to execute ``wmic`` to count physical cores on
    Windows.  That utility is absent on current Windows installations.  The
    documented environment override makes joblib use the logical CPU count
    directly and must be present before modules such as KMeans are imported.
    An operator-provided value is preserved.
    """
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, os.cpu_count() or 1)))


configure_process_runtime()

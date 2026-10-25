"""Environment-variable lookup with a deprecated legacy-name fallback.

The package used to be called ``energy-baseline`` and read ``ENERGY_BASELINE_*``
variables.  Operators may still have those exported, so :func:`get_env` reads the
``BATTERY_OPTIMIZER_*`` name first and only then falls back to the legacy name,
emitting a :class:`DeprecationWarning` when the fallback is what supplied the value.
When both are set the new name wins silently.
"""

from __future__ import annotations

import os
import warnings

_NEW_PREFIX = "BATTERY_OPTIMIZER_"
_LEGACY_PREFIX = "ENERGY_BASELINE_"


def legacy_name(name: str) -> str:
    """Return the legacy ``ENERGY_BASELINE_*`` spelling of a ``BATTERY_OPTIMIZER_*`` name."""
    if not name.startswith(_NEW_PREFIX):
        raise ValueError(f"expected a {_NEW_PREFIX}* variable name, got {name!r}")
    return _LEGACY_PREFIX + name[len(_NEW_PREFIX) :]


def get_env(name: str) -> str | None:
    """Return the value of *name*, falling back to its deprecated legacy spelling.

    An empty value counts as unset (matching how every caller treated the variable
    before).  Returns ``None`` when neither variable is set.
    """
    value = os.environ.get(name)
    if value:
        return value
    legacy = legacy_name(name)
    value = os.environ.get(legacy)
    if value:
        warnings.warn(
            f"environment variable {legacy} is deprecated; use {name} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return value
    return None

"""Scalar graph-level targets over molpy ``Frame.meta``.

molpy 0.20 exposes ``Frame.meta`` as a live mapping of plain Python values.
:class:`Targets` is the molhub view: read what is there, and on write replace
the whole mapping with scalars only (``bool`` / ``int`` / ``float`` / ``str``).
Numpy scalars are reduced first; non-scalars are rejected — molpy would
otherwise store them as JSON.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping

import numpy as np
from molpy import Frame

__all__ = ["Targets"]

# ``bool`` must precede ``int``: ``isinstance(True, int)`` is True.
_SCALAR_TYPES: tuple[type, ...] = (bool, int, float, str)


class Targets:
    """The graph-level targets of one :class:`~molpy.Frame`.

    A thin, cheap view — construct it wherever you need to touch a frame's
    metadata::

        Targets(frame).write({"energy": -40.5})
        Targets(frame)["energy"]        # -> -40.5
        Targets(frame).read()           # -> {"energy": -40.5}

    Writes **replace** rather than merge. Direct mutations of the live
    ``frame.meta`` mapping are also visible through this view.
    """

    def __init__(self, frame: Frame) -> None:
        self._frame = frame

    def read(self) -> dict[str, Any]:
        """Return every target as a plain Python value."""
        return dict(self._frame.meta)

    def write(self, values: Mapping[str, Any]) -> None:
        """Replace all targets with *values*.

        Args:
            values: Mapping of key to a ``bool``, ``int``, ``float``, ``str``,
                or the numpy scalar equivalent of one of those.

        Raises:
            TypeError: If a value is not a scalar target type.
        """
        plain: dict[str, Any] = {}
        for key, raw in values.items():
            value = raw.item() if isinstance(raw, np.generic) else raw
            if not isinstance(value, _SCALAR_TYPES):
                raise TypeError(
                    f"Metadata key {key!r} has unsupported type "
                    f"{type(raw).__name__}; molhub targets accept "
                    "bool, int, float, or str."
                )
            plain[key] = value
        self._frame.meta = plain

    def __getitem__(self, key: str) -> Any:
        """Return one target value.

        Raises:
            KeyError: If *key* is not a target of this frame.
        """
        return self._frame.meta[key]

    def __contains__(self, key: str) -> bool:
        return key in self._frame.meta

    def __iter__(self) -> Iterator[str]:
        return iter(self._frame.meta)

    def __len__(self) -> int:
        return len(self._frame.meta)

    def keys(self) -> list[str]:
        """Return the target names present on this frame."""
        return list(self._frame.meta)

"""Thin compatibility shim for the vendored N1.7 setup.py.

The upstream ``gr00t_n1d7.setup`` does ``from
gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor``,
but we renamed the vendored module to ``processing_gr00t_n1d7_upstream``
so that the fork-local slim processor could live at ``processing.py``.

This file preserves the upstream module name by re-exporting the symbols
that the upstream ``setup.py`` and ``gr00t_n1d7.py`` expect.
"""

from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7_upstream import (
    Gr00tN1d7DataCollator,
    Gr00tN1d7Processor,
)

__all__ = ["Gr00tN1d7DataCollator", "Gr00tN1d7Processor"]

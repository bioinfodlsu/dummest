#!/usr/bin/env python3
"""Transmark paper display labels and plot colors (thin wrapper).

All label/color logic lives in the ``bath-shared`` submodule so transmark
and ncbi-data share one scheme with no duplicated dicts. This module only
wires the submodule onto ``sys.path`` and keeps the transmark-specific
filepath helper name.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bath-shared"),
)

from style import (  # noqa: E402
    ALIASES,
    TOOL_COLORS,
    canonical_tool,
    color_for_tool,
    label_for_path,
)


def pretty_label(filepath: str):
    """Map an input file path to its paper display name, or None."""
    return label_for_path(filepath)


def color_for_file(filepath: str):
    """Map an input file path to its plot color, or None."""
    return color_for_tool(pretty_label(filepath))

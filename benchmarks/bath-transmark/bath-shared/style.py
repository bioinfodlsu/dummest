#!/usr/bin/env python3
"""Shared paper style: canonical tool labels, plot colors and label mapping.

Single source of truth for BATH-paper/transmark and ncbi-data so both
repos use the same color scheme and display names with no duplicated
dicts. Two label entry points cover the two callers:

    label_for_path(filepath)  -> transmark: filename basename -> label
    label_for_run(run_key)    -> ncbi: runner/tool key -> label

Canonical labels (true argument name uses hyphen: ``--no-fs``):
    BATH                      -> tab:red
    BATH --no-fs              -> tab:orange
    DUMMEST                   -> tab:green
    DUMMEST (more sensitive)  -> tab:blue
    HMMER3                    -> tab:purple
    LAST                      -> tab:brown

Aliases resolve to canonical labels before color lookup:
    "BATH --nofs" -> "BATH --no-fs"   (deprecated spelling)
"""
from __future__ import annotations

import os

TOOL_COLORS = {
    "DUMMEST (more sensitive)": "tab:blue",
    "DUMMEST": "tab:green",
    "BATH": "tab:red",
    "BATH (non-frameshift-aware)": "tab:orange",
    "HMMER3": "tab:purple",
    "LAST": "tab:brown",
}

# Non-canonical labels -> canonical label. Color lookup always goes
# through canonicalization, so aliases share the canonical color.
ALIASES = {
    "BATH --nofs": "BATH --no-fs",
}

# Filename basename substring rules -> canonical label. Order matters:
# the first rule whose substrings are ALL present in the lowercased
# basename wins, so the more specific "p002" rule must precede "dummer".
# NOTE: "blast" contains "last" as a substring; there is currently no
# blast rule and no blast inputs to step-last, so a bare "last" rule is
# safe. If blast support is added later, add a blast rule BEFORE this
# one (or anchor this rule to e.g. "last-" / "last_").
PATH_RULES = (
    (("dummer", "p002"), "DUMMEST (more sensitive)"),
    (("dummer",), "DUMMEST"),
    (("bath", "no-fs"), "BATH (non-frameshift-aware)"),
    (("bath",), "BATH"),
    (("hmmer",), "HMMER3"),
    (("last",), "LAST"),
)

# ncbi runner/tool keys -> canonical label (keys are lowercased).
RUN_LABELS = {
    "hmmsearch": "HMMER3",
    "hmmer": "HMMER3",
    "hmmer3": "HMMER3",
    "bath": "BATH",
    "bath --no-fs": "BATH --no-fs",
    "bath --nofs": "BATH --no-fs",
    "bath_nofs": "BATH --no-fs",
    "bath nofs": "BATH --no-fs",
    "dummer": "DUMMEST",
    "dummest": "DUMMEST",
    "last": "LAST",
    "lastal": "LAST",
    "last-fs": "LAST",
}


def canonical_tool(label: str | None):
    """Map a label (or alias) to its canonical paper label, or None."""
    if label is None:
        return None
    stripped = label.strip()
    if stripped in TOOL_COLORS:
        return stripped
    if stripped in ALIASES:
        return ALIASES[stripped]
    lowered = stripped.lower()
    for key in TOOL_COLORS:
        if key.lower() == lowered:
            return key
    for alias, canon in ALIASES.items():
        if alias.lower() == lowered:
            return canon
    return stripped


def color_for_tool(label: str | None):
    """Map a display name, alias or run key to its plot color, or None."""
    if label is None:
        return None
    canon = canonical_tool(label)
    if canon in TOOL_COLORS:
        return TOOL_COLORS[canon]
    return TOOL_COLORS.get(label_for_run(label))


def label_for_path(filepath: str):
    """Map an input file path to its paper display name, or None.

    Matches against ``os.path.basename`` only.
    """
    low = os.path.basename(filepath).lower()
    for substrings, label in PATH_RULES:
        if all(s in low for s in substrings):
            return label
    return None


def label_for_run(run_key: str | None):
    """Map an ncbi runner/tool key to its canonical paper label."""
    if run_key is None:
        return None
    return RUN_LABELS.get(run_key.strip().lower(), run_key)

#!/usr/bin/env python3
"""Estimate 1- and 2-base indel rates from a Badread error model.

This reproduces the rough approximation used for DUMMER's provisional
background frameshift constants.  The reported rates are occurrence rates
per reference base after scaling the empirical error proportions to the
requested identity.  They are not calibrated codon-level HMM transition
probabilities; see docs/frameshift-rate-approximation.md.
"""

import argparse
import gzip
import os
import sys
from collections import defaultdict
from pathlib import Path


DEFAULT_BADREAD_ROOT = os.environ.get("BADREAD_ROOT")


def load_align_kmers(badread_root):
    """Load Badread's alignment helper from a source checkout or install."""
    if badread_root:
        sys.path.insert(0, str(badread_root))
    try:
        import badread
        from badread.error_model import align_kmers
    except ImportError as exc:
        raise SystemExit(
            "Could not import Badread. Pass --badread-root pointing to its "
            "source checkout, or install Badread."
        ) from exc
    package_root = Path(badread.__file__).resolve().parent
    return align_kmers, package_root


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Badread error-model insertion/deletion rates."
    )
    parser.add_argument(
        "--identity",
        "-i",
        type=float,
        default=None,
        help="Target identity percentage, such as 90 or 95.",
    )
    parser.add_argument(
        "--badread-root",
        default=DEFAULT_BADREAD_ROOT,
        help="Badread source checkout or install root.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=Path,
        default=None,
        help="Gzipped Badread error model (defaults to nanopore2020).",
    )
    args = parser.parse_args()

    if args.identity is not None and not 0 <= args.identity <= 100:
        parser.error("--identity must be between 0 and 100")

    badread_root = Path(args.badread_root) if args.badread_root else None
    align_kmers, badread_package = load_align_kmers(badread_root)
    model_path = args.model or (badread_package / "error_models" / "nanopore2020.gz")
    if not model_path.exists():
        parser.error(f"model not found: {model_path}")
    counts = defaultdict(float)
    total_edit_distance = 0.0

    print(f"Loading error model from {model_path}...", file=sys.stderr)

    with gzip.open(model_path, "rt") as model_file:
        for line in model_file:
            parts = [item.split(",") for item in line.strip().split(";") if item]
            if not parts:
                continue
            ref_kmer = parts[0][0]

            # Subsequent parts are (alternative k-mer, probability).
            for alt_kmer, prob_str in parts[1:]:
                if alt_kmer == ref_kmer:
                    continue

                prob = float(prob_str)
                try:
                    aligned = align_kmers(ref_kmer, alt_kmer)
                except AssertionError:
                    continue

                # Count substitutions and insertions.  This intentionally
                # preserves the historical approximation used for DUMMER.
                for j, base_str in enumerate(aligned):
                    ref_base = ref_kmer[j]
                    if base_str == ref_base:
                        continue

                    if base_str == "":
                        pass
                    elif len(base_str) == 1:
                        counts["substitution"] += prob
                        total_edit_distance += prob
                    else:
                        ins_len = len(base_str) - 1
                        counts[f"{ins_len}-insertion"] += prob
                        total_edit_distance += ins_len * prob

                # Count contiguous deletion runs.
                current_del_run = 0
                for base_str in aligned:
                    if base_str == "":
                        current_del_run += 1
                    else:
                        if current_del_run > 0:
                            counts[f"{current_del_run}-deletion"] += prob
                            total_edit_distance += current_del_run * prob
                            current_del_run = 0
                if current_del_run > 0:
                    counts[f"{current_del_run}-deletion"] += prob
                    total_edit_distance += current_del_run * prob

    selected_events = [
        "substitution",
        "1-insertion",
        "2-insertion",
        "1-deletion",
        "2-deletion",
    ]

    print("\nError Profile & Proportions:")
    print("-" * 90)
    print(
        f"{'Event Type':<15} | {'Count Weight':<15} | "
        f"{'Edit Dist Contribution':<24} | {'% of All Errors':<16}"
    )
    print("-" * 90)

    for event in selected_events:
        weight = counts[event]
        length = 2 if "2-" in event else 1
        contribution = weight * length
        proportion = (
            contribution / total_edit_distance if total_edit_distance > 0 else 0
        )
        print(
            f"{event:<15} | {weight:<15.4f} | {contribution:<24.4f} | "
            f"{proportion:<16.2%}"
        )
    print("-" * 90)

    if args.identity is None:
        return

    error_rate = (100.0 - args.identity) / 100.0
    print(
        f"\nScaled Occurrence Rates at {args.identity}% Target Identity "
        f"(overall error rate: {error_rate:.1%}):"
    )
    print("-" * 80)
    print(
        f"{'Event Type':<15} | {'Event Length':<12} | "
        f"{'Occurrence Rate (Per Base)':<28} | {'Events per 100 bp':<20}"
    )
    print("-" * 80)

    for event in selected_events:
        length = 1 if event == "substitution" else (2 if "2-" in event else 1)
        contribution_proportion = (
            counts[event] * length / total_edit_distance
            if total_edit_distance > 0
            else 0
        )
        event_rate_per_base = error_rate * (contribution_proportion / length)
        print(
            f"{event:<15} | {length:<12} | {event_rate_per_base:<28.6%} | "
            f"{event_rate_per_base * 100:<20.4f}"
        )
    print("-" * 80)


if __name__ == "__main__":
    main()

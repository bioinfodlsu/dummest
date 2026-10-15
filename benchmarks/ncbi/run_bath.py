#!/usr/bin/env python3
"""BATH (bathsearch) against the NCBI AMRFinderPlus BATH HMM database.

Usage:
    python run_bath.py genome.fa [--bath-no-fs] [--max] [--bath-bin DIR]

Output (outdir):
    bath_output.txt, bath_tbl.txt, summary.tsv
"""

import argparse
import os
import subprocess
import sys
import time

import common


def main() -> None:
    parser = argparse.ArgumentParser(
        description="bathsearch NCBI AMRFinderPlus AMR HMMs against "
                    "nucleotide reads",
    )
    common.add_scan_args(parser)
    parser.set_defaults(outdir="bath_results")
    parser.add_argument(
        "--hmmdb", default=common.BHMM_DEFAULT,
        help=f"BATH HMM database (default: {common.BHMM_DEFAULT})",
    )
    parser.add_argument("--bath-bin", default=common.BATH_BIN_DEFAULT,
                        help="Path to BATH binaries directory "
                             "(default: $BATH_BIN, else PATH)")
    parser.add_argument("--bath-ct", type=int, default=1,
                        help="NCBI codon translation table ID (default: 1)")
    parser.add_argument("--bath-no-fs", action="store_true",
                        help="Disable frameshift-aware alignment")
    parser.add_argument("--bath-splice", action="store_true",
                        help="Enable splice-aware alignment")
    parser.add_argument("--bath-cigar", action="store_true",
                        help="Include CIGAR string in --tblout output")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Input not found: {args.input}")
    if not os.path.exists(args.hmmdb):
        sys.exit(f"BATH HMM database not found: {args.hmmdb}")
    os.makedirs(args.outdir, exist_ok=True)

    seqtype = common.detect_seqtype(args.input)
    if seqtype == "protein":
        sys.exit("--bath requires nucleotide input "
                 "(genomic DNA/reads), not protein.")

    t_total_start = time.time()
    raw_count = common.count_sequences(args.input)
    subsample = args.max_reads
    input_for_bath, n_query = common.ensure_subsample(
        args.input, subsample, common.CACHE_DIR_DEFAULT, True, args.outdir,
    )

    print(f"Running bathsearch with {args.cpu} threads ...", file=sys.stderr)
    print(f"  HMM DB:  {args.hmmdb}", file=sys.stderr)
    print(f"  Input:   {args.input} ({raw_count:,} raw → "
          f"{n_query:,} prepped)", file=sys.stderr)
    print(f"  Output:  {args.outdir}/", file=sys.stderr)

    bathsearch = os.path.join(args.bath_bin, "bathsearch")
    bathstat = os.path.join(args.bath_bin, "bathstat")
    bath_tbl = os.path.join(args.outdir, "bath_tbl.txt")
    bath_out = os.path.join(args.outdir, "bath_output.txt")
    summary_path = os.path.join(args.outdir, "summary.tsv")

    cmd = [bathsearch]
    if not args.bath_no_fs:
        cmd.append("--fs")
    if args.bath_splice:
        cmd.append("--splice")
    if args.bath_cigar:
        cmd.append("--cigar")
    if args.max:
        cmd.append("--max")
    cmd += ["--cpu", str(args.cpu)]
    if args.bath_ct != 1:
        cmd += ["--ct", str(args.bath_ct)]
    cmd += ["-E", str(args.evalue)]
    cmd += ["--tblout", bath_tbl, "-o", bath_out, args.hmmdb, input_for_bath]

    t0 = time.time()
    result = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print("bathsearch failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    print("  bathsearch finished.", file=sys.stderr)

    r = subprocess.run([bathstat, args.hmmdb],
                       capture_output=True, text=True)
    hmm_n = sum(1 for line in r.stdout.splitlines()
                if line.strip() and line.strip()[0].isdigit())

    common.finish_search(
        "BATH", common.parse_bath_tbl, bath_tbl, bath_out, bath_tbl,
        summary_path, args.input, raw_count, n_query, hmm_n, elapsed,
        time.time() - t_total_start,
    )


if __name__ == "__main__":
    main()

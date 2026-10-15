#!/usr/bin/env python3
"""hmmsearch against the NCBI AMRFinderPlus AMR profile HMM database.

Input type is auto-detected: nucleotide (FASTQ/FASTA) is subsampled and
6-frame translated; protein (faa/pep) is used as-is.

Usage:
    python run_hmmsearch.py reads.fastq [--max-reads 5000] [--max]

Output (outdir):
    hmmsearch_tbl.txt, hmmsearch_dom.txt, summary.tsv
"""

import argparse
import os
import subprocess
import sys
import time

import common
from bath_shared.style import label_for_run


def main() -> None:
    parser = argparse.ArgumentParser(
        description="hmmsearch NCBI AMRFinderPlus AMR profile HMMs against "
                    "metagenomic reads",
    )
    common.add_scan_args(parser)
    parser.set_defaults(outdir="hmmer_results")
    parser.add_argument(
        "--hmmdb", default=common.HMMDB_DEFAULT,
        help=f"Profile HMM database (default: {common.HMMDB_DEFAULT})",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Input not found: {args.input}")
    if not os.path.exists(args.hmmdb):
        sys.exit(f"HMM database not found: {args.hmmdb}")
    os.makedirs(args.outdir, exist_ok=True)

    # t_total_start covers seqtk prep + seqkit translation so the reported
    # Total is comparable with BATH/DUMMER (which translate internally).
    t_total_start = time.time()
    subsample = args.max_reads
    protein_fasta, n_query = common.prepare_translated_queries(
        args.input, subsample, args.cpu,
        common.CACHE_DIR_DEFAULT, True, args.outdir,
    )
    raw_count = common.count_sequences(args.input)

    tblout = os.path.join(args.outdir, "hmmsearch_tbl.txt")
    domtblout = os.path.join(args.outdir, "hmmsearch_dom.txt")

    cmd = [
        "hmmsearch",
        "--cpu", str(args.cpu),
        "--tblout", tblout,
        "--domtblout", domtblout,
        "-E", str(args.evalue),
    ]
    if args.max:
        cmd.append("--max")
    cmd += [args.hmmdb, protein_fasta]

    print(f"Running hmmsearch with {args.cpu} threads ...", file=sys.stderr)
    print(f"  HMM DB:  {args.hmmdb}", file=sys.stderr)
    print(f"  Query:   {protein_fasta} ({n_query:,} seqs)", file=sys.stderr)
    print(f"  Output:  {args.outdir}/", file=sys.stderr)

    t0 = time.time()
    result = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print("hmmsearch failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    print("  hmmsearch finished.", file=sys.stderr)

    with open(args.hmmdb) as f:
        hmm_n = sum(1 for line in f if line.startswith("HMMER"))
    summary_path = os.path.join(args.outdir, "summary.tsv")
    common.finish_search(
        label_for_run("hmmsearch"), common.parse_hmmer_tbl, tblout,
        tblout, domtblout, summary_path, args.input, raw_count, n_query,
        hmm_n, elapsed, time.time() - t_total_start,
    )


if __name__ == "__main__":
    main()

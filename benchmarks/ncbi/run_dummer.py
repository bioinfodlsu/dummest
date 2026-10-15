#!/usr/bin/env python3
"""Run the DUMMER pipeline (bin/dummest) against the NCBI AMRFinderPlus
MSA/HMM database.

Set DUMMER_REPO (or --dummer-pipeline) to the seq-position-probs tree.

Usage:
    python run_dummer.py genome.fa [--max]

Output (outdir):
    dummer_output.txt
"""

import argparse
import os
import subprocess
import sys
import time

import common

_DUMMER_REPO = os.environ.get("DUMMER_REPO")
DUMMER_PIPELINE_DEFAULT = os.environ.get(
    "DUMMER_PIPELINE",
    os.path.join(_DUMMER_REPO, "bin/dummest") if _DUMMER_REPO
    else "bin/dummest",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the DUMMER pipeline on nucleotide reads",
    )
    common.add_scan_args(parser)
    parser.set_defaults(outdir="dummer_results")
    parser.add_argument(
        "--hmmdb", default=common.HMMDB_DEFAULT,
        help=f"Profile HMM database (default: {common.HMMDB_DEFAULT})",
    )
    parser.add_argument(
        "--msa", default=common.MSA_DEFAULT,
        help=f"Combined MSA in Stockholm format (default: {common.MSA_DEFAULT})",
    )
    parser.add_argument("--dummer-pipeline", default=DUMMER_PIPELINE_DEFAULT,
                        help="Path to the dummest pipeline script (default: $DUMMER_PIPELINE "
                              "or $DUMMER_REPO/bin/dummest; legacy bin/pipeline.py symlink also works)")
    parser.add_argument("--dummest-bin", dest="dummer_bin",
                        help="Path to dummest binary (auto-detected by default)")
    parser.add_argument("--prefilter-mode", type=int,
                        help="MMseqs2 --prefilter-mode")
    parser.add_argument("--output-fa",
                        help="Save debug FASTA to this path")
    parser.add_argument("--skip-dummest", dest="skip_dummer", action="store_true",
                        help="Only generate debug FASTA, skip dummest")
    parser.add_argument("--target-db-mode", choices=["orf", "frame"],
                        default="orf",
                        help="MMseqs target construction in bin/dummest (default: orf; "
                             "ORF-split via extractorfs + offsetalignment). 'frame' keeps "
                             "the legacy full-frame 6-frame translation.")
    parser.add_argument("--target-db-pad", default=None,
                        help="Existing MMseqs targetDB_pad prefix, legacy frame mode "
                             "only (default: auto-cache in --cache-dir). Rejected with "
                             "--target-db-mode orf: full-frame caches are not reused.")
    parser.add_argument("--target-orf-db", default=None,
                        help="Existing ORF-split target DB prefix (ORF mode only; "
                             "requires --target-contig-db)")
    parser.add_argument("--target-contig-db", default=None,
                        help="Existing parent-contig nucleotide DB prefix (ORF mode "
                             "only; pairs with --target-orf-db)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Input not found: {args.input}")
    if not os.path.exists(args.hmmdb):
        sys.exit(f"HMM database not found: {args.hmmdb}")
    if not os.path.exists(args.msa):
        sys.exit(f"DUMMER MSA not found: {args.msa}")
    if not os.path.exists(args.dummer_pipeline):
        sys.exit(f"dummest pipeline script not found: {args.dummer_pipeline} "
                  f"(set DUMMER_REPO or --dummer-pipeline)")
    os.makedirs(args.outdir, exist_ok=True)

    seqtype = common.detect_seqtype(args.input)
    if seqtype == "protein":
        sys.exit("--dummer requires nucleotide input "
                 "(genomic DNA/reads), not protein.")

    use_cache = True
    t_total_start = time.time()
    raw_count = common.count_sequences(args.input)
    subsample = args.max_reads
    input_for_pipeline, n_query = common.ensure_subsample(
        args.input, subsample, common.CACHE_DIR_DEFAULT, use_cache,
        args.outdir,
    )

    print(f"Running DUMMER pipeline with {args.cpu} threads ...",
          file=sys.stderr)
    print(f"  HMM DB:  {args.hmmdb}", file=sys.stderr)
    print(f"  MSA:     {args.msa}", file=sys.stderr)
    print(f"  Input:   {args.input} ({raw_count:,} raw → "
          f"{n_query:,} prepped)", file=sys.stderr)
    print(f"  Output:  {args.outdir}/", file=sys.stderr)

    cmd = [
        sys.executable, args.dummer_pipeline,
        args.hmmdb, args.msa, input_for_pipeline, str(args.cpu),
    ]
    cmd += ["--evalue", str(args.evalue)]
    if args.dummer_bin:
        cmd += ["--dummest-bin", args.dummer_bin]
    if args.max:
        cmd += ["--max"]
    # MMseqs2 prefilter inputs are skipped entirely in --max mode.
    if not args.max:
        querydb = common.cached_querydb_prefix(args.msa)
        common.ensure_query_db(args.msa, querydb, args.cpu)
        cmd += ["--query-db", querydb]
        cmd += ["--target-db-mode", args.target_db_mode]
        if args.target_db_mode == "orf":
            if args.target_db_pad:
                sys.exit("--target-db-pad is a legacy full-frame cache and is not "
                         "reused in ORF-split mode (the default). Drop the flag to "
                         "auto-build the versioned ORF cache, pass "
                         "--target-orf-db/--target-contig-db for an existing ORF "
                         "pair, or select --target-db-mode frame.")
            if args.target_orf_db or args.target_contig_db:
                if not (args.target_orf_db and args.target_contig_db):
                    sys.exit("--target-orf-db and --target-contig-db must be "
                             "given together.")
                cmd += ["--target-orf-db", args.target_orf_db,
                        "--target-contig-db", args.target_contig_db]
            elif use_cache:
                pair_prefix = common.cached_orf_prefix(
                    args.input, subsample, common.CACHE_DIR_DEFAULT)
                try:
                    orf_prefix, contig_prefix = common.ensure_cached_target_orfs(
                        input_for_pipeline, pair_prefix, args.cpu)
                    cmd += ["--target-orf-db", orf_prefix,
                            "--target-contig-db", contig_prefix]
                except Exception as exc:
                    print(f"  Warning: ORF-split cache build failed ({exc}); "
                          f"letting pipeline.py build it inline.",
                          file=sys.stderr)
        else:
            if args.target_orf_db or args.target_contig_db:
                sys.exit("--target-orf-db/--target-contig-db require "
                         "--target-db-mode orf.")
            if use_cache and not args.target_db_pad:
                pad_prefix = common.cached_pad_prefix(
                    args.input, subsample, common.CACHE_DIR_DEFAULT)
                try:
                    common.ensure_cached_target_pad(
                        input_for_pipeline, pad_prefix, args.cpu)
                    cmd += ["--target-db-pad", pad_prefix]
                except Exception as exc:
                    print(f"  Warning: targetDB_pad cache build failed ({exc}); "
                          f"letting pipeline.py build it inline.",
                          file=sys.stderr)
            elif args.target_db_pad:
                cmd += ["--target-db-pad", args.target_db_pad]
    if args.prefilter_mode is not None:
        cmd += ["--prefilter-mode", str(args.prefilter_mode)]
    if args.output_fa:
        cmd += ["--output-fa", args.output_fa]
    if args.skip_dummer:
        cmd += ["--skip-dummest"]

    dummer_out = os.path.join(args.outdir, "dummer_output.txt")
    t0 = time.time()
    with open(dummer_out, "w") as outf:
        result = subprocess.run(
            cmd, stdout=outf, stderr=subprocess.PIPE, text=True,
        )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print("DUMMER pipeline failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)

    print("  DUMMER pipeline finished.", file=sys.stderr)
    print(f"  Output:   {dummer_out}", file=sys.stderr)
    print(f"  Time:     {elapsed:.1f}s (search only)", file=sys.stderr)
    print(f"  Total time:    {time.time() - t_total_start:.1f}s"
          f" (incl. prep)", file=sys.stderr)


if __name__ == "__main__":
    main()

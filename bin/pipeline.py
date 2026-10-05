#!/usr/bin/env python3
import sys
import os
import shutil
import subprocess
import tempfile
import argparse
import pybedtools

_DISK_TMP_THRESHOLD = 2 * 1024**3  # 2 GiB


def _tmp_base_dir(fa_file):
    if os.environ.get("DUMMEST_TMP_BYPASS", "1") in ("0", "false", "False", "no"):
        return None
    try:
        if os.path.getsize(fa_file) > _DISK_TMP_THRESHOLD:
            return "."
    except OSError:
        pass
    return None


def _cache_dir():
    # Shared with dummest: $DUMMEST_CACHE_DIR, then
    # $XDG_CACHE_HOME/.dummest-cache, $HOME/.dummest-cache, ./.dummest-cache,
    # $TMPDIR/.dummest-cache, /tmp/.dummest-cache, then the current directory.
    candidates = []
    override = os.environ.get("DUMMEST_CACHE_DIR")
    if override:
        candidates.append(override)
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        candidates.append(os.path.join(xdg, ".dummest-cache"))
    home = os.environ.get("HOME")
    if home:
        candidates.append(os.path.join(home, ".dummest-cache"))
    candidates.append(os.path.join(os.getcwd(), ".dummest-cache"))
    tmp = os.environ.get("TMPDIR") or tempfile.gettempdir()
    if tmp:
        candidates.append(os.path.join(tmp, ".dummest-cache"))
    candidates.append("/tmp/.dummest-cache")

    for d in candidates:
        try:
            os.makedirs(d, exist_ok=True)
            if os.access(d, os.W_OK):
                return d
        except OSError:
            continue
    return os.getcwd()


def _gpu_available():
    # True iff an NVIDIA GPU is actually usable: nvidia-smi must both exist
    # and succeed (in a container without --nv it may exist yet fail).
    try:
        r = subprocess.run(["nvidia-smi", "-L"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return False
    return r.returncode == 0


def _resolve_gpu(mode):
    if mode == "on":
        return True
    if mode == "off":
        return False
    if _gpu_available():
        print("# GPU detected (nvidia-smi); using GPU MMseqs2 prefilter")
        return True
    print("# No NVIDIA GPU detected; using CPU-only MMseqs2 prefilter")
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline: HMM-guided genomic search via MMseqs2 + dummest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python3 pipeline.py profile.hmm msa.msa genome.fa 8\n"
               "  python3 pipeline.py profile.hmm msa.msa genome.fa 8 --target-db-pad /path/to/targetDB_pad --query-db /path/to/queryDB\n"
    )
    parser.add_argument("hmm_file", help="HMM profile file")
    parser.add_argument("msa_file", help="MSA file (.msa for Stockholm, otherwise used as-is for query DB). Ignored when --query-db is provided.")
    parser.add_argument("fa_file", help="Genome FASTA file")
    parser.add_argument("cpus", help="Number of CPUs")
    parser.add_argument("--target-db-pad", dest="target_db_pad", default=None,
                        help="Path to an existing padded target DB (skips createdb)")
    parser.add_argument("--query-db", dest="query_db", default=None,
                        help="Path to an existing query profile DB (skips convertmsa + msa2profile)")
    parser.add_argument("--skip-dummest", dest="skip_dummer", action="store_true",
                        help="Skip running dummest (only generate debug.fa)")
    parser.add_argument("--output-fa", dest="output_fa", default=None,
                        help="Save debug FASTA to this path (persistent copy)")
    parser.add_argument("--max", action="store_true",
                        help="Max sensitivity: disable heuristic windowing (pad=full contig)")
    parser.add_argument("--dummest-bin", dest="dummer_bin", metavar="DUMMEST_BIN", default=None,
                        help="Path to dummest binary (overrides default dummest)")
    parser.add_argument("--prefilter-mode", type=int, default=1, choices=[1],
                        help="MMseqs2 prefilter mode (currently only 1 = ungapped "
                             "prefilter, applies --ungapped-pvalue).")
    parser.add_argument("--gpu", choices=["auto", "on", "off"], default="auto",
                        help="MMseqs2 GPU use: auto (default; GPU when nvidia-smi "
                             "reports one, else CPU), on (fail if no GPU), off (CPU only).")
    parser.add_argument("--gapped-evalue", dest="gapped_evalue", type=float, default=0.005,
                         help="E-value threshold for the gapped (Smith-Waterman) alignment, "
                              "stage 2 of the MMseqs2 search (default: 0.005). Stage 1 is "
                              "the ungapped prefilter, gated by --ungapped-pvalue. "
                              "Passed as -e <nseq*6*evalue> to 'mmseqs search', where nseq "
                              "is the number of genome entries (six reading frames each). "
                              "Ignored in --max mode, which skips MMseqs2.")
    parser.add_argument("--ungapped-pvalue", dest="ungapped_pvalue", type=float, default=0.02,
                        help="Max per-pair ungapped p-value for the prefilter stage (default: 0.02). "
                             "Passed as --ungapped-pvalue to 'mmseqs search'. "
                             "Use 1.0 to disable p-value filtering (score floor only).")
    parser.add_argument("--prefilter-max-seqs", dest="prefilter_max_seqs", type=int, default=2147483647,
                         help="MMseqs2 prefilter max hits per protein family "
                              "(per query profile) kept from the prefilter (default: 2147483647, i.e. keep all). "
                              "Passed as --max-seqs to 'mmseqs ungappedprefilter'. "
                              "Ignored in --max mode, which skips MMseqs2.")
    parser.add_argument("--ungapped-calib", dest="ungapped_calib", default=None,
                         help="Path to sidecar cache for per-profile ungapped calibration "
                              "(default: .dummest-cache/ungappedcalib.tsv, shared across runs: "
                              "rows are keyed by profile-content hash, so sharing is safe). "
                              "Passed as --ungapped-calib to 'mmseqs search'. "
                              "Ignored in --max mode, which skips MMseqs2.")
    parser.add_argument("--ungapped-recalibrate", dest="ungapped_recalibrate", action="store_true",
                         help="Ignore cache entries and recalibrate every profile "
                              "(passed as --ungapped-recalibrate 1 to 'mmseqs search'). "
                              "Ignored in --max mode, which skips MMseqs2.")
    parser.add_argument("--mmseqs-bin", dest="mmseqs_bin", default=None,
                         help="Path to mmseqs binary (overrides default mmseqs from PATH)")
    parser.add_argument("--insert1", type=float, default=None,
                        help="DUMMEST 1-base insertion rate per base (default: dummest default 0.0171)")
    parser.add_argument("--insert2", type=float, default=None,
                        help="DUMMEST 2-base insertion rate per base (default: dummest default 0.0018)")
    parser.add_argument("--delete1", type=float, default=None,
                        help="DUMMEST 1-base deletion rate per base (default: dummest default 0.0328)")
    parser.add_argument("--delete2", type=float, default=None,
                        help="DUMMEST 2-base deletion rate per base (default: dummest default 0.0083)")
    parser.add_argument("--stop-codon-prob", dest="stop_codon_prob", type=float, default=None,
                        help="DUMMEST stop codon probability (default: dummest default 0.0005)")
    parser.add_argument("--bg-stop-codon-prob", dest="bg_stop_codon_prob", type=float, default=None,
                        help="DUMMEST background stop codon probability (default: dummest default 0.046875)")
    parser.add_argument("--tantan-threshold", dest="tantan_threshold", type=float, default=None,
                        help="DUMMEST tantan masking threshold (default: dummest default 0.5)")
    parser.add_argument("--evalue", "-e", dest="evalue", type=float, default=10,
                        help="DUMMEST E-value threshold (default: 10). Forwarded as both -e and -W to dummest.")
    parser.add_argument("--batch", dest="batch", type=int, default=800000,
                        help="DUMMEST stream chunk size in sequences (default: 800000). "
                             "Small values force multi-chunk streaming.")
    parser.add_argument("--trials", dest="trials", type=int, default=None,
                        help="DUMMEST calibration random sequences (-t). "
                             "Default: dummest default (1000).")

    args = parser.parse_args()

    if args.gapped_evalue is None or not args.gapped_evalue >= 0:
        parser.error("--gapped-evalue must be >= 0")
    if args.ungapped_pvalue is None or not 0 <= args.ungapped_pvalue <= 1:
        parser.error("--ungapped-pvalue must be in [0, 1]")
    if args.prefilter_max_seqs is None or args.prefilter_max_seqs < 1:
        parser.error("--prefilter-max-seqs must be >= 1")
    for _name in ("insert1", "insert2", "delete1", "delete2",
                  "stop_codon_prob", "bg_stop_codon_prob"):
        _v = getattr(args, _name)
        if _v is not None and not 0 <= _v < 1:
            parser.error(f"--{_name.replace('_', '-')} must be in [0, 1)")
    if args.tantan_threshold is not None and not 0 <= args.tantan_threshold <= 1:
        parser.error("--tantan-threshold must be in [0, 1]")
    if args.evalue is None or not args.evalue >= 0:
        parser.error("--evalue must be >= 0")
    if args.batch is not None and args.batch < 1:
        parser.error("--batch must be >= 1")
    if args.trials is not None and args.trials < 1:
        parser.error("--trials must be >= 1")

    dummer_extra_args = []
    if args.batch is not None:
        dummer_extra_args += ["--batch", str(args.batch)]
    if args.trials is not None:
        dummer_extra_args += ["-t", str(args.trials)]
    if args.insert1 is not None:
        dummer_extra_args += ["--insert1", str(args.insert1)]
    if args.insert2 is not None:
        dummer_extra_args += ["--insert2", str(args.insert2)]
    if args.delete1 is not None:
        dummer_extra_args += ["--delete1", str(args.delete1)]
    if args.delete2 is not None:
        dummer_extra_args += ["--delete2", str(args.delete2)]
    if args.stop_codon_prob is not None:
        dummer_extra_args += ["--stop-codon-prob", str(args.stop_codon_prob)]
    if args.bg_stop_codon_prob is not None:
        dummer_extra_args += ["--bg-stop-codon-prob", str(args.bg_stop_codon_prob)]
    if args.tantan_threshold is not None:
        dummer_extra_args += ["--tantan-threshold", str(args.tantan_threshold)]

    hmm_file = args.hmm_file
    msa_file = args.msa_file
    fa_file = args.fa_file
    cpus = args.cpus
    if args.ungapped_calib is None:
        calib_dir = _cache_dir()
        args.ungapped_calib = os.path.join(calib_dir, "ungappedcalib.tsv")

    tmp_parent = _tmp_base_dir(fa_file)
    if tmp_parent:
        print("# Genome >2GB, using CWD for temp instead of /tmp")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    dummer_exec = args.dummer_bin or os.path.join(script_dir, "../bin/dummestl")
    mmseqs_exec = args.mmseqs_bin or "mmseqs"

    # ---------------------------------------------------------
    # 1. Parse sequences and HMMs
    # ---------------------------------------------------------
    hmm_lens = {}
    with open(hmm_file, 'r') as f:
        curr_acc, curr_name = None, None
        for line in f:
            if line.startswith("NAME"):
                curr_name = line.split()[1].strip()
            elif line.startswith("ACC"):
                curr_acc = line.split()[1].strip()
            elif line.startswith("LENG"):
                length = int(line.split()[1].strip())
                if curr_acc: hmm_lens[curr_acc] = length
                if curr_name: hmm_lens[curr_name] = length
                curr_acc, curr_name = None, None

    dna_lens = {}
    dna_seqs = {}
    with open(fa_file, 'r') as f:
        curr_id = None
        curr_seq = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if curr_id:
                    dna_lens[curr_id] = len("".join(curr_seq))
                    dna_seqs[curr_id] = "".join(curr_seq)
                curr_id = line[1:].split()[0]
                curr_seq = []
            elif curr_id:
                curr_seq.append(line)
        if curr_id:
            dna_lens[curr_id] = len("".join(curr_seq))
            dna_seqs[curr_id] = "".join(curr_seq)

    # total search space: all genome lengths, doubled for both strands
    tot_seq_len = sum(dna_lens.values()) * 2

    # ---------------------------------------------------------
    # --max mode: skip mmseqs2, run dummer directly on all profiles x contigs
    # ---------------------------------------------------------
    if args.max:
        _base = tmp_parent or tempfile.gettempdir()
        merged_fa_path = os.path.join(_base, f"dummest_max.{os.getpid()}.fa")
        trans = str.maketrans("ACGTacgt", "TGCAtgca")

        with open(merged_fa_path, "w") as fout:
            for chrom, seq in dna_seqs.items():
                L = len(seq)
                rc_seq = seq.translate(trans)[::-1]
                for prof in sorted(hmm_lens.keys()):
                    fout.write(f">{chrom}/1-{L} length={L} profile={prof} plus_strand\n")
                    fout.write(f"{seq}\n")
                    fout.write(f">{chrom}/1-{L} length={L} profile={prof} minus_strand_revcomp\n")
                    fout.write(f"{rc_seq}\n")

        print(f"# Max-mode FASTA written to: {merged_fa_path}")

        if not args.skip_dummer:
            try:
                subprocess.run(
                    [dummer_exec, hmm_file, merged_fa_path, '-T', str(cpus), '--max', '-e', str(args.evalue), '-N', str(tot_seq_len)]
                    + dummer_extra_args,
                    env=os.environ.copy(), check=True,
                )
            except subprocess.CalledProcessError as e:
                print(f"Error: dummest encountered an issue (Exit status: {e.returncode})")
                sys.exit(1)

        if args.output_fa:
            import shutil
            shutil.copy2(merged_fa_path, args.output_fa)
            print(f"# Debug FASTA saved to: {args.output_fa}")
        else:
            os.remove(merged_fa_path)

        return

    with tempfile.TemporaryDirectory(prefix="mmseqs_tmp_", dir=tmp_parent, delete=True) as tmpdir:
        print(f"# Temporary directory is: {tmpdir}")

        # ---------------------------------------------------------
        # 3. MMseqs Protein-Protein Search
        # ---------------------------------------------------------
        db_dir = os.path.join(tmpdir, "db")
        os.makedirs(db_dir)

        use_gpu = _resolve_gpu(args.gpu)
        gpu_flag = "1" if use_gpu else "0"
        print(f"# MMseqs2 prefilter: {'GPU' if use_gpu else 'CPU-only'} (--gpu {gpu_flag})")

        ali_file = os.path.join(db_dir, f"result.ali")
        tmp_file = os.path.join(db_dir, f"tmp")

        if args.target_db_pad:
            target_db_pad = args.target_db_pad
            print(f"# Using existing target_db_pad: {target_db_pad}")
        else:
            prot_fa_path = os.path.join(tmpdir, "translated_6frame.pfa")
            # Parity with step3-dummer-precompute.sh:
            #   seqkit seq -m 3 <fa> | seqkit translate -f 6 -F
            # Short (<3nt) DNAs are auto-removed by the -m 3 pre-filter so
            # per-run and prebuilt DBs stay identical; empty protein frames
            # from >=3nt DNAs are intentionally retained.
            print("# 6-frame translate with -m 3 pre-filter (parity with step3-dummer-precompute.sh)")
            p1 = subprocess.Popen(["seqkit", "seq", "-m", "3", fa_file],
                                  stdout=subprocess.PIPE)
            p2 = subprocess.Popen(["seqkit", "translate", "-f", "6", "-F",
                                   "--threads", str(cpus),
                                   "-o", prot_fa_path],
                                  stdin=p1.stdout)
            p1.stdout.close()
            ret2 = p2.wait()
            ret1 = p1.wait()
            if ret1 != 0 or ret2 != 0:
                raise subprocess.CalledProcessError(ret2 if ret2 != 0 else ret1,
                                                    "seqkit seq -m 3 | seqkit translate -f 6 -F")

            target_db_pad = os.path.join(db_dir, "targetDB_pad")
            subprocess.run([mmseqs_exec, "createdb", prot_fa_path, target_db_pad, "--gpu", gpu_flag, "--threads", cpus], check=True, stdout=subprocess.DEVNULL)

        if args.query_db:
            query_db = args.query_db
            print(f"# Using existing query_db: {query_db}")
        else:
            query_db = os.path.join(db_dir, "queryDB")
            msa_db = os.path.join(db_dir, "msa_db")
            subprocess.run([mmseqs_exec, "convertmsa", msa_file, msa_db, "--identifier-field", "0"], check=True, stdout=subprocess.DEVNULL)
            subprocess.run([mmseqs_exec, "msa2profile", msa_db, query_db, "--threads", cpus], check=True, stdout=subprocess.DEVNULL)

        # Gapped search (Smith-Waterman) on prefilter survivors. The
        # prefilter stage is ungappedprefilter, where --ungapped-pvalue
        # filters hits before SW.
        # Final significance is the SW E-value (-e); dummer re-scores windows.
        mmseqs_cmd = [
            mmseqs_exec, "search", query_db, target_db_pad, ali_file, tmpdir,
            "--gpu", gpu_flag,
            "--threads", cpus,
            "-e", str(len(dna_seqs) * 6 * args.gapped_evalue),
            "--max-seqs", str(args.prefilter_max_seqs),
            "--prefilter-mode", str(args.prefilter_mode),
            "--ungapped-pvalue", str(args.ungapped_pvalue),
            "--ungapped-calib", args.ungapped_calib,
            "--alignment-mode", "2",
        ]
        if not use_gpu:
            mmseqs_cmd.extend(["--spaced-kmer-mode", "0"])
            mmseqs_cmd.extend(["-s", "7.5"])
        if args.ungapped_recalibrate:
            mmseqs_cmd.extend(["--ungapped-recalibrate", "1"])
        print(f"# Ungapped calibration cache: {args.ungapped_calib}"
              + (" (recalibrating)" if args.ungapped_recalibrate else ""))

        subprocess.run(mmseqs_cmd, check=True)
        subprocess.run([mmseqs_exec, "convertalis", query_db, target_db_pad, ali_file, tmp_file, "--threads", cpus], check=True, stdout=subprocess.DEVNULL)

        # ---------------------------------------------------------
        # 4. Map Amino Acid Hits -> Genomic DNA Windows
        # ---------------------------------------------------------
        #
        # Coordinate conventions:
        #
        #   mmseqs convertalis output (BLAST-tab fields):
        #     fields[6]=qstart, fields[7]=qend  — 1-indexed inclusive profile positions
        #     fields[8]=tstart, fields[9]=tend  — 1-indexed inclusive target protein positions
        #
        #   Six-frame translation:
        #     Forward frame N (N=1,2,3): protein[1] ↔ genome nucleotide [N-1] (0-indexed)
        #       protein position p ↔ genome nucleotide (N-1) + 3*(p-1)   (1st nt of codon)
        #       protein position p ↔ genome nucleotide (N-1) + 3*(p-1)+2 (3rd nt of codon)
        #     Reverse frame N (N=1,2,3): translation runs over reverse-complement of genome.
        #       RC position i (0-indexed) ↔ original genome position L-1-i
        #       protein position p in frame N ↔ 1st nt of codon at original position:
        #         L - N - 3*p + 3   (0-indexed, inclusive)
        #
        def parse_mmseqs_to_intervals(filepath):
            malformed = 0
            for lineno, line in enumerate(open(filepath), 1):
                if line.startswith("#"):
                    continue

                fields = line.split()
                if len(fields) < 12:
                    continue

                query_acc = fields[0]
                target_full = fields[1]
                try:
                    q_start = int(fields[6])
                    q_end   = int(fields[7])
                    t_start = int(fields[8])
                    t_end = int(fields[9])
                except ValueError:
                    malformed += 1
                    if malformed <= 5:
                        print(f"# WARNING: skipping convertalis line {lineno} "
                              f"with non-integer coords: {line.strip()!r}",
                              file=sys.stderr)
                    continue
                p_pos = max(1, (t_start + t_end) // 2)
                e_value = fields[10]
                bitscore = fields[11]

                *target_parts, frame_str = target_full.rsplit('_', 1)
                target_base = '_'.join(target_parts)
                try:
                    frame_val = int(frame_str.split('=')[1])
                except (IndexError, ValueError):
                    # Run-A hardening: log-and-skip a malformed convertalis
                    # target instead of crashing the hour-long run. The
                    # offender row is preserved via the failure copy below.
                    malformed += 1
                    if malformed <= 5:
                        print(f"# WARNING: skipping convertalis line {lineno} "
                              f"with malformed target (expected *_frame=±N): "
                              f"{target_full!r} :: {line.strip()!r}",
                              file=sys.stderr)
                    continue
                strand, frame = ('F', frame_val) if frame_val > 0 else ('R', abs(frame_val))

                L = dna_lens.get(target_base, 0)
                if L == 0:
                    continue
                pad = 3 * hmm_lens.get(query_acc, 0)

                base_pos = (frame - 1) + 3 * (p_pos - 1)

                if strand == 'F':
                    start = max(0, base_pos - pad)
                    end   = min(L, base_pos + 1 + pad)
                    fake_chrom = f"{target_base}|{query_acc}|+"
                    yield pybedtools.Interval(fake_chrom, start, end, query_acc, bitscore, '+')
                else:
                    start = max(0, L - base_pos - 1 - pad)
                    end = min(L, L - base_pos + pad)
                    fake_chrom = f"{target_base}|{query_acc}|-"
                    yield pybedtools.Interval(fake_chrom, start, end, query_acc, bitscore, '-')

            if malformed:
                print(f"# WARNING: skipped {malformed} malformed convertalis "
                      f"row(s) (see first 5 above)", file=sys.stderr)

        def unpack_intervals(feature):
            real_chrom, query_acc, strand = feature.chrom.split('|')
            return pybedtools.Interval(real_chrom, feature.start, feature.end, query_acc, ".", strand)

        try:
            merged_bed = pybedtools.BedTool(parse_mmseqs_to_intervals(tmp_file)) \
                .sort() \
                .merge() \
                .each(unpack_intervals)
        except Exception:
            # Preserve the offender for Run-A forensics: tmpdir is deleted
            # on exit, so copy the convertalis output to CWD before re-raise.
            import shutil
            rescue = os.path.join(os.getcwd(),
                                  f"convertalis.{os.path.basename(fa_file)}.tmp.failed")
            try:
                shutil.copy2(tmp_file, rescue)
                print(f"# convertalis output preserved at: {rescue}",
                      file=sys.stderr)
            except OSError as _e:
                print(f"# WARNING: could not preserve convertalis output: {_e}",
                      file=sys.stderr)
            raise

        # ---------------------------------------------------------
        # 5. Extract Final Genomic FASTA (Bedtools)
        # ---------------------------------------------------------
        merged_bed_path = os.path.join(tmpdir, "merged.bed")
        merged_bed.saveas(merged_bed_path)
        
        raw_fa_path = os.path.join(tmpdir, "raw_ext.fa")
        merged_fa_path = os.path.join(tmpdir, f"debug.fa")

        # bedtools writes <fi>.fai next to the input FASTA; if fa_file lives on
        # a read-only mount (e.g. the Apptainer image), that fails. Symlink it
        # into the writable tmpdir so the index is created there instead.
        bedtools_fa = os.path.join(tmpdir, os.path.basename(fa_file))
        if os.path.lexists(bedtools_fa):
            os.remove(bedtools_fa)
        os.symlink(os.path.abspath(fa_file), bedtools_fa)

        subprocess.run(["bedtools", "getfasta", "-fi", bedtools_fa, "-bed", merged_bed_path, "-s", "-name+", "-fo", raw_fa_path], check=True)

        with open(raw_fa_path, 'r') as fin, open(merged_fa_path, 'w') as fout:
            for line in fin:
                if line.startswith(">"):
                    header_data = line[1:].strip()
                    query, coord_part = header_data.split("::") if "::" in header_data else ("UNKNOWN", header_data)
                    chrom, rest = coord_part.split(":")
                    pos, strand_part = rest.split("(")
                    start_str, end_str = pos.split("-")
                    strand_sign = strand_part[0]
                    
                    start, end = int(start_str), int(end_str)
                    strand_label = "plus_strand" if strand_sign == '+' else "minus_strand_revcomp"

                    fout.write(f">{chrom}/{start+1}-{end} length={dna_lens.get(chrom, 0)} profile={query} {strand_label}\n")
                else:
                    fout.write(line)

        if args.output_fa:
            import shutil
            shutil.copy2(merged_fa_path, args.output_fa)
            print(f"# Debug FASTA saved to: {args.output_fa}")

        # ---------------------------------------------------------
        # 6. Run Dummer
        # ---------------------------------------------------------
        if not args.skip_dummer:
            custom_env = os.environ.copy()
            #custom_env["ASAN_OPTIONS"] = "detect_container_overflow=1:strict_memcmp=1"
            
            try:
                subprocess.run([dummer_exec, hmm_file, merged_fa_path, '-T', str(cpus), '-e', str(args.evalue), '-W', str(args.evalue), '-N', str(tot_seq_len)] + dummer_extra_args, env=custom_env, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: dummest encountered an issue (Exit status: {e.returncode})")
                sys.exit(1)

if __name__ == "__main__":
    main()

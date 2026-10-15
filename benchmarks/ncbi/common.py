#!/usr/bin/env python3
"""Shared routines for the NCBI AMRFinderPlus tool runners
(run_hmmsearch.py, run_bath.py, run_dummer.py) and compare_methods.py.

Covers sequence IO, the persistent cache (~/.cache/ncbi-data), input
preparation, raw hit parsers, family-stat aggregation and shared CLI args.
"""

import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import pysam
from Bio import AlignIO, SeqIO

from bath_shared.hitcounts import add_hit


SUBSAMPLE_DEFAULT = 1000
NUCLEOTIDE_CHARS = set("ACGTUNacgtun")
HMMDB_DEFAULT = os.environ.get("HMM_DB", "ncbi_hmms/ncbi_amr_families.hmm")
BHMM_DEFAULT = os.environ.get("BHMM", "ncbi_hmms/ncbi_amr_families.bhmm")
MSA_DEFAULT = os.environ.get("MSA", "ncbi_hmms/ncbi_amr_families.msa")
# Empty default means BATH binaries are resolved from PATH.
BATH_BIN_DEFAULT = os.environ.get("BATH_BIN", "")


def _default_cache_dir() -> str:
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(base, "ncbi-data")


CACHE_DIR_DEFAULT = _default_cache_dir()


def iter_fastx(path: str):
    """Yield records from a FASTA/FASTQ (optionally gzip) file via pysam."""
    handle = pysam.FastxFile(str(path))
    try:
        for rec in handle:
            yield rec
    finally:
        handle.close()


def count_sequences(path: str) -> int:
    return sum(1 for _ in iter_fastx(path))


def detect_seqtype(path: str) -> str:
    """Return 'nucleotide', 'protein' or 'unknown'."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".fastq", ".fq"):
        return "nucleotide"
    if ext in (".faa", ".pep"):
        return "protein"
    try:
        for rec in iter_fastx(path):
            seq = rec.sequence or ""
            if seq:
                nuc = sum(1 for c in seq if c in NUCLEOTIDE_CHARS)
                return "nucleotide" if nuc / len(seq) > 0.85 else "protein"
            break
    except Exception:
        pass
    return "unknown"


def read_fasta_records(path: str) -> list[tuple[str, str]]:
    """Read an (aligned) FASTA file into [(seqid, seq)] via Bio.SeqIO."""
    return [(rec.id, str(rec.seq)) for rec in SeqIO.parse(str(path), "fasta")]


def parse_seed_alignment(path: str) -> list[tuple[str, str]]:
    """Parse a SEED file (Stockholm or aligned FASTA) into [(seqid, seq)]."""
    with open(path) as f:
        head = f.read(2048)
    if head.startswith("# STOCKHOLM"):
        aln = AlignIO.read(str(path), "stockholm")
        return [(rec.id, str(rec.seq)) for rec in aln]
    return read_fasta_records(path)


# Persistent cache (~/.cache/ncbi-data), keyed by input path + subset N.
# Freshness: cache mtime >= input mtime and non-empty.

def _cache_stem(inpath: str) -> str:
    base = os.path.basename(os.path.abspath(inpath))
    stem, _ = os.path.splitext(base)
    digest = hashlib.sha1(os.path.abspath(inpath).encode()).hexdigest()[:8]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", stem) or "input"
    return f"{digest}_{safe}"


def _subset_tag(subsample: int | None) -> str:
    return f"sub{subsample}" if subsample is not None else "full"


def cached_subsample_path(inpath: str, subsample: int | None,
                          cache_dir: str) -> str:
    return os.path.join(
        cache_dir, f"{_cache_stem(inpath)}.{_subset_tag(subsample)}.fa"
    )


def cached_protein_path(inpath: str, subsample: int | None,
                        translate: str, cache_dir: str) -> str:
    return os.path.join(
        cache_dir,
        f"{_cache_stem(inpath)}.{_subset_tag(subsample)}.{translate}.faa",
    )


def cached_pad_prefix(inpath: str, subsample: int | None,
                      cache_dir: str) -> str:
    return os.path.join(
        cache_dir, f"{_cache_stem(inpath)}.{_subset_tag(subsample)}_targetDB_pad"
    )


# ORF-split cache tag. Bumped whenever the extractorfs parameters change so
# stale caches from older parameter sets are never silently reused.
ORF_CACHE_TAG = "orf20"


def cached_orf_prefix(inpath: str, subsample: int | None,
                      cache_dir: str) -> str:
    """Versioned prefix for the cached ORF-split target pair.

    The pair is <prefix>.contigDB (parent contigs, createdb --dbtype 2) and
    <prefix>.orfDB (extractorfs output). Legacy full-frame _targetDB_pad
    caches are intentionally not reused here.
    """
    return os.path.join(
        cache_dir,
        f"{_cache_stem(inpath)}.{_subset_tag(subsample)}_targetDB_{ORF_CACHE_TAG}",
    )


def _gpu_flag() -> str:
    """Mirror bin/dummest GPU default: 1 iff an NVIDIA GPU is usable."""
    try:
        r = subprocess.run(["nvidia-smi", "-L"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return "0"
    return "1" if r.returncode == 0 else "0"


def _is_fresh(cache_file: str, src: str) -> bool:
    try:
        return (
            os.path.exists(cache_file)
            and os.path.getsize(cache_file) > 0
            and os.path.getmtime(cache_file) >= os.path.getmtime(src)
        )
    except OSError:
        return False


def _is_pad_fresh(prefix: str, src: str) -> bool:
    # MMseqs createdb emits a multi-file DB; .index is the reliable marker.
    return _is_fresh(prefix + ".index", src) and os.path.exists(prefix)


def ensure_subsample(inpath: str, subsample: int | None, cache_dir: str,
                     use_cache: bool = True,
                     outdir: str | None = None) -> tuple[str, int]:
    """Subsample (if requested) and convert FASTQ->FASTA.

    Cached in *cache_dir* when *use_cache* (reused when fresh), otherwise
    written to *outdir*. Returns (fasta_path, n_seqs).
    """
    ext = os.path.splitext(inpath)[1].lower()
    is_fastq = ext in (".fastq", ".fq")

    if subsample is None and not is_fastq:
        return inpath, count_sequences(inpath)

    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        dest = cached_subsample_path(inpath, subsample, cache_dir)
        if _is_fresh(dest, inpath):
            n_cached = count_sequences(dest)
            # Guard against partial/truncated cache files: reuse only when
            # the sequence count matches the requested subsample size.
            if subsample is None or n_cached == subsample:
                print(f"  Reusing cached subsample: {dest}", file=sys.stderr)
                return dest, n_cached
            print(f"  WARNING: cached subsample {dest} has {n_cached} "
                  f"sequence(s), expected {subsample}; rebuilding.",
                  file=sys.stderr)
        print(f"  Caching subsample → {dest} ...", file=sys.stderr)
    else:
        dest = os.path.join(outdir, "prepped_input.fa")
        print(f"  Subsampling → {dest} ...", file=sys.stderr)

    pipe_cmds = []
    if subsample is not None:
        pipe_cmds.append(["seqtk", "sample", "-s42", inpath, str(subsample)])
    else:
        pipe_cmds.append(["cat", inpath])
    if is_fastq:
        pipe_cmds.append(["seqtk", "seq", "-A"])
    pipeline = " | ".join(" ".join(cmd) for cmd in pipe_cmds)

    tmp = f"{dest}.tmp.{os.getpid()}"
    with open(tmp, "w") as outf:
        subprocess.run(
            ["bash", "-c", pipeline],
            stdout=outf, stderr=subprocess.PIPE, text=True, check=True,
        )
    os.replace(tmp, dest)
    return dest, count_sequences(dest)


def ensure_cached_target_orfs(prepped_fa: str, pair_prefix: str,
                              cpu: int) -> tuple[str, str]:
    """Build (or reuse) the cached ORF-split target pair for run_dummer.py.

    Replicates bin/dummest's ORF-split build (createdb --dbtype 2 +
    extractorfs ANY->STOP, min-length 20, table 1, 6 frames) so the caller
    can pass --target-orf-db/--target-contig-db. Returns (orf_prefix,
    contig_prefix). Legacy full-frame _targetDB_pad caches are never
    consulted: the versioned pair name forces a clean rebuild.
    """
    contig_prefix = pair_prefix + ".contigDB"
    orf_prefix = pair_prefix + ".orfDB"
    os.makedirs(os.path.dirname(pair_prefix) or ".", exist_ok=True)
    if (_is_pad_fresh(contig_prefix, prepped_fa)
            and _is_pad_fresh(orf_prefix, prepped_fa)):
        print(f"  Reusing cached ORF-split target pair: {orf_prefix}",
              file=sys.stderr)
        return orf_prefix, contig_prefix
    print(f"  Building cached ORF-split target pair → {orf_prefix} ...",
          file=sys.stderr)
    tmp_prefix = pair_prefix + f".tmp.{os.getpid()}"
    tmp_contig = tmp_prefix + ".contigDB"
    tmp_orf = tmp_prefix + ".orfDB"
    for f in glob.glob(tmp_prefix + "*"):
        os.remove(f)
    try:
        # NOTE: no --gpu on createdb: a GPU-built nucl contigDB yields an
        # empty extractorfs result (0 ORFs). Search keeps the GPU default.
        subprocess.run(
            ["mmseqs", "createdb", prepped_fa, tmp_contig,
             "--dbtype", "2", "--threads", str(cpu)],
            check=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True,
        )
        subprocess.run(
            ["mmseqs", "extractorfs", tmp_contig, tmp_orf,
             "--translate", "1", "--min-length", "20",
             "--orf-start-mode", "1",
             "--contig-start-mode", "2", "--contig-end-mode", "2",
             "--forward-frames", "1,2,3", "--reverse-frames", "1,2,3",
             "--translation-table", "1",
             "--threads", str(cpu)],
            check=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True,
        )
        for f in glob.glob(tmp_prefix + "*"):
            os.replace(f, pair_prefix + f[len(tmp_prefix):])
    finally:
        for f in glob.glob(tmp_prefix + "*"):
            try:
                os.remove(f)
            except OSError:
                pass
    return orf_prefix, contig_prefix


def ensure_cached_target_pad(prepped_fa: str, pad_prefix: str,
                             cpu: int) -> str:
    """Build (or reuse) the MMseqs targetDB_pad for run_dummer.py.

    Replicates pipeline.py's translate + createdb step so the caller can
    pass --target-db-pad without modifying pipeline.py.
    """
    os.makedirs(os.path.dirname(pad_prefix) or ".", exist_ok=True)
    if _is_pad_fresh(pad_prefix, prepped_fa):
        print(f"  Reusing cached targetDB_pad: {pad_prefix}",
              file=sys.stderr)
        return pad_prefix
    print(f"  Building cached targetDB_pad → {pad_prefix} ...",
          file=sys.stderr)
    tmpdir = tempfile.mkdtemp(prefix="targetdb_pad_")
    try:
        prot_fa = os.path.join(tmpdir, "translated_6frame.pfa")
        # Parity with bin/dummest: short (<3nt) DNAs are removed by the -m 3
        # pre-filter so per-run and prebuilt DBs stay identical. seqkit seq
        # streams into seqkit translate (no intermediate file needed).
        with open(prot_fa, "w") as outf:
            p1 = subprocess.Popen(
                ["seqkit", "seq", "-m", "3", prepped_fa],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            p2 = subprocess.run(
                ["seqkit", "translate", "-f", "6", "-F",
                 "--threads", str(cpu), "-o", prot_fa],
                stdin=p1.stdout, stderr=subprocess.PIPE, text=True,
            )
            p1.stdout.close()
            p1.wait()
            if p1.returncode != 0 or p2.returncode != 0:
                raise subprocess.CalledProcessError(
                    p2.returncode if p2.returncode != 0 else p1.returncode,
                    "seqkit seq -m 3 | seqkit translate -f 6 -F")
        tmp_prefix = pad_prefix + f".tmp.{os.getpid()}"
        for f in glob.glob(tmp_prefix + "*"):
            os.remove(f)
        subprocess.run(
            ["mmseqs", "createdb", prot_fa, tmp_prefix,
             "--gpu", _gpu_flag(), "--threads", str(cpu)],
            check=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True,
        )
        for f in glob.glob(tmp_prefix + "*"):
            os.replace(f, pad_prefix + f[len(tmp_prefix):])
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return pad_prefix


def cached_querydb_prefix(msa: str) -> str:
    """Input-independent MMseqs query-profile prefix, co-located with the MSA."""
    return os.path.join(os.path.dirname(os.path.abspath(msa)), "queryDB")


def ensure_query_db(msa: str, prefix: str, cpu: int) -> str:
    """Build (or reuse) the MMseqs query-profile DB from *msa*."""
    if os.path.exists(prefix + ".index"):
        print(f"  Reusing MMseqs2 query-profile cache: {prefix}",
              file=sys.stderr)
        return prefix
    print(f"  Building MMseqs2 query-profile cache → {prefix} ...",
          file=sys.stderr)
    subprocess.run(
        ["mmseqs", "convertmsa", msa, prefix + ".msa_db",
         "--identifier-field", "0"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True,
    )
    subprocess.run(
        ["mmseqs", "msa2profile", prefix + ".msa_db", prefix,
         "--threads", str(cpu)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True,
    )
    return prefix


def prepare_translated_queries(
    inpath: str, subsample: int | None,
    cpu: int, cache_dir: str, use_cache: bool = True,
    outdir: str | None = None,
) -> tuple[str, int]:
    """Prepare a 6-frame translated protein FASTA for hmmsearch.

    Returns (path_to_protein_fasta, n_queries). Protein input is returned
    unchanged. Nucleotide input is subsampled (cached) and translated with
    seqkit; the translation mode tag is 'seqkit' to preserve existing cache
    filenames.
    """
    seqtype = detect_seqtype(inpath)

    if seqtype == "protein":
        print(f"Detected protein FASTA ({inpath}).", file=sys.stderr)
        return inpath, count_sequences(inpath)

    if seqtype == "unknown":
        print(f"Warning: could not detect type of {inpath}, "
              f"treating as protein.", file=sys.stderr)
        return inpath, count_sequences(inpath)

    print(f"Detected nucleotide ({inpath}).", file=sys.stderr)

    fasta_input, _ = ensure_subsample(
        inpath, subsample, cache_dir, use_cache, outdir)

    if use_cache:
        prepped = cached_protein_path(inpath, subsample, "seqkit", cache_dir)
        if _is_fresh(prepped, inpath):
            print(f"  Reusing cached translation: {prepped}",
                  file=sys.stderr)
            return prepped, count_sequences(prepped)
        print("  6-frame translation via seqkit ...", file=sys.stderr)
        print(f"  Caching translation → {prepped} ...", file=sys.stderr)
    else:
        prepped = os.path.join(outdir, "prepped.faa")
        print("  6-frame translation via seqkit ...", file=sys.stderr)

    # No --append-frame: hmmsearch must see bare read IDs so tblout col 1 is
    # already the parent read ID (see compare_methods.py).
    cmd = ["seqkit", "translate", "--frame", "6", "--threads", str(cpu),
           fasta_input]
    tmp = f"{prepped}.tmp.{os.getpid()}"
    with open(tmp, "w") as outf:
        subprocess.run(cmd, stdout=outf, stderr=subprocess.PIPE,
                       text=True, check=True)
    os.replace(tmp, prepped)

    if not use_cache and fasta_input != inpath:
        os.remove(fasta_input)

    n = count_sequences(prepped)
    print(f"  Translated → {n} protein sequences.", file=sys.stderr)
    return prepped, n


def format_evalue(val: float) -> str:
    if val == 0.0:
        return "0.0"
    return f"{val:.1e}"


def strip_aln_suffix(name: str) -> str:
    return name[:-4] if name.endswith(".aln") else name


# Raw hit parsers. All return (hits, pair_best) where hits is
# [(read, family, evalue, score, span)] (read may be None for DUMMER
# single-'s' blocks) and pair_best is {(read, family): smallest_E}.
# hmmsearch span is 0 (filled from --domtblout); BATH span is nucleotide
# aln length; DUMMER span is the reported span.
def _parse_tbl(path, min_fields, read_col, fam_col, evalue_col, score_col,
               span_fn):
    hits = []
    pair_best = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split()
            if len(fields) < min_fields:
                continue
            try:
                evalue = float(fields[evalue_col])
            except ValueError:
                continue
            fam = strip_aln_suffix(fields[fam_col])
            try:
                score = float(fields[score_col])
            except (ValueError, IndexError):
                score = float("-inf")
            add_hit(hits, pair_best, fields[read_col], fam, evalue, score,
                    span_fn(fields))
    return hits, pair_best


def parse_hmmer_tbl(path: str):
    return _parse_tbl(path, 19, 0, 2, 4, 5, lambda _f: 0)


def parse_bath_tbl(path: str):
    # BATH v2.0.0-rc6 --tblout layout (--fs 18 fields, --nofs 16 fields):
    #   hitID target acc query acc hmm_len hmm_from hmm_to seq_len
    #   ali_from ali_to E-value score bias PID [shifts stops] description
    def span(fields):
        try:
            return abs(int(fields[9]) - int(fields[10])) + 1
        except (ValueError, IndexError):
            return 0
    return _parse_tbl(path, 12, 1, 3, 11, 12, span)


def parse_hmmer_dom(path: str) -> dict:
    """Per-pair domain spans from hmmsearch --domtblout.

    Returns {(read, family): [(domain_ievalue, span_bp)]} with
    span_bp = (|ali_to - ali_from| + 1) * 3. Missing file -> {}.
    """
    lens = {}
    try:
        f = open(path)
    except OSError:
        return lens
    with f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split()
            if len(fields) < 22:
                continue
            try:
                ievalue = float(fields[12])
                span_bp = (abs(int(fields[18]) - int(fields[17])) + 1) * 3
            except (ValueError, IndexError):
                continue
            key = (fields[0], strip_aln_suffix(fields[3]))
            lens.setdefault(key, []).append((ievalue, span_bp))
    return lens


def parse_dummer(path: str, vocab=None):
    """Parse DUMMER output.

    Each 'a score=' block is followed by two 's ' lines: the profile
    (family) and the target read, usually in that order. *vocab* (model
    names seen in hmmsearch/BATH outputs) disambiguates; falls back to
    profile-first ordering for families only DUMMER found. Single-'s'-line
    blocks have read None (family count kept, no pair).
    """
    vocab = vocab or set()
    vocab_stripped = {strip_aln_suffix(v) for v in vocab}
    hits = []
    pair_best = {}
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if not line.startswith("a score="):
            continue
        parts = line.strip().split()
        e_val = None
        for p in parts:
            if p.startswith("E="):
                try:
                    e_val = float(p[2:])
                except ValueError:
                    e_val = None
                break
        if e_val is None:
            continue
        try:
            score = float(parts[1].split("=")[1])
        except (IndexError, ValueError):
            score = float("-inf")
        s_lines = []
        for j in (i + 1, i + 2):
            if j < len(lines) and lines[j].startswith("s "):
                toks = lines[j].split()
                if len(toks) < 2:
                    continue
                s_lines.append((toks[1], toks))
        if not s_lines:
            continue
        if len(s_lines) == 2:
            (t1, _), (t2, _) = s_lines
            s1, s2 = strip_aln_suffix(t1), strip_aln_suffix(t2)
            if s1 in vocab_stripped and s2 not in vocab_stripped:
                fam_tok, seq_tok = t1, t2
            elif s2 in vocab_stripped and s1 not in vocab_stripped:
                fam_tok, seq_tok = t2, t1
            else:
                fam_tok, seq_tok = t1, t2  # profile-first fallback
        else:
            fam_tok = s_lines[0][0]
            seq_tok = None
        fam = strip_aln_suffix(fam_tok)
        span = 0
        if seq_tok is not None:
            for tok, toks in s_lines:
                if tok == seq_tok and len(toks) > 3:
                    try:
                        span = int(toks[3])
                    except ValueError:
                        span = 0
                    break
            add_hit(hits, pair_best, seq_tok, fam, e_val, score, span)
        else:
            add_hit(hits, pair_best, None, fam, e_val, score, span)
    return hits, pair_best


def build_family_stats(hits) -> list[tuple[str, dict]]:
    """Aggregate hit tuples into sorted [(family, stats_dict)]."""
    family_stats: dict[str, dict] = {}
    for _read, fam, evalue, score, _span in hits:
        if fam not in family_stats:
            family_stats[fam] = {
                "n_hits": 0,
                "best_evalue": float("inf"),
                "best_score": float("-inf"),
            }
        family_stats[fam]["n_hits"] += 1
        if evalue < family_stats[fam]["best_evalue"]:
            family_stats[fam]["best_evalue"] = evalue
        if score > family_stats[fam]["best_score"]:
            family_stats[fam]["best_score"] = score
    return sorted(
        family_stats.items(), key=lambda x: x[1]["n_hits"], reverse=True
    )


def write_summary_tsv(path: str,
                      sorted_families: list[tuple[str, dict]]) -> None:
    with open(path, "w") as sf:
        sf.write("family\tn_hits\tbest_evalue\tbest_score\n")
        for fam, stats in sorted_families:
            sf.write(
                f"{fam}\t{stats['n_hits']}\t"
                f"{format_evalue(stats['best_evalue'])}\t"
                f"{stats['best_score']:.1f}\n"
            )


def write_empty_summary(path: str) -> None:
    write_summary_tsv(path, [])


def print_summary(
    label: str,
    sorted_families: list[tuple[str, dict]],
    hmm_n: int,
    raw_count: int,
    n_query: int,
    n_seq_with_hits: int,
    input_path: str,
    outfile_primary: str,
    outfile_secondary: str | None,
    summary_path: str,
    elapsed: float,
    total: float | None = None,
) -> None:
    print(file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  {label} Search Summary", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Input file:    {input_path}", file=sys.stderr)
    print(f"  Input seqs:    {raw_count:,}", file=sys.stderr)
    print(f"  Query seqs:    {n_query:,}", file=sys.stderr)
    print(f"  HMM database:  {hmm_n} families", file=sys.stderr)
    print(f"  Seqs w/ hits:  {n_seq_with_hits:,}", file=sys.stderr)
    if n_query > 0:
        print(f"  Hit rate:      {n_seq_with_hits / n_query * 100:.1f}%",
              file=sys.stderr)
    print(file=sys.stderr)

    if sorted_families:
        top_n = min(20, len(sorted_families))
        print(f"  Top {top_n} families by hit count:", file=sys.stderr)
        print(f"  {'Family':55s} {'Hits':>8s} {'Best E-val':>12s} "
              f"{'Best Score':>10s}", file=sys.stderr)
        print(f"  {'-'*55} {'-'*8} {'-'*12} {'-'*10}", file=sys.stderr)
        for fam, stats in sorted_families[:top_n]:
            fam_short = fam if len(fam) <= 55 else fam[:52] + "..."
            print(
                f"  {fam_short:55s} {stats['n_hits']:>8d} "
                f"{format_evalue(stats['best_evalue']):>12s} "
                f"{stats['best_score']:>8.1f}",
                file=sys.stderr,
            )
    else:
        print("  No hits found.", file=sys.stderr)

    print(file=sys.stderr)
    print(f"  Raw output:   {outfile_primary}", file=sys.stderr)
    if outfile_secondary:
        print(f"                {outfile_secondary}", file=sys.stderr)
    print(f"  Summary:      {summary_path}", file=sys.stderr)
    print(f"  Time:          {elapsed:.1f}s (search only)", file=sys.stderr)
    if total is not None:
        print(f"  Total time:    {total:.1f}s (incl. prep/translate)",
              file=sys.stderr)
    print("=" * 60, file=sys.stderr)


def finish_search(label, parse_fn, parse_path, display_primary,
                  display_secondary, summary_path, input_path, raw_count,
                  n_query, hmm_n, elapsed, total=None) -> None:
    """Parse hits, write summary.tsv and print the summary block.

    Handles the no-hits case (empty/missing *parse_path*) uniformly.
    *display_primary*/*display_secondary* are the paths shown to the user,
    which may differ from the file actually parsed.
    """
    if not os.path.exists(parse_path) or os.path.getsize(parse_path) == 0:
        print(f"No hits found ({os.path.basename(parse_path)} is empty).",
              file=sys.stderr)
        write_empty_summary(summary_path)
        print_summary(label, [], hmm_n, raw_count, n_query, 0, input_path,
                      display_primary, display_secondary, summary_path,
                      elapsed, total)
        return
    hits, _ = parse_fn(parse_path)
    n_seq_with_hits = len({read for read, *_ in hits})
    sorted_families = build_family_stats(hits)
    write_summary_tsv(summary_path, sorted_families)
    print_summary(label, sorted_families, hmm_n, raw_count, n_query,
                  n_seq_with_hits, input_path, display_primary,
                  display_secondary, summary_path, elapsed, total)


def add_scan_args(parser) -> None:
    """Add the arguments common to every tool runner."""
    parser.add_argument(
        "input",
        help="Input file: protein/nucleotide FASTA or FASTQ.  "
             "Auto-detects type by extension and content.",
    )
    parser.add_argument(
        "-o", "--outdir", default="results",
        help="Output directory (default: results)",
    )
    parser.add_argument("--cpu", type=int, default=16,
                        help="CPU threads (default: 16)")
    parser.add_argument("--evalue", type=float, default=0.01,
                        help="E-value threshold (default: 0.01)")
    parser.add_argument("--max-reads", type=int, default=SUBSAMPLE_DEFAULT,
                        help=f"Max reads to subsample "
                             f"(default: {SUBSAMPLE_DEFAULT}).")
    parser.add_argument("--max", action="store_true",
                        help="Max sensitivity: disable heuristic filters.")

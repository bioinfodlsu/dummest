#!/usr/bin/env python3
"""Build a profile-HMM database from NCBI AMRFinderPlus curated SEED
alignments (NCBIfam-AMRFinder) for metagenomic AMR gene detection.

Per model: hmmbuild -> hmmpress -> Stockholm MSA -> BATH .bhmm -> MMseqs2 DB.
No MAFFT: SEED files are already curated alignments, used verbatim.

Filtering: keep only type==AMR and scope==core rows from
NCBIfam-AMRFinder.tsv, map each to its SEED file via hmm_PGAP.tsv, and drop
models with fewer than --min-sequences sequences.

Usage:
    python build_ncbi_hmms.py <seed_dir> <ncbfam_tsv> <pgap_tsv>

Output (outdir, default ncbi_hmms/):
    ncbi_amr_families.hmm[/.h3*]  Combined pressed HMM database
    ncbi_amr_families.msa         Combined Stockholm MSA (BATH/DUMMER)
    ncbi_amr_families.bhmm        BATH-format HMM database
    ncbi_amr_ref.faa, ncbi_amr_refDB*  Reference proteins + MMseqs2 DB
    manifest.tsv, *.aln.fa

Requires: Python 3, HMMER 3 (hmmbuild/hmmpress), BATH (bathbuild),
          MMseqs2 (optional via --skip-mmseqs).
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict

import common


MIN_SEQUENCES = 10


def sanitise_filename(name: str) -> str:
    """Replace characters unsafe for filenames with underscores."""
    return re.sub(r'[^\w.-]+', '_', name)


def load_amr_accessions(tsv_path: str) -> dict[str, dict]:
    """Return {ncbi_accession: metadata} for type==AMR, scope==core rows."""
    kept: dict[str, dict] = {}
    with open(tsv_path, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if row.get('type') != 'AMR' or row.get('scope') != 'core':
                continue
            acc = row['#hmm_accession'].strip()
            kept[acc] = {
                'gene_symbol': row.get('gene_symbol', '').strip(),
                'description': row.get('hmm_description', '').strip(),
                'class': row.get('class', '').strip(),
                'subclass': row.get('subclass', '').strip(),
                'hierarchy_node': row.get('hierarchy_node', '').strip(),
            }
    return kept


def load_label_map(pgap_tsv: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return ({accession: label}, {label: accession}) from hmm_PGAP.tsv."""
    acc2label: dict[str, str] = {}
    with open(pgap_tsv, newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            acc = row['#ncbi_accession'].strip()
            label = row['label'].strip()
            if acc and label:
                acc2label[acc] = label
    label2acc = {v: k for k, v in acc2label.items()}
    return acc2label, label2acc


def validate_alignment(label: str, seed_path: str,
                       seqs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Validate a SEED alignment; exit hard if malformed.

    Returns records with '.' normalised to '-' (character replacement
    only, not realignment). Malformed = zero sequences, any empty
    sequence, or unequal row lengths.
    """
    if not seqs:
        sys.exit(f'MALFORMED SEED {label}: no sequences parsed '
                 f'from {seed_path}')
    clean: list[tuple[str, str]] = []
    for sid, seq in seqs:
        s = seq.replace('.', '-').replace(' ', '')
        if not s:
            sys.exit(f'MALFORMED SEED {label}: empty sequence {sid!r} '
                     f'in {seed_path}')
        clean.append((sid, s))
    lengths = {len(s) for _, s in clean}
    if len(lengths) != 1:
        sys.exit(f'MALFORMED SEED {label}: unequal row lengths '
                 f'{sorted(lengths)} in {seed_path}')
    return clean


def write_stockholm_block(outfh, model_name: str,
                          aln_records: list[tuple[str, str]]) -> None:
    """Append one Stockholm alignment block to an open file handle."""
    outfh.write('# STOCKHOLM 1.0\n\n')
    outfh.write(f'#=GF ID {model_name}\n\n')
    for sid, seq in aln_records:
        outfh.write(f'{sid:<30s} {seq}\n')
    outfh.write('//\n\n')


def find_mmseqs(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    found = shutil.which('mmseqs')
    if found:
        return found
    home_bin = os.path.expanduser('~/bin/mmseqs')
    if os.path.exists(home_bin):
        return home_bin
    return None


def audit(seed_dir: str, ncbfam_tsv: str, pgap_tsv: str, outdir: str,
          min_seqs: int) -> None:
    """Report which core-AMR models survive the filter, and why others drop.

    Writes <outdir>.kept_models.tsv and prints kept/removed/missing counts
    plus the class balance of kept models.
    """
    amr = load_amr_accessions(ncbfam_tsv)
    acc2label, _ = load_label_map(pgap_tsv)
    kept, thin, missing = [], [], []
    for acc, meta in sorted(amr.items()):
        label = acc2label.get(acc)
        if label is None:
            missing.append((acc, meta['gene_symbol'], 'no PGAP label'))
            continue
        sp = os.path.join(seed_dir, f'{label}-NCBIFAM.SEED')
        if not os.path.exists(sp):
            sp = os.path.join(seed_dir, f'{label}.SEED')
        if not os.path.exists(sp):
            missing.append((acc, label, 'no SEED file'))
            continue
        n = len(common.parse_seed_alignment(sp))
        (kept if n >= min_seqs else thin).append(
            (label, acc, n, meta['class']))
    print(f'core-AMR models in table : {len(amr)}')
    print(f'kept (>={min_seqs} seqs)      : {len(kept)}')
    print(f'removed (too few)        : {len(thin)}')
    print(f'missing (drift)          : {len(missing)}')
    for acc, lab, why in missing:
        print(f'  MISSING {acc} {lab}: {why}')
    print('class balance of kept models:')
    for cls, n in Counter(k[3] or '(none)' for k in kept).most_common():
        print(f'  {cls:40s} {n}')
    out = outdir + '.kept_models.tsv'
    with open(out, 'w') as f:
        f.write('model\taccession\tnseqs\tclass\n')
        for label, acc, n, cls in sorted(kept, key=lambda k: -k[2]):
            f.write(f'{label}\t{acc}\t{n}\t{cls}\n')
    print(f'wrote {out}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build profile HMMs from NCBI AMRFinderPlus SEED alignments'
    )
    parser.add_argument('seed_dir', nargs='?',
                        help='Extracted SEED directory '
                        '(e.g., /tmp/ncbiseed/SEED)')
    parser.add_argument('ncbfam_tsv', nargs='?',
                        help='NCBIfam-AMRFinder.tsv filter table')
    parser.add_argument('pgap_tsv', nargs='?',
                        help='hmm_PGAP.tsv (label <-> accession map)')
    parser.add_argument('-o', '--outdir', default='ncbi_hmms',
                        help='Output directory (default: ncbi_hmms)')
    parser.add_argument('-n', '--min-sequences', type=int,
                        default=MIN_SEQUENCES,
                        help=f'Minimum sequences per model '
                        f'(default: {MIN_SEQUENCES})')
    parser.add_argument('--skip-press', action='store_true',
                        help='Skip hmmpress step')
    parser.add_argument('--keep-raw-fasta', action='store_true',
                        help='Keep the raw (unaligned) per-model FASTA files')
    parser.add_argument('--skip-mmseqs', action='store_true',
                        help='Skip MMseqs2 DB creation (still writes ref FAA)')
    parser.add_argument('--mmseqs-bin', default=None,
                        help='Path to mmseqs binary '
                        '(default: auto-detect mmseqs or ~/bin/mmseqs)')
    parser.add_argument('--bath-bin', default=common.BATH_BIN_DEFAULT,
                        help='Path to BATH binaries directory '
                             '(default: $BATH_BIN, else PATH)')
    parser.add_argument('--ct', '--codon-table', type=int, default=1,
                        dest='codon_table',
                        help='NCBI codon translation table ID (default: 1)')
    parser.add_argument('--regenerate-msa', action='store_true',
                        help='Regenerate ncbi_amr_families.msa from existing '
                        '.aln.fa files')
    parser.add_argument('--audit', action='store_true',
                        help='Only run the filter audit (kept/removed/missing '
                             '+ class balance), then exit')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.regenerate_msa:
        print('Regenerating combined MSA from existing .aln.fa files ...',
              file=sys.stderr)
        aln_files = sorted(
            f for f in os.listdir(args.outdir) if f.endswith('.aln.fa')
        )
        if not aln_files:
            sys.exit(f'No .aln.fa files found in {args.outdir}')
        msa_path = os.path.join(args.outdir, 'ncbi_amr_families.msa')
        with open(msa_path, 'w') as msa_out:
            for fname in aln_files:
                model = fname[:-len('.aln.fa')]
                recs = common.read_fasta_records(os.path.join(args.outdir, fname))
                write_stockholm_block(msa_out, model, recs)
        print(f'  Written: {msa_path} ({len(aln_files)} models)',
              file=sys.stderr)
        return

    if not args.seed_dir:
        sys.exit('SEED directory is required (or use --regenerate-msa)')
    if not args.ncbfam_tsv:
        sys.exit('NCBIfam TSV is required (or use --regenerate-msa)')
    if not args.pgap_tsv:
        sys.exit('PGAP TSV is required (or use --regenerate-msa)')
    for p in (args.seed_dir, args.ncbfam_tsv, args.pgap_tsv):
        if not os.path.exists(p):
            sys.exit(f'Input not found: {p}')

    if args.audit:
        audit(args.seed_dir, args.ncbfam_tsv, args.pgap_tsv, args.outdir,
              args.min_sequences)
        return

    print('Loading core-AMR model list from NCBIfam-AMRFinder.tsv ...',
          file=sys.stderr)
    amr_models = load_amr_accessions(args.ncbfam_tsv)
    print(f'  {len(amr_models)} core-AMR models in filter table',
          file=sys.stderr)
    acc2label, _ = load_label_map(args.pgap_tsv)
    print(f'  {len(acc2label)} accession->label mappings from hmm_PGAP.tsv',
          file=sys.stderr)

    print('Resolving SEED files and applying count filter ...',
          file=sys.stderr)
    kept: list[dict] = []
    skipped_no_label = 0
    skipped_no_seed = 0
    for acc, meta in sorted(amr_models.items()):
        label = acc2label.get(acc)
        if label is None:
            skipped_no_label += 1
            print(f'  Warning: no PGAP label for {acc} '
                  f'({meta["gene_symbol"]}); skipped (version drift)',
                  file=sys.stderr)
            continue
        seed_path = os.path.join(args.seed_dir, f'{label}-NCBIFAM.SEED')
        alt_seed = os.path.join(args.seed_dir, f'{label}.SEED')
        if not os.path.exists(seed_path):
            seed_path = alt_seed
        if not os.path.exists(seed_path):
            skipped_no_seed += 1
            print(f'  Warning: SEED file missing for {label} ({acc}); skipped',
                  file=sys.stderr)
            continue
        seqs = validate_alignment(
            label, seed_path, common.parse_seed_alignment(seed_path))
        if len(seqs) < args.min_sequences:
            continue
        kept.append({'acc': acc, 'label': label, 'meta': meta,
                     'seqs': seqs, 'seed_path': seed_path})

    kept.sort(key=lambda k: len(k['seqs']), reverse=True)
    total_seqs = sum(len(k['seqs']) for k in kept)
    print(f'  {len(kept)} models with >={args.min_sequences} sequences '
          f'({total_seqs} total sequences)', file=sys.stderr)
    if skipped_no_label:
        print(f'  Warning: {skipped_no_label} models without PGAP label',
              file=sys.stderr)
    if skipped_no_seed:
        print(f'  Warning: {skipped_no_seed} models without SEED file',
              file=sys.stderr)
    class_counts: dict[str, int] = defaultdict(int)
    for k in kept:
        class_counts[k['meta']['class'] or '(none)'] += 1
    print('  Class balance of kept models:', file=sys.stderr)
    for cls, n in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f'    {cls:40s} {n:5d}', file=sys.stderr)

    if not kept:
        sys.exit('No models met the minimum sequence threshold. Exiting.')

    hmm_files: list[str] = []
    msa_path = os.path.join(args.outdir, 'ncbi_amr_families.msa')
    msa_out = open(msa_path, 'w')
    ref_faa_path = os.path.join(args.outdir, 'ncbi_amr_ref.faa')
    ref_faa_out = open(ref_faa_path, 'w')
    manifest_rows: list[dict] = []

    for i, k in enumerate(kept, 1):
        label, acc, meta, seqs = (k['label'], k['acc'], k['meta'], k['seqs'])
        model = sanitise_filename(label)
        print(f'\n[{i}/{len(kept)}] {label} ({acc}): {len(seqs)} sequences',
              file=sys.stderr)

        fa_path = os.path.join(args.outdir, f'{model}.fa')
        with open(fa_path, 'w') as fh:
            for sid, seq in seqs:
                fh.write(f'>{sid}\n{seq}\n')
                ref_faa_out.write(f'>{model}|{sid}\n{seq}\n')

        # SEED columns are curated; used verbatim (validated upfront).
        aln_path = os.path.join(args.outdir, f'{model}.aln.fa')
        with open(aln_path, 'w') as fh:
            for sid, seq in seqs:
                fh.write(f'>{sid}\n{seq}\n')
        print(f'    Alignment staged: {aln_path}', file=sys.stderr)

        hmm_path = os.path.join(args.outdir, f'{model}.hmm')
        subprocess.run(
            ['hmmbuild', '-n', f'{model}', hmm_path, aln_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        print(f'    HMM written: {hmm_path}', file=sys.stderr)
        hmm_files.append(hmm_path)

        aln_recs = common.read_fasta_records(aln_path)
        write_stockholm_block(msa_out, model, aln_recs)

        manifest_rows.append({
            'model': model, 'accession': acc, 'nseqs': str(len(seqs)),
            'class': meta['class'], 'subclass': meta['subclass'],
            'gene_symbol': meta['gene_symbol'],
            'description': meta['description'],
            'hierarchy_node': meta['hierarchy_node'],
        })

        if not args.keep_raw_fasta:
            os.remove(fa_path)

    msa_out.close()
    ref_faa_out.close()

    manifest_path = os.path.join(args.outdir, 'manifest.tsv')
    with open(manifest_path, 'w', newline='') as mf:
        writer = csv.DictWriter(
            mf, fieldnames=['model', 'accession', 'nseqs', 'class',
                            'subclass', 'gene_symbol', 'description',
                            'hierarchy_node'], delimiter='\t')
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f'\nManifest written: {manifest_path} ({len(manifest_rows)} models)',
          file=sys.stderr)

    combined_hmm = os.path.join(args.outdir, 'ncbi_amr_families.hmm')
    print(f'\nConcatenating {len(hmm_files)} HMMs into {combined_hmm} ...',
          file=sys.stderr)
    with open(combined_hmm, 'w') as out:
        for hmm_file in hmm_files:
            with open(hmm_file) as f:
                out.write(f.read())
            os.remove(hmm_file)

    if not args.skip_press:
        print('Running hmmpress ...', file=sys.stderr)
        subprocess.run(
            ['hmmpress', combined_hmm],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            check=True,
        )
        print(f'  Indexed: {combined_hmm}.h3f, .h3i, .h3m, .h3p',
              file=sys.stderr)

    combined_bhmm = os.path.join(args.outdir, 'ncbi_amr_families.bhmm')
    print(f'\nBuilding BATH HMM database {combined_bhmm} ...', file=sys.stderr)
    subprocess.run(
        [os.path.join(args.bath_bin, 'bathbuild'),
         '--ct', str(args.codon_table), combined_bhmm, msa_path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        check=True,
    )
    print(f'  BATH HMMs: {combined_bhmm}', file=sys.stderr)

    if not args.skip_mmseqs:
        mmseqs = find_mmseqs(args.mmseqs_bin)
        if mmseqs is None:
            print('  Warning: mmseqs binary not found; ref FAA kept at '
                  f'{ref_faa_path} but DB not created. Pass --mmseqs-bin.',
                  file=sys.stderr)
        else:
            ref_db = os.path.join(args.outdir, 'ncbi_amr_refDB')
            print(f'\nBuilding MMseqs2 DB {ref_db} with {mmseqs} ...',
                  file=sys.stderr)
            subprocess.run(
                [mmseqs, 'createdb', ref_faa_path, ref_db],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                check=True,
            )
            print(f'  MMseqs2 DB: {ref_db}', file=sys.stderr)

    print(f'\nDone. {len(kept)} models in {combined_hmm}', file=sys.stderr)
    print('\nExample searches:', file=sys.stderr)
    print('  # hmmsearch (via run_hmmsearch.py):', file=sys.stderr)
    print(f'    python run_hmmsearch.py genome.fa --hmmdb {combined_hmm}',
          file=sys.stderr)
    print('  # BATH (via run_bath.py):', file=sys.stderr)
    print(f'    python run_bath.py genome.fa --hmmdb {combined_bhmm}',
          file=sys.stderr)
    print('  # DUMMER (via run_dummer.py):', file=sys.stderr)
    print(f'    python run_dummer.py genome.fa --hmmdb {combined_hmm} '
          f'--msa {msa_path}', file=sys.stderr)


if __name__ == '__main__':
    main()

# Badread helpers

Transmark-specific helper scripts used by `../step2-frameshift.sh`. They are
vendored here (instead of being patched into a Badread checkout) so the
pipeline depends only on an **upstream** Badread clone.

## Contents

| Script | Purpose |
|---|---|
| `map_gene_reads.py` | Map gene coordinates from a reference BED onto Badread-simulated reads, producing a TSV of gene occurrences on reads. Enforces the 3' clamp, a parent-fragment alignment guard, and `--min-length` (default 10). Needs the `badread` + `edlib` Python packages. |
| `pos_bed_converter.py` | Convert between Transmark `.pos` and BED6 (`pos2bed` / `bed2pos`). Pure standard library. |
| `audit_mapping.py` | Validate a `mapped.tsv` against its reads and gene BED (bounds, strand, ref-interval membership, duplicates). Exits non-zero on any violation. |
| `test_helpers.py` | Fast unit tests for the converter and the mapper's coordinate/overlap math. No simulator. |
| `test_integration.py` | End-to-end tests: runs Badread on small synthetic references and checks mapping correctness. |
| `verify_boundaries.py` | Manual visual check of single-base mapping at the gene start/end boundaries. |

## Requirements

* An upstream Badread clone (`github.com/rrwick/Badread`); point `BADREAD_DIR`
  at it (default: `/home/harvsftw/Documents/bioinfo/Badread`).
* Python packages: `badread` and `edlib` (Badread itself also uses `numpy`,
  `matplotlib`, `scipy`).

`../step2-frameshift.sh` runs `map_gene_reads.py` + `pos_bed_converter.py` from
this directory and `$BADREAD_DIR/badread-runner.py` for the simulation.

## Tests

```bash
# fast unit tests (no Badread)
python3 test_helpers.py

# end-to-end (invokes Badread; slower). Skips automatically if BADREAD_DIR is unset.
BADREAD_DIR=/path/to/Badread python3 test_integration.py
BADREAD_DIR=/path/to/Badread python3 test_integration.py -k linear   # one scenario
```

## Auditing a run

```bash
# point at a step2 scratch dir (infers mapped.tsv / reads.fastq / genes.bed)
python3 audit_mapping.py --dir ../dumpsite/tmp-frameshift/transmark-full

# or give files explicitly (reads may be FASTQ/FASTA, plain or .gz)
python3 audit_mapping.py --mapped mapped.tsv --reads reads.fastq --genes genes.bed
```

Exit status is non-zero if any coordinate falls outside the read bounds, any
`read_gene_strand` is wrong, or any ref interval is not a BED gene.

## Manual boundary check

```bash
BADREAD_DIR=/path/to/Badread python3 verify_boundaries.py
BADREAD_DIR=/path/to/Badread python3 verify_boundaries.py --strand - --seed 7
```

Prints `Ref`/`Read` alignments around the gene start and end with the mapped
boundary positions bracketed, plus a `MATCH`/`mismatch` summary per boundary.

## Mapping guarantees

`map_gene_reads.py` enforces three things when placing a gene on a read:

1. **3' end clamp.** When a gene's 3' end coincides with the fragment end and
   that terminal base aligns as an insertion (present in the reference fragment,
   absent from the read), `map_coordinate` returns the end-of-read target
   position; the `+ 1` used to close the interval would then land one base past
   the read. The interval is clamped to `[0, len(read)]`:
   ```python
   read_len = len(seq)
   if read_start < 0:
       read_start = 0
   if read_end > read_len:
       read_end = read_len
   if read_start >= read_end:
       continue
   ```
2. **Parent-fragment alignment guard.** If the read does not actually contain
   the reference fragment, edlib returns an all-insertion CIGAR (e.g. `10I`)
   with `start_loc == 0` and `editDistance == len(fragment)`, which would map
   every gene to read position 0. Such reads are skipped:
   ```python
   start_loc, end_loc = res['locations'][0]
   if start_loc is None or start_loc < 0 or res['editDistance'] >= len(fragment_seq):
       continue
   ```
3. **Minimum mapped length (`--min-length`, default 10).** Gene occurrences
   whose mapped interval on the read is shorter than this are dropped. This
   removes tiny edge/sliver positives (e.g. a gene that only clips the read's
   parent fragment by a few bases):
   ```python
   if read_end - read_start < args.min_length:
       continue
   ```

## Regenerating old runs (`amend-fs-pos.sh`)

Old runs can be rebuilt from the retained simulated reads **without re-running
Badread or any alignment tool** (bathsearch/dummer/last). The search tools run
only on the query + target files; the `.pos` is consumed afterwards by
`transmark-idpositives.pl` and `transmark-pp.sh`, and the raw sorted hits
(`*.out.tmp.sort`) are retained. `amend-fs-pos.sh` is **regenerate-only** and
strict: it needs the step-2 scratch `reads.fastq` + `genes.bed` and the step-1
`<base>.fa`, and errors otherwise.

```bash
cd ..                              # transmark/
./amend-fs-pos.sh --dry-run all    # diff the regenerated .pos vs current (no changes)
./amend-fs-pos.sh all              # regenerate .pos, re-run step-2.5, relabel .out, re-run transmark-pp.sh
./amend-fs-pos.sh lite             # one variant only
./amend-fs-pos.sh --scratch DIR all   # override the step2 scratch dir
```

Past runs were regenerated with the current mapper (3' clamp + alignment
guard + `--min-length` 10), so their `.pos` gained/lost rows versus the
originals and the `.out/.mer/.cover/.tpr/.xy` analytics were regenerated
accordingly. The mapper re-aligns every read (no simulator), so the full
benchmarks take a while.

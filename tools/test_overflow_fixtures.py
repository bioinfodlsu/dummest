#!/usr/bin/env python3
"""Deterministic fixtures for test_overflow.sh.

Usage: test_overflow_fixtures.py <fixdir>

Reads transmark-full.AA.hmm from the current directory and writes:
  <fixdir>/herpes.hmm, <fixdir>/herpes.fa  -- a 1359-codon profile and a
      10077-nt contig with its consensus gene inserted at ~3003.
  <fixdir>/medium.fa                        -- a 500-codon slice of that gene
      with random flanks, a case that overflows float but not double.
"""
import random
import re
import sys


def make_herpes(fix):
    START, END = 360383, 364485
    with open("transmark-full.AA.hmm") as f:
        lines = f.readlines()
    block = lines[START - 1:END]
    with open(f"{fix}/herpes.hmm", "w") as f:
        f.write("".join(block))

    hi = next(i for i, l in enumerate(block) if l.startswith("HMM "))
    AA = block[hi].split()[1:]
    assert len(AA) == 20

    pos, cur, match_vals = hi + 1, None, []
    while pos < len(block) and not block[pos].startswith("//"):
        line = block[pos]
        m = re.match(r"^\s*(\d+)\s+(.*)$", line)
        if m and len(m.group(2).split()) >= 5:
            if cur and len(cur) >= 20:
                match_vals.append(cur[:20])
            cur = []
            rest = m.group(2)
        else:
            rest = line
        if cur is not None:
            for tok in rest.split():
                try:
                    cur.append(float(tok))
                except ValueError:
                    break
                if len(cur) == 20:
                    break
        pos += 1
    if cur and len(cur) >= 20:
        match_vals.append(cur[:20])

    cons = "".join(AA[v.index(min(v))] for v in match_vals)
    CODON = {"A": "GCT", "R": "CGT", "N": "AAT", "D": "GAT", "C": "TGT",
             "Q": "CAA", "E": "GAA", "G": "GGT", "H": "CAT", "I": "ATT",
             "L": "CTG", "K": "AAA", "M": "ATG", "F": "TTT", "P": "CCT",
             "S": "TCT", "T": "ACT", "W": "TGG", "Y": "TAT", "V": "GTT"}
    dna = "".join(CODON.get(aa, "NNN") for aa in cons)
    assert "NNN" not in dna, "nonstandard AA in consensus"

    rng = random.Random(7)
    flank = lambda n: "".join(rng.choice("ACGT") for _ in range(n))
    contig = flank(3000) + dna + flank(3000)
    assert len(contig) == 10077, len(contig)
    with open(f"{fix}/herpes.fa", "w") as f:
        f.write(">herpes0\n")
        for k in range(0, len(contig), 80):
            f.write(contig[k:k + 80] + "\n")
    print(f"herpes fixtures ok: {len(match_vals)} match states, {len(contig)} nt")


def make_medium(fix):
    seq = "".join(l.strip() for l in open(f"{fix}/herpes.fa") if not l.startswith(">"))
    assert len(seq) == 10077, len(seq)
    dna = seq[3000:3000 + 4077]
    slice_dna = dna[430 * 3:(430 + 500) * 3]
    assert len(slice_dna) == 1500

    rng = random.Random(1234)
    flank = lambda n: "".join(rng.choice("ACGT") for _ in range(n))
    contig = flank(2000) + slice_dna + flank(2000)
    assert len(contig) == 5500, len(contig)
    with open(f"{fix}/medium.fa", "w") as f:
        f.write(">medium0\n")
        for k in range(0, len(contig), 80):
            f.write(contig[k:k + 80] + "\n")
    print(f"medium fixture ok: {len(contig)} nt")


def main():
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <fixdir>")
    fix = sys.argv[1]
    make_herpes(fix)
    make_medium(fix)


if __name__ == "__main__":
    main()

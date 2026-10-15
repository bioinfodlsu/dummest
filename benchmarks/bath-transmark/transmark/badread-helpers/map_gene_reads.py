#!/usr/bin/env python3
"""
This script takes Badread simulated reads (FASTQ format) and maps coordinates of genes from a 
reference genome (BED format) to find the exact positions of those genes on the simulated reads.
"""

import argparse
import gzip
import re
import sys
import edlib

from badread.misc import load_fasta, reverse_complement

def get_compression_type(filename):
    magic_dict = {'gz': (b'\x1f', b'\x8b', b'\x08')}
    max_len = max(len(x) for x in magic_dict.values())
    with open(str(filename), 'rb') as f:
        file_start = f.read(max_len)
    for file_type, magic_bytes in magic_dict.items():
        if file_start.startswith(magic_bytes):
            return file_type
    return 'plain'

def get_open_func(filename):
    if get_compression_type(filename) == 'gz':
        return gzip.open
    return open

def iterate_fastq(filename):
    with get_open_func(filename)(filename, 'rt') as fastq:
        for line in fastq:
            line = line.strip()
            if len(line) == 0:
                continue
            if not line.startswith('@'):
                continue
            name = line[1:].split()[0]
            header = line
            sequence = next(fastq).strip()
            _ = next(fastq)
            qualities = next(fastq).strip()
            yield name, header, sequence, qualities

def load_genes(bed_file):
    # Dict mapping contig -> list of (start, end, gene_name, gene_strand)
    genes = {}
    with open(bed_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                parts = line.split()
                if len(parts) < 4:
                    continue
            chrom = parts[0]
            start = int(parts[1])
            end = int(parts[2])
            name = parts[3]
            strand = parts[5] if len(parts) >= 6 else '+'
            if chrom not in genes:
                genes[chrom] = []
            genes[chrom].append((start, end, name, strand))
    return genes

def map_coordinate(cigar, start_loc, q_target):
    q_pos = 0
    t_pos = start_loc
    
    parts = re.findall(r'\d+[\w=]', cigar)
    for part in parts:
        op_len = int(part[:-1])
        op_type = part[-1]
        
        if op_type in ('=', 'X'):
            if q_pos <= q_target < q_pos + op_len:
                return t_pos + (q_target - q_pos)
            q_pos += op_len
            t_pos += op_len
        elif op_type == 'I':
            if q_pos <= q_target < q_pos + op_len:
                return t_pos
            q_pos += op_len
        elif op_type == 'D':
            t_pos += op_len
    return t_pos

def get_overlaps(g_start, g_end, r_start, r_end, N):
    overlaps = []
    if r_end <= N:
        o_start = max(g_start, r_start)
        o_end = min(g_end, r_end)
        if o_start < o_end:
            overlaps.append((o_start - r_start, o_end - r_start))
    else:
        # Segment 1: [r_start, N)
        o_start1 = max(g_start, r_start)
        o_end1 = min(g_end, N)
        if o_start1 < o_end1:
            overlaps.append((o_start1 - r_start, o_end1 - r_start))
            
        # Segment 2: [0, r_end - N)
        o_start2 = max(g_start, 0)
        o_end2 = min(g_end, r_end - N)
        if o_start2 < o_end2:
            offset = N - r_start
            overlaps.append((offset + o_start2, offset + o_end2))
    return overlaps

def main():
    parser = argparse.ArgumentParser(description="Map gene coordinates from reference genome to Badread simulated reads.")
    parser.add_argument('--reference', required=True, help="Reference FASTA file")
    parser.add_argument('--reads', required=True, help="FASTQ file of simulated reads (can be gzipped)")
    parser.add_argument('--genes', required=True, help="BED file of gene coordinates on reference (0-based, half-open)")
    parser.add_argument('--output', required=True, help="Output file (BED/TSV format of gene positions on reads)")
    parser.add_argument('--min-length', type=int, default=10,
                        help="Minimum mapped length (bp) on a read for a gene occurrence to be "
                             "reported; shorter edge/sliver overlaps are rejected (default: 10)")
    args = parser.parse_args()

    print("Loading reference FASTA...", file=sys.stderr)
    ref_seqs, _, ref_circular, _, _ = load_fasta(args.reference)
    rev_comp_ref_seqs = {name: reverse_complement(seq) for name, seq in ref_seqs.items()}

    print("Loading gene BED file...", file=sys.stderr)
    genes = load_genes(args.genes)

    print("Processing reads and mapping coordinates...", file=sys.stderr)
    mapped_count = 0
    with open(args.output, 'w') as out_f:
        # Write header for TSV
        out_f.write("read_id\tgene_name\tread_start\tread_end\tread_gene_strand\tref_contig\tref_start\tref_end\n")
        
        for name, header, seq, qual in iterate_fastq(args.reads):
            # Skip non-standard reads
            if ' chimera ' in header or ' random_seq ' in header or ' junk_seq ' in header:
                continue
            
            # FASTQ header format: @uuid contig,strand,start-end length=...
            header_parts = header.split(' ')
            if len(header_parts) < 2:
                continue
            meta_str = header_parts[1]
            meta_parts = meta_str.split(',')
            if len(meta_parts) < 3:
                continue
            
            contig = meta_parts[0]
            strand = meta_parts[1]  # '+strand' or '-strand'
            coord_str = meta_parts[2]
            
            if '-' not in coord_str:
                continue
            try:
                r_start, r_end = map(int, coord_str.split('-'))
            except ValueError:
                continue
            
            if contig not in ref_seqs or contig not in genes:
                continue
            
            N = len(ref_seqs[contig])
            
            # Determine fragment sequence
            if strand == '+strand':
                if r_end <= N:
                    fragment_seq = ref_seqs[contig][r_start:r_end]
                else:
                    fragment_seq = ref_seqs[contig][r_start:] + ref_seqs[contig][:r_end - N]
            else:  # -strand
                if r_end <= N:
                    fragment_seq = rev_comp_ref_seqs[contig][r_start:r_end]
                else:
                    fragment_seq = rev_comp_ref_seqs[contig][r_start:] + rev_comp_ref_seqs[contig][:r_end - N]
            
            # Align parent fragment to read
            # HW (infix) mode: aligns entire query (fragment_seq) to a substring of target (read)
            res = edlib.align(fragment_seq, seq, mode='HW', task='path')
            cigar = res['cigar']
            start_loc, end_loc = res['locations'][0]
            # Reject reads that don't actually contain the parent fragment. When the
            # fragment has no real match in the read, edlib returns an all-insertion
            # CIGAR (e.g. '10I') with start_loc 0 and editDistance == len(fragment),
            # which would otherwise map every gene to read position 0.
            if start_loc is None or start_loc < 0 or res['editDistance'] >= len(fragment_seq):
                continue

            # Map overlaps
            for G_start, G_end, gene_name, gene_strand in genes[contig]:
                # Convert gene coordinates to current strand
                if strand == '+strand':
                    g_start = G_start
                    g_end = G_end
                else:
                    g_start = N - G_end
                    g_end = N - G_start
                
                overlaps = get_overlaps(g_start, g_end, r_start, r_end, N)
                for o_frag_start, o_frag_end in overlaps:
                    # Map fragment-relative coordinates to read-relative coordinates
                    read_start = map_coordinate(cigar, start_loc, o_frag_start)
                    read_end = map_coordinate(cigar, start_loc, o_frag_end - 1) + 1
                    # Clamp to the read's bounds. When a gene's end coincides with the
                    # fragment end and that terminal base aligns as an insertion (present
                    # in the reference fragment, absent from the read), map_coordinate
                    # returns the end-of-read target position and the "+ 1" above lands one
                    # base past the read (read_end == len(read) + 1). The mapped interval
                    # must stay within [0, len(read)].
                    read_len = len(seq)
                    if read_start < 0:
                        read_start = 0
                    if read_end > read_len:
                        read_end = read_len
                    if read_start >= read_end:
                        continue
                    # Reject tiny edge/sliver overlaps (e.g. a gene that only clips
                    # the read's parent fragment by a few bases).
                    if read_end - read_start < args.min_length:
                        continue

                    # Calculate correct gene strand on the read:
                    # If read is +strand, the gene strand on the read matches its genomic strand.
                    # If read is -strand, it is the opposite.
                    if strand == '+strand':
                        read_gene_strand = gene_strand
                    else:
                        read_gene_strand = '-' if gene_strand == '+' else '+'
                    
                    # We output the mapping
                    out_f.write(f"{name}\t{gene_name}\t{read_start}\t{read_end}\t{read_gene_strand}\t{contig}\t{G_start}\t{G_end}\n")
                    mapped_count += 1

    print(f"Done! Mapped {mapped_count} gene occurrences to reads. Output saved to {args.output}", file=sys.stderr)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import sys
import argparse

def pos_to_bed(pos_path, bed_path):
    with open(pos_path, 'r') as f_in, open(bed_path, 'w') as f_out:
        for line in f_in:
            line_strip = line.strip()
            if not line_strip or line_strip.startswith('#'):
                continue
            cols = line_strip.split()
            if len(cols) < 5:
                print(f"Warning: skipping malformed line: {line_strip}", file=sys.stderr)
                continue
            
            col1 = cols[0]
            col2 = cols[1]
            col3 = cols[2]
            try:
                col4 = int(cols[3])
                col5 = int(cols[4])
            except ValueError:
                print(f"Warning: skipping line with invalid coordinates: {line_strip}", file=sys.stderr)
                continue
            
            # Determine strand based on start vs end position
            if col4 <= col5:
                strand = '+'
                start = col4 - 1
                end = col5
            else:
                strand = '-'
                start = col5 - 1
                end = col4
                
            # Pack col1 and col3 to avoid losing information
            name = f"{col1}|{col3}"
            score = "0"
            
            f_out.write(f"{col2}\t{start}\t{end}\t{name}\t{score}\t{strand}\n")

def bed_to_pos(bed_path, pos_path):
    with open(bed_path, 'r') as f_in, open(pos_path, 'w') as f_out:
        for line in f_in:
            line_strip = line.strip()
            if not line_strip or line_strip.startswith('#') or line_strip.startswith('track') or line_strip.startswith('browser'):
                continue
            cols = line_strip.split()
            if len(cols) < 6:
                print(f"Warning: skipping line with fewer than 6 columns (BED6 required): {line_strip}", file=sys.stderr)
                continue
            
            chrom = cols[0]
            try:
                start = int(cols[1])
                end = int(cols[2])
            except ValueError:
                print(f"Warning: skipping line with invalid coordinates: {line_strip}", file=sys.stderr)
                continue
                
            name_field = cols[3]
            # Split by | to retrieve col1 and col3 if possible
            if '|' in name_field:
                parts = name_field.split('|', 1)
                col1 = parts[0]
                col3 = parts[1]
            else:
                col1 = name_field
                col3 = name_field
                
            score = cols[4]
            strand = cols[5]
            
            if strand == '+':
                col4 = start + 1
                col5 = end
            elif strand == '-':
                col4 = end
                col5 = start + 1
            else:
                print(f"Warning: skipping line with invalid strand '{strand}': {line_strip}", file=sys.stderr)
                continue
            
            # Format columns like transmark-create does:
            # fprintf(tmppossummfp, "%-35s %-10s %-35s %8" PRId64 " %8" PRId64 "\n", desc, bmk_name, pos_name, start, end)
            f_out.write(f"{col1:<35} {chrom:<10} {col3:<35} {col4:>8} {col5:>8}\n")

def main():
    parser = argparse.ArgumentParser(description="Convert between Transmark .pos and BED6 formats.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")
    
    # pos2bed
    p2b = subparsers.add_parser("pos2bed", help="Convert .pos to .bed")
    p2b.add_argument("input_pos", help="Input .pos file path")
    p2b.add_argument("output_bed", help="Output .bed file path")
    
    # bed2pos
    b2p = subparsers.add_parser("bed2pos", help="Convert .bed to .pos")
    b2p.add_argument("input_bed", help="Input .bed file path")
    b2p.add_argument("output_pos", help="Output .pos file path")
    
    args = parser.parse_args()
    
    if args.command == "pos2bed":
        print(f"Converting {args.input_pos} to {args.output_bed}...")
        pos_to_bed(args.input_pos, args.output_bed)
    elif args.command == "bed2pos":
        print(f"Converting {args.input_bed} to {args.output_pos}...")
        bed_to_pos(args.input_bed, args.output_pos)
    print("Done!")

if __name__ == '__main__':
    main()

#! /usr/bin/perl
#
# Parse nail tabular output and convert to transmark DNA coordinates.
#
# Usage: perl transmark-nail-parse.pl <nail.tbl> [target_dna.fa]
#
# Nail format (10 columns):
# 0: target
# 1: query
# 2: target start (protein)
# 3: target end (protein)
# 4: query start
# 5: query end
# 6: score
# 7: bias
# 8: evalue
# 9: cell frac

use strict;
use warnings;

my $nail_tbl = shift;
my $target_fa = shift;

my %lengths = ();
if (defined $target_fa && -e $target_fa) {
    open(FA, $target_fa) or die "Could not open $target_fa: $!";
    my $current_id = "";
    while (<FA>) {
        if (/^>(\S+)/) {
            $current_id = $1;
            $lengths{$current_id} = 0;
        } else {
            chomp;
            $lengths{$current_id} += length($_);
        }
    }
    close(FA);
}

open(TBL, $nail_tbl) or die "Could not open $nail_tbl: $!";
while (<TBL>) {
    chomp;
    if (/^\#/ || /^\s*$/) { next; }
    my @fields = split(/\s+/, $_);
    
    if (scalar(@fields) < 10) { next; }

    my $target_full = $fields[0];
    my $raw_query   = $fields[1];
    my $p_start     = $fields[2];
    my $p_end       = $fields[3];
    my $bitscore    = $fields[6];
    my $evalue      = $fields[8];
    
    my $target      = $target_full;
    my $dna_from    = $p_start;
    my $dna_to      = $p_end;

    # Handle 6-frame translation suffixes (_F1 to _R3)
    if ($target_full =~ /^(.*)_([FR])([1-3])$/) {
        my $stem   = $1;
        my $strand = $2;
        my $frame  = $3; # 1, 2, or 3
        
        $target = $stem;
        my $L = $lengths{$target} || 0;

        if ($strand eq 'F') {
            # Forward mapping: AA pos p -> DNA 3*(p-1) + frame
            $dna_from = 3 * ($p_start - 1) + $frame;
            $dna_to   = 3 * ($p_end - 1) + $frame + 2;
        } else {
            # Reverse mapping
            # AA pos p (1-based) in RC frame f (1,2,3)
            # DNA from (1-based) = L - 3*(p-1) - f + 1
            # DNA to (1-based)   = L - 3*(p-1) - f - 1
            if ($L > 0) {
                $dna_from = $L - 3 * ($p_start - 1) - $frame + 1;
                $dna_to   = $L - 3 * ($p_end - 1) - $frame - 1;
            } else {
                # If we don't have L, we can't map reverse strand correctly.
                die "ERROR: Found reverse strand hit for $target but sequence length is unknown. Did you provide the correct DNA FASTA file?";
            }
        }
    }

    # Standard transmark query name handling (remove TRAIN. prefix)
    my @query_parts = split(/\./, $raw_query);
    my $query = defined($query_parts[1]) ? $query_parts[1] : $query_parts[0];

    # Format expected by transmark: evalue bitscore from to target query
    printf "%10g %10.1f %10d %10d %20s %35s\n", 
           $evalue, $bitscore, $dna_from, $dna_to, $target, $query;
}
close(TBL);

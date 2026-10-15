use strict;
use warnings;
use List::Util qw[min max];
my @hit;
my $i;
my $j;
my $k;

my $usage = "Usage: perl transmark-tpr.pl <merfile> <toolname>\n";

my $merfile = shift;
my $tool = shift;

my %num_targets = ();
my %tp_found = ();
my %fp_found = ();
my %tp_before_fp = ();
my @fields;
my $fam;
my $line;
my $type;

#get the total numner of target sequences (true positives) for each familiy
open(MER, $merfile) || die "ERROR, couldn't open $merfile";
while ($line = <MER>)
{
    chomp $line;
    if    ($line =~ m/^\#/) { next; }
    elsif ($line =~ m/^\=/) { next; }
    else  {
        @fields   = split(' ', $line, 6);
        $fam = $fields[0];
        if ($fam =~ m/^\*/) { next; }
        $num_targets{$fam}   = $fields[2]; 
        $tp_found{$fam}      = -1;
	$fp_found{$fam}      = -1;
        $tp_before_fp{$fam}  = -1;
    }
}
close(MER);

# count all the true positives found for each familiy
# Once the first false positive is found record the tp found up to that point
open(MER, $merfile) || die "ERROR, couldn't open $merfile";
while ($line = <MER>)
{
    chomp $line;
    if    ($line =~ m/^\=/) {  
        @fields   = split(' ', $line, 6);
	$fam = $fields[2];
	$type = $fields[5];
        
        #if this is a positive
	if ($type =~ m/\+/) {
            if ($tp_found{$fam} != -1) { $tp_found{$fam}++;   } 
            else                       { $tp_found{$fam} = 1; }
        #if this is a negative
        } else {
            if ($fp_found{$fam} != -1) { $fp_found{$fam}++;   }
            else                       { $fp_found{$fam} = 1; }

            if    ($tp_before_fp{$fam} != -1) { next; }
	    elsif ($tp_found{$fam}     != -1) { $tp_before_fp{$fam} = $tp_found{$fam}; }
            else                              { $tp_before_fp{$fam} = 0; }
        }
    }
}

close(MER);

printf("%10s %20s %10s %10s %16s\n", "# tool", "family", "total tp", "tp found", "tp before 1st fp");

foreach $fam (sort keys %num_targets) {
    if($fp_found{$fam} == -1) { $tp_before_fp{$fam} = $tp_found{$fam}; }

    printf("%10s %20s %10d ", $tool, $fam, $num_targets{$fam});

    if ($tp_found{$fam} != -1) { printf("%10d ", $tp_found{$fam}); }
    else                       { printf("%10d ", 0); }

    if ($tp_before_fp{$fam} != -1) { printf("%10d\n", $tp_before_fp{$fam}); }
    else                           { printf("%10d\n", 0); }
}


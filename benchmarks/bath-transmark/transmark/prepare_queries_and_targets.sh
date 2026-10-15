#!/bin/bash

if [ "$#" -ne 7 ]; then
	exit 1;
fi


echo "benchmark basename " $1
echo "benchmark directory " $2
echo "hmmer directory " $3
echo "bath directory " $4
echo "bath2 directory " $5
echo "dummer directory " $6
echo "cpus available " $7

#cd into benchmark directory
cd $2

#DUMMER
# make dummer query hmms
$3/hmmbuild $1.AA.hmm $1.AA.msa

#BATH
# make bath query hmms
#$4/bathbuild --cpu $7 $1.AA.hmm.filter.bhmm $1.AA.msa
$5/bathbuild --cpu $7 $1.AA.bhmm $1.AA.msa
#cp $1.AA.bhmm $1.AA.hmm.filter.bhmm

#NHMMER
# make hmmer DNA hmms
#$3/hmmbuild --cpu $7 $1.DNA.hmm $1.DNA.msa

# NOTE: MMseqs DBs (profileDB/targetDB_pad) are intentionally NOT prebuilt
# here. pipeline.py builds them on demand in a per-run tmpdir, and
# dummer --max skips them entirely.
rm -f $1.fa.fai

# Transmark

Transmark is a benchmark for testing the speed and accuracy of a series of sequence alignment tools in the annotation of protein-coding DNA with or without frameshift errors.

This directory includes:
1. The seed-files directory which includes zipped files containing all pre-made test and train sets
2. All code necessary to build a transmark benchmark
3. All code necessary to create the required query and target formats for the following sequence alignment software:
     1. bathsearch
     2. tblastn
     3. mmseqs
     4. lastal
     5. nhmmer
     6. diamond
4. Code to run and analyze the results from the above mentioned software
5. The badread-helpers directory, which contains the Transmark-specific helper scripts (vendored here rather than patched into Badread) used by step2-frameshift.sh

Optional frameshift-read simulation requires an upstream Badread clone (https://github.com/rrwick/Badread) for its badread-runner.py, plus the Python packages `badread` and `edlib`. Point the BADREAD_DIR environment variable at the clone if it is not in the default location.

This directory DOES NOT include the following:
1. A protein sequence dataset from which to create decoy sequences for the benchmark.  We used Pfam 38.1 seed alignments (Pfam-A.seed, 27,481 families, released 01/2026).  To reproduce this benchmark, download the versioned file:
   https://ftp.ebi.ac.uk/pub/databases/Pfam/releases/Pfam38.1/Pfam-A.seed.gz
   The release's md5_checksums file lists md5 ec8eb9b0d224f74f1d841672e5b7a1fc for Pfam-A.seed.gz; after gunzip the seed should be 592,631,194 bytes with sha256 2423041a02b9fcfb8f4afa50a2e3c4572a6223660a4339387041f516031f5156.
   Do not substitute a newer Pfam release: decoys are drawn from this file and silently change the benchmark results.
2. The actual sequence alignment software. Transmark is designed to run all sequence alignment tools from local copies of source code or binaries.  It accepts the full local paths to all executables so you do not need to have installed them via root access. The tools tested by transmark can be cloned or downloaded from the following locations. Be sure to follow directions to compile and/or make.
    1. BATH    - https://github.com/TravisWheelerLab/BATH
    2. BLAST   - https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/
    3. MMSEQS2 - https://github.com/soedinglab/MMseqs2
    4. LAST    - https://gitlab.com/mcfrith/last
    5. HMMER3  - https://github.com/EddyRivasLab/hmmer
    6. DIAMOND - https://github.com/bbuchfink/diamond

Steps to create a transmark benchmark
1. Clone the BATH-paper repository
2. Ensure you have a local copy of the HMMER3 repository - including EASEL - as well as the BLAST software (see above)
3. Ensure you have a dataset of protein sequences to use in the construction of decoys (see above)
4. cd into the transmark directory
5. Open the MAKEFILE and edit the ESLDIR and HMMERDIR variable to include the path to your local copies
6. run make
7. cd into the seed-files directory and unzip both the train-test.AA.tfile.zip and train-test.DNA.tfile.zip files
8. Run transmark-create.  To create a benchmark similar to the ones used in the BATH manuscript use the following set of command line arguments:
```
./transmark-create --pre -N 10 -L 100000000 -R 10 -E 10 --maxtrain 30 --maxtest 20 --maxfams 1500 -D <decoy file> <benchmark name> <DNA seed file> <AA seed file> <blast path> <easel path> <background hmm file> <frameshift rate>
```
The \<DNA seed file\>, \<AA seed file\>, and \<background hmm file\> are all files included in this directory.
The \<decoy file\>, \<blast path\>, and \<easel path\> are the paths to your local copies of the executables or data sets
The \<benchmark name\> is whatever you want the base names of the output files to be
The \<frameshift rate\> is your desired rate of simulated frameshifts in the embedded test sequences of this benchmark

For example:
```
./transmark-create --pre -N 10 -L 100000000 -R 10 -E 10 --maxtrain 30 --maxtest 20 --maxfams 1500 -D ~/Pfam38.1/Pfam-A.seed tanskmark00 seed-files/train-test.DNA.tfile seed-files/train-test.AA.tfile  ~/ncbi-blast-2.14.1+/bin/ ~/hmmer/easel/  0.0
```
To create an overextension benchmark like the ones used in the BATH manuscript (which embeds only the middle 50% of each test sequence) simply add the --over flag. 

Steps to run a transmark benchmark
1. Ensure you have local copies of all 6 tested software packages (see above)
2. Build a transmark benchmark (see above)
3. Run prepare_queries_and_targets.sh to create all the necessary file formats
```
bash prepare_queries_and_targets.sh <benchmark name> <benchmark directory> <blast directory> <last directory> <mmseqs directory> <hmmer directory> <bath directory> <diamond directory> <cpus available>
```
For example:
```
bash prepare_queries_and_targets.sh  tanskmark00 ~/BATH-paper/transmark/ ~/ncbi-blast-2.14.1+/bin/ ~/last/bin/ ~/MMseqs2/build/src ~/hmmer/src/ ~/BATH/src ~/diamond 16
```
The /<cpus available/> is what it sounds like - the number of CPUs available for paralyzing on your system. 

4. You should now have all the files you need to run the tools on your benchmark. To do this you will run each of the files starting with "x-" followed by the toolname. Each of these files takes the same set of command line arguments (with the exception of x-last):
```
x-toolname <tool directory> <benchmark directory> <results directory> <results filename>  <queryfile> <posfile> <targetfile> <cpus> <e-max>
```
For example:
```
x-bathsearch ~/BATH/src ~/BATH-paper/transmark/ ~/transmark00-results/BATH/ bath-results  transmark00-AA.bhmm transamark00.pos transmark00.fa 16 100
```

To run x-last you need one additional argument at the end of the command giving the average length of your target sequences. This can be determined by running esl-seqstat on the target file.

5. Once you have run a tool on a transmark benchmark you can analyze the results by running transmark-pp.sh with the following command line arguments.
```
bash transmark-pp.sh <benchmark directory> <results filename> <results directory> <analysis filename> <toolname> 
```
For example:
```
bash transmark-pp.sh ~/BATH-paper/transmark bath-results ~/transmark00-results/BATH/ bath-analysis bathsearch
``` 
This will produce the following files:
	1. bath-analysis.xy  - contains the x and y coordinates for creating ROC plots
	2. bath-analysis.mer - contains important per hit and per family information
        3. bath-analysis.tpr - contains the per-family data needed to make RBFP ridgeline plots
	4. bath-analysis.cover - contains coverage and overextension data for all true positives
        5. bath-analysis.time  - contains run time information

Steps to run the DUMMEST benchmark (step scripts)
The step*.sh scripts wrap the whole DUMMEST pipeline (benchmark creation,
Badread frameshift reads, query HMMs, dummest, analysis and figures) and
read all paths/settings from config.sh. They run dummest via x-last and
BATH via x-bathsearch; step5-last.sh runs LAST via the restored x-lastal
wrapper.

1. Unzip the seed files. The *.tfile.zip files are tracked; the unpacked
   *.tfile files are generated and git-ignored:
```
cd seed-files
unzip train-test.DNA.tfile.zip
unzip train-test.AA.tfile.zip
cd ..
```
2. Set paths/config in config.sh (or via environment variables):
   DUMMER_DIR (seq-position-probs/bin containing dummest), PREP_HMMER_DIR,
   PREP_BATH1_DIR, PREP_BATH2_DIR, BATHSEARCH_DIR, BADREAD_DIR, LAST_DIR.
   CREATE_PFAM_SEED must point at the Pfam 38.1 Pfam-A.seed (see above);
   step1 reuses existing DBs unless --force, so this matters only when
   rebuilding, and a different release changes the decoys and results.
   seqkit and mmseqs must be on PATH. Frameshift simulation additionally
   needs the upstream Badread clone plus the badread/edlib packages.
3. Run the full pipeline. By default it benchmarks the frameshifted reads
   for every variant, writing timed logs to step*-*.log:
```
./run-steps.sh                  # full lite full-over lite-over
./run-steps.sh full             # a single variant
./run-steps.sh --clean full     # pristine step-1 reads (skip Badread)
./run-steps.sh --force lite     # rebuild existing DBs
```
4. Or run the steps individually, in this order:
```
./step1-create.sh full              # transmark-create DB (reused unless --force)
./step2-frameshift.sh full          # Badread reads -> <base>-fs.fa/.pos
./step2-prepare.sh full             # query HMMs -> <base>.AA.hmm/.AA.bhmm
./step-2.5-rename-fs.sh full        # unique ids in <base>-fs.pos (+ rename.tsv)
./step3-bathsearch.sh full          # BATH -> <base>-fs-results*/bath-fs-analysis.*
./step3-bathsearch.sh --no-fs full  # BATH control without frameshift HMM
./step3-dummer-precompute.sh full   # 6-frame fa + MMseqs targetDB (full/full-over only)
./step4-dummer.sh full              # DUMMEST -> <base>-fs-results-v2*/dummer-fs-analysis.*
./step4-dummer.sh full --gapped-pvalue 0.002   # more sensitive run (suffix -p002)
./step-last-generate-figures.sh full  # ROC/e-value plots + coverage tables
```
5. Outputs are the per-run results directories transmark-<base>[-fs]-results*/
   holding *.xy, *.mer, *.tpr, *.cover and *.time; step4 reuses the
   step3-precomputed 6-frame FASTA and MMseqs DB when present and builds
   them in a per-run tmpdir otherwise. All generated files (data, results,
   logs, scratch) are git-ignored.
 

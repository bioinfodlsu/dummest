# config.example.sh — template for local settings.
#
# Copy to config.sh and edit. config.sh is gitignored. Every value is
# optional; leave one unset to fall back to the script default.
#
# The wrapper scripts source config.sh (override the path with CONFIG=...)
# and export these variables, which the Python runners read directly.

# --- sequence inputs ---
export READS="/path/to/reads.fastq"            # long-read FASTQ/FASTA
export GENOME="${REPO_DIR}/GCF_000911955.1_ViralProj215788_genomic.fna"
# export PROTEINS="/path/to/proteins.faa"      # protein input (skips translation)

# --- tool locations ---
export DUMMER_REPO="/path/to/seq-position-probs"
# export DUMMER_ENGINE="/path/to/seq-position-probs/bin/dummestl_core"
export BATH_BIN="/path/to/BATH/bin"
# export MMSEQS_BIN="/path/to/mmseqs"

# --- database paths (default: repo-relative ncbi_hmms/) ---
# export HMM_DB="${REPO_DIR}/ncbi_hmms/ncbi_amr_families.hmm"
# export BHMM="${REPO_DIR}/ncbi_hmms/ncbi_amr_families.bhmm"
# export MSA="${REPO_DIR}/ncbi_hmms/ncbi_amr_families.msa"

# --- run parameters ---
export N_READS=10000
export CPU=8
export SEARCH_EVALUE=1
export EVALUE=0.00001
export BASE="/tmp/comparison_ncbi"

# --- build parameters (prepare_ncbi_hmms.sh) ---
export MIN_SEQS=10
# export OUTDIR="ncbi_hmms"
# export SRC_DIR="./ncbi_source"

#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --partition=standard
#SBATCH --account=twheeler
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2


#stop the script if it attempts to use any unset variables 
set -o nounset
#stop the script if any command fails (returns non zero)
set -o errexit

#debug the script
set -x


if [ $# -lt 5 ]; then
    echo "ERROR: Your command line contains less than five arguments ($scriptdir, $inbase, $resultdir, $outbase, $toolname),"
    exit 1
fi

sort_option='-g'

echo "script directory", $1
echo "input basename", $2
echo "result directory", $3
echo "output basename ", $4
echo "tool name", $5

# get running time
ls $3/*.time   | perl $1/transmark-time.pl > $3/$4.time

# get MER
cat $3/*out | sort ${sort_option} | perl $1/transmark-mer.pl $2.pos $3/$4.time >  $3/$4.mer

# get coverage
cat $3/*out | sort ${sort_option} | perl $1/transmark-coverage.pl $2.pos $5 > $3/$4.cover

#get tpr
perl $1/transmark-tpr.pl $3/$4.mer $5 > $3/$4.tpr

#get roc (ROCPLOT_SEED pins the bootstrap RNG for reproducible .xy; see config.sh)
echo "# $5" > $3/$4.xy
cat $3/*.out | sort -g | $1/transmark-rocplot --max 1000 --seed ${ROCPLOT_SEED:-1} $2 - >> $3/$4.xy
#perl $1/transmark-roc.pl $3/$4.mer $5 > $3/$4.roc 

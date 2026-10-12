#!/bin/bash
cd "$(dirname "$0")/.."
time python3 bin/dummest samples/MET-test.hmm samples/MET.msa samples/MET-target.fa 16 --batch 1

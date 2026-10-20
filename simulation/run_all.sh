#!/bin/bash
# Runs the simulation in two batches (replicates 1-200, then 201-500) from code_revision/.
# Usage: bash simulation/run_all.sh <n_cores>
cd "$(dirname "$0")/.."
N=${1:-14}
Rscript simulation/run_simulation.R 1 200 "$N" >> simulation/results/batch1_log.txt 2>&1
echo "$(date '+%F %T') batch 1 finished, exit $?" >> simulation/results/progress.log
Rscript simulation/run_simulation.R 201 500 "$N" >> simulation/results/batch2_log.txt 2>&1
echo "$(date '+%F %T') batch 2 finished, exit $?" >> simulation/results/progress.log

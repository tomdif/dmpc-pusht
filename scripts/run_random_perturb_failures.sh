#!/usr/bin/env bash
# Random-perturb fallback on the 39 seeds that failed n=500 K=5+disc.
# Distributes seeds across 6 GPUs, K=10 random_perturb attempts per seed.
set -euo pipefail
ROOT=/workspace/cser-jepa-v2
LOGDIR=/root/step5/logs/n500_perturb
mkdir -p $LOGDIR
cd $ROOT
source .venv/bin/activate

# 39 failed seeds, sorted ascending. Round-robin across GPUs.
FAILED=(1 12 23 33 36 63 77 79 87 102 105 141 154 186 191 194 197 202 210 214 229 249 266 288 295 297 299 300 340 360 370 392 401 418 432 436 470 471 475)
N=${#FAILED[@]}
NGPU=6

for g in $(seq 0 $((NGPU - 1))); do
    SEEDS=()
    for i in $(seq 0 $((N - 1))); do
        if [ $((i % NGPU)) -eq $g ]; then
            SEEDS+=(${FAILED[$i]})
        fi
    done
    LIST=$(IFS=,; echo "${SEEDS[*]}")
    echo "[gpu$g] seeds: $LIST"
    CUDA_VISIBLE_DEVICES=$g python -u scripts/eval_seed102_strategies.py \
        --seeds-list $LIST --strategy random_perturb --n-attempts 10 \
        > $LOGDIR/perturb_gpu${g}.log 2>&1 &
done
wait
echo "=== ALL DONE ==="

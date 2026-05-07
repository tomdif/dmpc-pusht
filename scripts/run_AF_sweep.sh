#!/usr/bin/env bash
# AF proximity-weight sweep + held-out comparison.
# Tests {0 (no-proximity control), 0.001, 0.003, 0.01} × {train, holdout} seeds.

set -euo pipefail
ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_AF_sweep
mkdir -p $LOGDIR

CKPT=$ROOT/ckpts/stage_AF/reward/ckpt_step5000.pt
BC=$ROOT/ckpts/stage_Z/bc_policy_Z.pt

run_one() {
    local name=$1 prox=$2 use_prox=$3 seed_base=$4

    echo "=== AF prox=$prox seed_base=$seed_base ===" | tee -a $LOGDIR/sweep.log
    python -u scripts/stage1b_online_eval.py \
        --config configs/stage1af_reward.yaml \
        --ckpt $CKPT --bc-ckpt $BC \
        --device cuda --n-episodes 10 --max-steps 200 \
        --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 \
        --cem-init-std 0.3 --replan-every 2 --seed $seed_base \
        --use-value --value-weight 1.0 \
        $use_prox \
        > $LOGDIR/${name}.log 2>&1
    grep -E "summary|mean return|mean max_cov|success rate" $LOGDIR/${name}.log | tee -a $LOGDIR/sweep.log
    echo "" | tee -a $LOGDIR/sweep.log
}

# Training seeds (0-9): no-proximity control + 3 proximity weights
run_one train_p0    0     ""                                      0
run_one train_p001  0.001 "--use-proximity --proximity-weight 0.001" 0
run_one train_p003  0.003 "--use-proximity --proximity-weight 0.003" 0
run_one train_p010  0.010 "--use-proximity --proximity-weight 0.010" 0

# Held-out seeds (100-109): only the best-from-train re-tested. We do all
# four to avoid extra round trips; each only takes ~5 min.
run_one ho_p0       0     ""                                      100
run_one ho_p001     0.001 "--use-proximity --proximity-weight 0.001" 100
run_one ho_p003     0.003 "--use-proximity --proximity-weight 0.003" 100
run_one ho_p010     0.010 "--use-proximity --proximity-weight 0.010" 100

echo "=== SWEEP DONE ===" | tee -a $LOGDIR/sweep.log

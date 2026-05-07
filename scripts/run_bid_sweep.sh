#!/usr/bin/env bash
# BID weight sweep: test {0.05, 0.1, 0.3, 0.5} on D-MPC at n=10 quick screen.
set -euo pipefail
ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_bid_sweep
mkdir -p $LOGDIR
cd $ROOT
source .venv/bin/activate

CKPT=$ROOT/ckpts/stage_AB/reward/ckpt_step5000.pt
GOAL=$ROOT/goals/pusht_goal_lerobot.pt

run_one() {
    local w=$1
    echo "=== bid_weight=$w (n=10, seed=100) ===" | tee -a $LOGDIR/sweep.log
    date | tee -a $LOGDIR/sweep.log
    python -u scripts/eval_dmpc.py \
        --world-config configs/stage1ab_reward.yaml \
        --world-ckpt $CKPT \
        --device cuda --n-episodes 10 --max-steps 300 \
        --n-samples 64 --n-action-steps 8 --seed 100 \
        --use-value --value-weight 1.0 \
        --use-goal --goal-weight 0.3 --goal-aggregate min \
        --goal-multi --goal-file $GOAL \
        --rerank-mode dmpc \
        --use-bid --bid-weight $w --bid-rho 0.7 \
        > $LOGDIR/bid_w${w}.log 2>&1
    grep -E "summary|mean return|mean max_cov|success rate" $LOGDIR/bid_w${w}.log \
        | tee -a $LOGDIR/sweep.log
    echo "" | tee -a $LOGDIR/sweep.log
}

run_one 0.05
run_one 0.1
run_one 0.3
run_one 0.5

echo "=== SWEEP DONE ===" | tee -a $LOGDIR/sweep.log

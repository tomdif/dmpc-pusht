#!/usr/bin/env bash
# Goal-weight sweep on AB held-out. Tests {0.3, 0.5, 3.0, 10.0} (1.0 baseline
# was +15.95). All same horizon=12, init_std=0.3, replan_every=2,
# use-value, goal_aggregate=min.

set -euo pipefail
ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_goalweight_sweep
mkdir -p $LOGDIR

CKPT=$ROOT/ckpts/stage_AB/reward/ckpt_step5000.pt
BC=$ROOT/ckpts/stage_AB/bc_policy_AB.pt
GOAL=$ROOT/goals/pusht_goal.pt

run_one() {
    local gw=$1
    echo "=== goal_weight=$gw held-out ===" | tee -a $LOGDIR/sweep.log
    python -u scripts/stage1b_online_eval.py \
        --config configs/stage1ab_reward.yaml \
        --ckpt $CKPT --bc-ckpt $BC \
        --device cuda --n-episodes 10 --max-steps 200 \
        --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 \
        --cem-init-std 0.3 --replan-every 2 --seed 100 \
        --use-value --value-weight 1.0 \
        --use-goal --goal-weight $gw --goal-aggregate min \
        --goal-file $GOAL \
        > $LOGDIR/gw_${gw}.log 2>&1
    grep -E "summary|mean return|mean max_cov|success rate" $LOGDIR/gw_${gw}.log | tee -a $LOGDIR/sweep.log
    echo "" | tee -a $LOGDIR/sweep.log
}

run_one 0.3
run_one 0.5
run_one 3.0
run_one 10.0

echo "=== SWEEP DONE ===" | tee -a $LOGDIR/sweep.log

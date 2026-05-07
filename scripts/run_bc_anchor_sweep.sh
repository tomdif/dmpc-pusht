#!/usr/bin/env bash
# 30-iter CEM + BC-anchor sweep on AB ckpt at gw=0.3 (calibrated peak), seed=100.
#
# Hypothesis: BC-anchor penalty -w * ||a_first - a_BC||² dampens world-model
# exploitation under high-iter CEM. Tests whether 30-iter CEM with anchor
# beats the 4-iter baseline (+18.75 mean) without the bimodal "0.000 episode"
# failure mode seen in 30-iter no-anchor run.
set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_bc_anchor_sweep
mkdir -p $LOGDIR

cd $ROOT
source .venv/bin/activate

CKPT=$ROOT/ckpts/stage_AB/reward/ckpt_step5000.pt
BC=$ROOT/ckpts/stage_AB/bc_policy_AB.pt
GOAL=$ROOT/goals/pusht_goal.pt

run_one() {
    local w=$1
    echo "=== bc_anchor_weight=$w (30-iter, gw=0.3, seed=100) ===" | tee -a $LOGDIR/sweep.log
    date | tee -a $LOGDIR/sweep.log
    python -u scripts/stage1b_online_eval.py \
        --config configs/stage1ab_reward.yaml \
        --ckpt $CKPT --bc-ckpt $BC \
        --device cuda --n-episodes 10 --max-steps 200 \
        --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 30 \
        --cem-init-std 0.3 --replan-every 2 --seed 100 \
        --use-value --value-weight 1.0 \
        --use-goal --goal-weight 0.3 --goal-aggregate min \
        --goal-file $GOAL \
        --use-bc-anchor --bc-anchor-weight $w \
        > $LOGDIR/anchor_w${w}.log 2>&1
    grep -E "summary|mean return|mean max_cov|success rate" $LOGDIR/anchor_w${w}.log \
        | tee -a $LOGDIR/sweep.log
    echo "" | tee -a $LOGDIR/sweep.log
}

run_one 0.1
run_one 1.0
run_one 10.0

echo "=== SWEEP DONE ===" | tee -a $LOGDIR/sweep.log

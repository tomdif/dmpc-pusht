#!/usr/bin/env bash
# Pixel-grounded planning sweep on AB ckpt at gw=0.3, 4-iter CEM (fast).
# Tests pixel_ground_weight ∈ {1, 10, 100} vs +18.75 no-pixel baseline.
#
# If pixel grounding helps, expect lift over +18.75.
set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_pixel_ground_sweep
mkdir -p $LOGDIR

cd $ROOT
source .venv/bin/activate

CKPT=$ROOT/ckpts/stage_AB/reward/ckpt_step5000.pt
BC=$ROOT/ckpts/stage_AB/bc_policy_AB.pt
GOAL=$ROOT/goals/pusht_goal.pt
DEC=$ROOT/ckpts/decoder/decoder_AB.pt

run_one() {
    local w=$1
    echo "=== pixel_ground_weight=$w (4-iter, gw=0.3, seed=100) ===" | tee -a $LOGDIR/sweep.log
    date | tee -a $LOGDIR/sweep.log
    python -u scripts/stage1b_online_eval.py \
        --config configs/stage1ab_reward.yaml \
        --ckpt $CKPT --bc-ckpt $BC \
        --device cuda --n-episodes 10 --max-steps 200 \
        --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 \
        --cem-init-std 0.3 --replan-every 2 --seed 100 \
        --use-value --value-weight 1.0 \
        --use-goal --goal-weight 0.3 --goal-aggregate min \
        --goal-file $GOAL \
        --use-pixel-ground --pixel-ground-weight $w --pixel-ground-aggregate min \
        --decoder-ckpt $DEC \
        > $LOGDIR/pix_w${w}.log 2>&1
    grep -E "summary|mean return|mean max_cov|success rate" $LOGDIR/pix_w${w}.log \
        | tee -a $LOGDIR/sweep.log
    echo "" | tee -a $LOGDIR/sweep.log
}

run_one 1.0
run_one 10.0
run_one 100.0

echo "=== SWEEP DONE ===" | tee -a $LOGDIR/sweep.log

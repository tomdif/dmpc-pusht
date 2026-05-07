#!/usr/bin/env bash
# Phase 1 eval — SIGReg-only ckpt, 30-iter CEM, goal-cond gw=0.3 (calibrated peak),
# both training and held-out seeds. Tests whether dropping IDM-z + reward/value heads
# during training (LeWM-recipe transplant) matches or beats AB.
set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_phase1
GOAL=$ROOT/goals/pusht_goal.pt
CKPT=$ROOT/ckpts/stage_phase1/ckpt_step16000.pt
mkdir -p $LOGDIR

# Phase 1 has no value head trained — disable --use-value (head exists but
# never saw gradient).
COMMON="--config configs/stage1_phase1_sigreg.yaml \
    --ckpt $CKPT \
    --device cuda --n-episodes 10 --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 30 \
    --cem-init-std 0.3 --replan-every 2 \
    --use-goal --goal-weight 0.3 --goal-aggregate min \
    --goal-file $GOAL"

echo "=== Phase 1 eval: training seeds (no BC, no value, gw=0.3, CEM iters=30) ===" \
    | tee -a $LOGDIR/eval.log
python -u scripts/stage1b_online_eval.py $COMMON --seed 0 \
    > $LOGDIR/phase1_train.log 2>&1
grep -E "summary|mean return|mean max_cov|success rate" $LOGDIR/phase1_train.log \
    | tee -a $LOGDIR/eval.log
echo "" | tee -a $LOGDIR/eval.log

echo "=== Phase 1 eval: held-out seeds ===" | tee -a $LOGDIR/eval.log
python -u scripts/stage1b_online_eval.py $COMMON --seed 100 \
    > $LOGDIR/phase1_holdout.log 2>&1
grep -E "summary|mean return|mean max_cov|success rate" $LOGDIR/phase1_holdout.log \
    | tee -a $LOGDIR/eval.log
echo "=== PHASE 1 EVAL DONE ===" | tee -a $LOGDIR/eval.log

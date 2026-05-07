#!/usr/bin/env bash
# Self-GAD scale sweep: test {0.1, 0.2, 1.0, 2.0} guidance scales on D-MPC.
# Each at n=10 for cheap signal; pick best for full n=50 confirmation.
set -euo pipefail
ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_selfgad_sweep
mkdir -p $LOGDIR
cd $ROOT
source .venv/bin/activate

CKPT=$ROOT/ckpts/stage_AB/reward/ckpt_step5000.pt
BC=$ROOT/ckpts/stage_AB/bc_policy_AB.pt
GOAL=$ROOT/goals/pusht_goal_lerobot.pt

run_one() {
    local s=$1
    echo "=== gad_scale=$s (n=10, seed=100) ===" | tee -a $LOGDIR/sweep.log
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
        --self-gad --gad-scale $s --gad-anchor-len 4 \
        > $LOGDIR/gad_${s}.log 2>&1
    grep -E "summary|mean return|mean max_cov|success rate" $LOGDIR/gad_${s}.log \
        | tee -a $LOGDIR/sweep.log
    echo "" | tee -a $LOGDIR/sweep.log
}

run_one 0.1
run_one 0.2
run_one 1.0
run_one 2.0

echo "=== SWEEP DONE ===" | tee -a $LOGDIR/sweep.log

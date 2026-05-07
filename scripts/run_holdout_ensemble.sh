#!/usr/bin/env bash
# Run all 5 specialists on held-out seeds 100-109, then oracle-aggregate.

set -euo pipefail
ROOT=/workspace/cser-jepa-v2

run_eval() {
    local name=$1
    local cfg=$2
    local ckpt=$3
    local bc=$4
    local std=$5
    local extra=$6

    echo "=== $name on held-out seeds 100-109 ===" | tee -a $ROOT/runs_holdout.log
    python -u scripts/stage1b_online_eval.py \
        --config $cfg \
        --ckpt $ckpt --bc-ckpt $bc \
        --device cuda --n-episodes 10 --max-steps 200 \
        --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 \
        --cem-init-std $std --replan-every 2 --seed 100 \
        $extra > $ROOT/runs_holdout_${name}.log 2>&1
    echo "[$name] done" | tee -a $ROOT/runs_holdout.log
}

run_eval T2 configs/stage1b_reward_M.yaml \
    $ROOT/ckpts/stage_M/reward/ckpt_step10000.pt \
    $ROOT/ckpts/stage_M/bc_policy_M.pt 0.3 ""

run_eval U configs/stage1u_reward.yaml \
    $ROOT/ckpts/stage_U/reward/ckpt_step5000.pt \
    $ROOT/ckpts/stage_U/bc_policy_U.pt 0.3 ""

run_eval W configs/stage1w_reward.yaml \
    $ROOT/ckpts/stage_W/reward/ckpt_step5000.pt \
    $ROOT/ckpts/stage_W/bc_policy_W.pt 0.3 ""

run_eval Z configs/stage1z_reward.yaml \
    $ROOT/ckpts/stage_Z/reward/ckpt_step5000.pt \
    $ROOT/ckpts/stage_Z/bc_policy_Z.pt 0.3 "--use-value"

run_eval AB configs/stage1ab_reward.yaml \
    $ROOT/ckpts/stage_AB/reward/ckpt_step5000.pt \
    $ROOT/ckpts/stage_AB/bc_policy_AB.pt 0.3 "--use-value"

echo "=== ALL HOLD-OUT EVALS DONE ===" | tee -a $ROOT/runs_holdout.log

#!/usr/bin/env bash
# Stage M downstream from step-10000 pretrain ckpt:
#   reward fine-tune (10K) -> BC retrain (8K) -> online eval.

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_M
CKPTROOT=$ROOT/ckpts/stage_M
PRETRAIN_CKPT=$CKPTROOT/pretrain/ckpt_step10000.pt

mkdir -p $CKPTROOT/reward

echo "=== [chain-from10k] Step 1/3: 10K reward fine-tune ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/stage1_pretrain.py \
    --config configs/stage1b_reward_M.yaml \
    --device cuda --steps 10000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $PRETRAIN_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step10000.pt
echo "[chain-from10k] reward done at $(date), ckpt=$REWARD_CKPT" | tee -a $LOGDIR/chain.log

echo "=== [chain-from10k] Step 2/3: 8K BC retrain ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1c_bc_train.py \
    --config configs/stage1b_reward_M.yaml \
    --ckpt $REWARD_CKPT \
    --out $CKPTROOT/bc_policy_M.pt \
    --device cuda \
    --steps 8000 \
    --log-every 200 \
    --num-workers 4 \
    > $LOGDIR/bc.log 2>&1
echo "[chain-from10k] BC done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [chain-from10k] Step 3/3: online eval ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1b_reward_M.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $CKPTROOT/bc_policy_M.pt \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 8 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.5 \
    > $LOGDIR/eval.log 2>&1
echo "[chain-from10k] eval done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log
echo "  reward ckpt : $REWARD_CKPT" | tee -a $LOGDIR/chain.log
echo "  bc ckpt     : $CKPTROOT/bc_policy_M.pt" | tee -a $LOGDIR/chain.log
echo "  eval log    : $LOGDIR/eval.log" | tee -a $LOGDIR/chain.log

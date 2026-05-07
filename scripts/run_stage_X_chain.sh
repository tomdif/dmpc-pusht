#!/usr/bin/env bash
# Stage X chain — 155M (ViT-Base) on W's 4-way combined data:
#   pretrain (16K) -> reward (5K) -> BC (8K) -> eval

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_X
CKPTROOT=$ROOT/ckpts/stage_X

mkdir -p $LOGDIR $CKPTROOT/pretrain $CKPTROOT/reward

echo "=== [X] Step 1/4: 16K combined-data pretrain (4-way, 155M) ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1x_combined.yaml \
    --device cuda --steps 16000 \
    --log-every 200 --diag-every 2000 \
    --num-workers 4 \
    --ckpt-every 4000 \
    --ckpt-dir $CKPTROOT/pretrain \
    > $LOGDIR/pretrain.log 2>&1
PRETRAIN_CKPT=$CKPTROOT/pretrain/ckpt_step16000.pt
echo "[X] pretrain done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [X] Step 2/4: 5K reward fine-tune ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1x_reward.yaml \
    --device cuda --steps 5000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $PRETRAIN_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step5000.pt
echo "[X] reward done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [X] Step 3/4: 8K BC retrain ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1c_bc_train.py \
    --config configs/stage1x_reward.yaml \
    --ckpt $REWARD_CKPT \
    --out $CKPTROOT/bc_policy_X.pt \
    --device cuda \
    --steps 8000 \
    --log-every 200 \
    --num-workers 4 \
    > $LOGDIR/bc.log 2>&1
echo "[X] BC done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [X] Step 4/4: online eval (T2 hyperparams) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1x_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $CKPTROOT/bc_policy_X.pt \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.3 \
    --replan-every 2 \
    > $LOGDIR/eval.log 2>&1
echo "[X] eval done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log

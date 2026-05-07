#!/usr/bin/env bash
# Stage V chain:
#   1. Collect self-play rollouts from S2-best ckpt
#   2. Pretrain on combined data (10K)
#   3. Reward fine-tune (5K)
#   4. BC retrain on combined data's encoder (8K)
#   5. Online eval

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_V
CKPTROOT=$ROOT/ckpts/stage_V
ROLLDIR=$ROOT/rollouts/self_play

S2_REWARD_CKPT=$ROOT/ckpts/stage_M/reward/ckpt_step10000.pt
S2_BC_CKPT=$ROOT/ckpts/stage_M/bc_policy_M.pt

mkdir -p $LOGDIR $CKPTROOT/pretrain $CKPTROOT/reward $ROLLDIR

echo "=== [V] Step 1/5: collect self-play rollouts ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/collect_self_play.py \
    --config configs/stage1b_reward_M.yaml \
    --ckpt $S2_REWARD_CKPT \
    --bc-ckpt $S2_BC_CKPT \
    --out $ROLLDIR \
    --device cuda \
    --n-episodes 400 \
    --max-steps 200 \
    --noise-std 0.4 \
    --random-frac 0.25 \
    --min-max-coverage 0.02 \
    > $LOGDIR/collect.log 2>&1
echo "[V] collect done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [V] Step 2/5: 10K combined-data pretrain ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1v_combined.yaml \
    --device cuda --steps 10000 \
    --log-every 200 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/pretrain \
    > $LOGDIR/pretrain.log 2>&1
PRETRAIN_CKPT=$CKPTROOT/pretrain/ckpt_step10000.pt
echo "[V] pretrain done at $(date), ckpt=$PRETRAIN_CKPT" | tee -a $LOGDIR/chain.log

echo "=== [V] Step 3/5: 5K reward fine-tune ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1v_reward.yaml \
    --device cuda --steps 5000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $PRETRAIN_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step5000.pt
echo "[V] reward done at $(date), ckpt=$REWARD_CKPT" | tee -a $LOGDIR/chain.log

echo "=== [V] Step 4/5: 8K BC retrain ===" | tee -a $LOGDIR/chain.log
# BC trainer reads single-dataset config — point it at the canonical pusht
# config and load the V-stage encoder. BC is trained on the canonical
# expert-action distribution, NOT on self-play noisy actions.
python -u scripts/stage1c_bc_train.py \
    --config configs/stage1b_reward_M.yaml \
    --ckpt $REWARD_CKPT \
    --out $CKPTROOT/bc_policy_V.pt \
    --device cuda \
    --steps 8000 \
    --log-every 200 \
    --num-workers 4 \
    > $LOGDIR/bc.log 2>&1
echo "[V] BC done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [V] Step 5/5: online eval ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1b_reward_M.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $CKPTROOT/bc_policy_V.pt \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 8 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.5 \
    --replan-every 2 \
    > $LOGDIR/eval.log 2>&1
echo "[V] eval done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log
echo "  reward ckpt : $REWARD_CKPT" | tee -a $LOGDIR/chain.log
echo "  bc ckpt     : $CKPTROOT/bc_policy_V.pt" | tee -a $LOGDIR/chain.log
echo "  eval log    : $LOGDIR/eval.log" | tee -a $LOGDIR/chain.log

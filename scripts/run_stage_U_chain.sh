#!/usr/bin/env bash
# Stage U chain — self-play iteration 2 with T policy:
#   collect 600 episodes with T-best setup -> retrain 80M on 3-way data
#   (LeRobot + V-self-play + T-self-play) -> eval with T2 hyperparams.

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_U
CKPTROOT=$ROOT/ckpts/stage_U
ROLLDIR_NEW=$ROOT/rollouts/self_play_t

T_REWARD_CKPT=$ROOT/ckpts/stage_T/reward/ckpt_step5000.pt
T_BC_CKPT=$ROOT/ckpts/stage_T/bc_policy_T.pt

mkdir -p $LOGDIR $CKPTROOT/pretrain $CKPTROOT/reward $ROLLDIR_NEW

echo "=== [U] Step 1/5: collect 600 self-play with T2-best policy ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/collect_self_play.py \
    --config configs/stage1t_reward.yaml \
    --ckpt $T_REWARD_CKPT \
    --bc-ckpt $T_BC_CKPT \
    --out $ROLLDIR_NEW \
    --device cuda \
    --n-episodes 600 \
    --max-steps 200 \
    --noise-std 0.25 \
    --random-frac 0.05 \
    --min-max-coverage 0.02 \
    --seed-base 90000 \
    > $LOGDIR/collect.log 2>&1
echo "[U] collect done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [U] Step 2/5: 14K combined-data pretrain (3-way, 80M) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1u_combined.yaml \
    --device cuda --steps 14000 \
    --log-every 200 --diag-every 1500 \
    --num-workers 4 \
    --ckpt-every 3500 \
    --ckpt-dir $CKPTROOT/pretrain \
    > $LOGDIR/pretrain.log 2>&1
PRETRAIN_CKPT=$CKPTROOT/pretrain/ckpt_step14000.pt
echo "[U] pretrain done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [U] Step 3/5: 5K reward fine-tune ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1u_reward.yaml \
    --device cuda --steps 5000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $PRETRAIN_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step5000.pt
echo "[U] reward done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [U] Step 4/5: 8K BC retrain ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1c_bc_train.py \
    --config configs/stage1u_reward.yaml \
    --ckpt $REWARD_CKPT \
    --out $CKPTROOT/bc_policy_U.pt \
    --device cuda \
    --steps 8000 \
    --log-every 200 \
    --num-workers 4 \
    > $LOGDIR/bc.log 2>&1
echo "[U] BC done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [U] Step 5/5: online eval with T2 hyperparams ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1u_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $CKPTROOT/bc_policy_U.pt \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.3 \
    --replan-every 2 \
    > $LOGDIR/eval.log 2>&1
echo "[U] eval done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log

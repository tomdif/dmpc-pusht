#!/usr/bin/env bash
# Stage W chain — self-play iteration 3 with U policy:
#   collect 700 episodes with U-best setup -> retrain 80M on 4-way data
#   (LeRobot + V-self-play + T-self-play + U-self-play) -> eval.

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_W
CKPTROOT=$ROOT/ckpts/stage_W
ROLLDIR_NEW=$ROOT/rollouts/self_play_u

U_REWARD_CKPT=$ROOT/ckpts/stage_U/reward/ckpt_step5000.pt
U_BC_CKPT=$ROOT/ckpts/stage_U/bc_policy_U.pt

mkdir -p $LOGDIR $CKPTROOT/pretrain $CKPTROOT/reward $ROLLDIR_NEW

echo "=== [W] Step 1/5: collect 700 self-play with U-best policy ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/collect_self_play.py \
    --config configs/stage1u_reward.yaml \
    --ckpt $U_REWARD_CKPT \
    --bc-ckpt $U_BC_CKPT \
    --out $ROLLDIR_NEW \
    --device cuda \
    --n-episodes 700 \
    --max-steps 200 \
    --noise-std 0.2 \
    --random-frac 0.05 \
    --min-max-coverage 0.02 \
    --seed-base 130000 \
    > $LOGDIR/collect.log 2>&1
echo "[W] collect done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [W] Step 2/5: 16K combined-data pretrain (4-way, 80M) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1w_combined.yaml \
    --device cuda --steps 16000 \
    --log-every 200 --diag-every 2000 \
    --num-workers 4 \
    --ckpt-every 4000 \
    --ckpt-dir $CKPTROOT/pretrain \
    > $LOGDIR/pretrain.log 2>&1
PRETRAIN_CKPT=$CKPTROOT/pretrain/ckpt_step16000.pt
echo "[W] pretrain done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [W] Step 3/5: 5K reward fine-tune ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1v_combined_train.py \
    --config configs/stage1w_reward.yaml \
    --device cuda --steps 5000 \
    --log-every 100 --diag-every 1000 \
    --num-workers 4 \
    --ckpt-every 2500 \
    --ckpt-dir $CKPTROOT/reward \
    --resume $PRETRAIN_CKPT \
    > $LOGDIR/reward.log 2>&1
REWARD_CKPT=$CKPTROOT/reward/ckpt_step5000.pt
echo "[W] reward done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [W] Step 4/5: 8K BC retrain ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1c_bc_train.py \
    --config configs/stage1w_reward.yaml \
    --ckpt $REWARD_CKPT \
    --out $CKPTROOT/bc_policy_W.pt \
    --device cuda \
    --steps 8000 \
    --log-every 200 \
    --num-workers 4 \
    > $LOGDIR/bc.log 2>&1
echo "[W] BC done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [W] Step 5/5: online eval with T2 hyperparams ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1w_reward.yaml \
    --ckpt $REWARD_CKPT \
    --bc-ckpt $CKPTROOT/bc_policy_W.pt \
    --device cuda \
    --n-episodes 10 \
    --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 --cem-init-std 0.3 \
    --replan-every 2 \
    > $LOGDIR/eval.log 2>&1
echo "[W] eval done at $(date)" | tee -a $LOGDIR/chain.log
echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log

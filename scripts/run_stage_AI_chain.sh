#!/usr/bin/env bash
# Stage AI chain — goal-conditioned planning eval.
#   1. Extract goal frame from highest-coverage self-play episodes
#   2. Eval AB ckpt with goal-conditioned CEM (training + held-out)
#   3. Eval AH ckpt with goal-conditioned CEM (training + held-out)

set -euo pipefail

ROOT=/workspace/cser-jepa-v2
LOGDIR=$ROOT/runs_stage_AI
GOALDIR=$ROOT/goals
mkdir -p $LOGDIR $GOALDIR

# Step 1: extract goal frame
echo "=== [AI] Step 1/3: extract goal frame ===" | tee -a $LOGDIR/chain.log
date | tee -a $LOGDIR/chain.log
python -u scripts/extract_goal_frame.py \
    --rollout-dirs $ROOT/rollouts/self_play_af $ROOT/rollouts/self_play_z $ROOT/rollouts/self_play_u \
    --out $GOALDIR/pusht_goal.pt \
    --top-k 8 \
    > $LOGDIR/extract.log 2>&1
echo "[AI] goal extracted at $(date)" | tee -a $LOGDIR/chain.log

# Step 2: goal-conditioned eval on AB ckpt (best deployable to date)
AB_CKPT=$ROOT/ckpts/stage_AB/reward/ckpt_step5000.pt
AB_BC=$ROOT/ckpts/stage_AB/bc_policy_AB.pt
echo "=== [AI] Step 2a: AB + goal-cond + value (training) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1ab_reward.yaml \
    --ckpt $AB_CKPT --bc-ckpt $AB_BC \
    --device cuda --n-episodes 10 --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 \
    --cem-init-std 0.3 --replan-every 2 --seed 0 \
    --use-value --value-weight 1.0 \
    --use-goal --goal-weight 1.0 --goal-aggregate min \
    --goal-file $GOALDIR/pusht_goal.pt \
    > $LOGDIR/AB_train.log 2>&1
echo "[AI] AB train done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [AI] Step 2b: AB + goal-cond + value (held-out) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1ab_reward.yaml \
    --ckpt $AB_CKPT --bc-ckpt $AB_BC \
    --device cuda --n-episodes 10 --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 \
    --cem-init-std 0.3 --replan-every 2 --seed 100 \
    --use-value --value-weight 1.0 \
    --use-goal --goal-weight 1.0 --goal-aggregate min \
    --goal-file $GOALDIR/pusht_goal.pt \
    > $LOGDIR/AB_holdout.log 2>&1
echo "[AI] AB held-out done at $(date)" | tee -a $LOGDIR/chain.log

# Step 3: same with AH (best world model)
AH_CKPT=$ROOT/ckpts/stage_AH/reward/ckpt_step5000.pt
echo "=== [AI] Step 3a: AH + goal-cond + value (training) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1ah_reward.yaml \
    --ckpt $AH_CKPT --bc-ckpt $ROOT/ckpts/stage_Z/bc_policy_Z.pt \
    --device cuda --n-episodes 10 --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 \
    --cem-init-std 0.3 --replan-every 2 --seed 0 \
    --use-value --value-weight 1.0 \
    --use-goal --goal-weight 1.0 --goal-aggregate min \
    --goal-file $GOALDIR/pusht_goal.pt \
    > $LOGDIR/AH_train.log 2>&1
echo "[AI] AH train done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== [AI] Step 3b: AH + goal-cond + value (held-out) ===" | tee -a $LOGDIR/chain.log
python -u scripts/stage1b_online_eval.py \
    --config configs/stage1ah_reward.yaml \
    --ckpt $AH_CKPT --bc-ckpt $ROOT/ckpts/stage_Z/bc_policy_Z.pt \
    --device cuda --n-episodes 10 --max-steps 200 \
    --horizon 12 --cem-samples 256 --cem-elite 32 --cem-iters 4 \
    --cem-init-std 0.3 --replan-every 2 --seed 100 \
    --use-value --value-weight 1.0 \
    --use-goal --goal-weight 1.0 --goal-aggregate min \
    --goal-file $GOALDIR/pusht_goal.pt \
    > $LOGDIR/AH_holdout.log 2>&1
echo "[AI] AH held-out done at $(date)" | tee -a $LOGDIR/chain.log

echo "=== ALL DONE ===" | tee -a $LOGDIR/chain.log

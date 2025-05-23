#!/bin/bash
seed=3   # desired seed
algo=ardt  # ardt, dt, esper
device=cpu
export WANDB_API_KEY=91513e3bf41fe06839a645ba89f6dd2e1724a218

python main.py \
    --seed $seed \
    --data_name "mstoy" \
    --env_name "mstoy" \
    --ret_file "offline_data/${algo}_mstoy_seed${seed}" \
    --device $device \
    --algo $algo \
    --config "configs/${algo}/mstoy.yaml" \
    --checkpoint_dir "checkpoints/${algo}_mstoy_seed${seed}" \
    --model_type "dt" \
    --K 4 \
    --train_iters 1 \
    --num_steps_per_iter 10000 \
    --run_implicit False \
    --offline_file "kuhn_poker_mccfr_expert_vs_expert_results.json"
    

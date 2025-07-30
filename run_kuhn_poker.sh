#!/bin/bash

seed=3  # desired seed
algo=ardt   # ardt, dt, esper
device=cpu  # or cuda
export WANDB_API_KEY=91513e3bf41fe06839a645ba89f6dd2e1724a218

echo "⚙️  Starting main.py with debugpy on port 5678"
echo "🚦 Waiting for VS Code debugger to attach..."

# Launch Python with debugpy, disabling frozen modules issue
python -Xfrozen_modules=off -m debugpy --listen 5678 --wait-for-client main.py \
    --seed $seed \
    --data_name "toy" \
    --env_name "toy" \
    --ret_file "offline_data/${algo}_kuhn_poker${seed}" \
    --device $device \
    --algo $algo \
    --config "configs/${algo}/toy.yaml" \
    --checkpoint_dir "checkpoints/${algo}_kuhn_poker_seed${seed}" \
    --model_type "dt" \
    --K 4 \
    --train_iters 1 \
    --num_steps_per_iter 100 \
    --run_implicit False \
    --offline_file "kuhn_poker_cfr_expert_vs_expert_results.json"



# #!/bin/bash
# seed=3 
# algo=ardt
# device=cpu  #cuda
# export WANDB_API_KEY=91513e3bf41fe06839a645ba89f6dd2e1724a218

# python main.py \
#     --seed $seed \
#     --data_name "toy" \
#     --env_name "toy" \
#     --ret_file "offline_data/${algo}_kuhn_poker${seed}" \
#     --device $device \
#     --algo $algo \
#     --config "configs/${algo}/toy.yaml" \
#     --checkpoint_dir "checkpoints/${algo}_kuhn_poker_seed${seed}" \
#     --model_type "dt" \
#     --K 4 \
#     --train_iters 1 \
#     --num_steps_per_iter 100 \
#     --run_implicit False \
#     --offline_file "kuhn_poker_cfr_expert_vs_random_results.js"
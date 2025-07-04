import argparse
import random
import sys

import gym
import numpy as np
import torch

from data_loading.load_mujoco import load_mujoco_env, Trajectory
from decision_transformer.experiment import experiment
from return_transforms.generate import generate_expected, generate_maxmin
from offline_setup.offline_dataset_utils import get_trajectory_for_offline
from offline_setup.toy_env import ToyOfflineEnv

MUJOCO_TARGETS_DICT = {'halfcheetah': [2000, 3000], 'hopper': [500, 1000], 'walker2d': [800, 1000]}

def set_seed_everywhere(seed: int, env: int | None = None):
    """
    Set seed for every possible source of randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    if env is not None:
        env.seed = seed
        env.action_space.seed = seed

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--run_implicit', type=bool, required=False)
    parser.add_argument('--offline_file', type=str, required=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--data_name', type=str, required=True)
    parser.add_argument('--added_data_name', type=str, default='')
    parser.add_argument('--added_data_prop', type=float, default=0.0)
    parser.add_argument('--env_name', type=str, required=True, choices=['toy', 'mstoy', 'connect_four', 'halfcheetah', 'hopper', 'walker2d'])
    parser.add_argument('--ret_file', type=str, required=True)
    parser.add_argument('--device', type=str, required=True)
    parser.add_argument('--n_cpu', type=int, default=1)

    # For returns transformation: 
    parser.add_argument('--algo', type=str, required=True, choices=['ardt', 'dt', 'esper', 'bc'])
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--is_simple_maxmin_model', action='store_true')

    # For decision transformer training:
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--is_collect_data_only', action='store_true')
    parser.add_argument('--is_relabeling_only', action='store_true')
    parser.add_argument('--is_training_only', action='store_true')
    parser.add_argument('--is_testing_only', action='store_true')

    parser.add_argument('--traj_len', type=int, default=None)
    parser.add_argument('--top_pct_traj', type=float, default=1.)

    parser.add_argument('--model_type', type=str, default='dt', choices=['adt', 'dt', 'bc'])
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--grad_clip_norm', type=float, default=0.25)

    parser.add_argument('--train_iters', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--argmax', type=bool, default=False)
    parser.add_argument('--rtg_seq', type=bool, default=True)
    parser.add_argument('--normalize_states', action='store_true')
    
    # For decision transformer evaluation:
    parser.add_argument('--env_data_dir', type=str, default="")
    parser.add_argument('--test_adv', type=str, default='0.8')
    parser.add_argument('--env_alpha', type=float, default=0.1)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--log_to_wandb', type=bool, default=True)
    
    
    # Process args and check for consistency
    args = parser.parse_args()
    variant = vars(args)
    if variant['algo'] == 'bc':
        assert variant['model_type'] == 'bc', "Behavioural Cloning algo requires BC model type"
    if variant['device'] == 'gpu':
        variant['device'] = 'cuda'
    set_seed_everywhere(variant['seed'])
    print(f"Running with arguments:\n{variant}")

    if variant['is_collect_data_only']:
        # if the flag is set to only collect data, exit after loading the environment
        sys.exit(0)

    if not variant['is_training_only']:
        # if not only training the protagonist, (re-)do the relabeling process
        print("############### Relabeling Returns Data ###############")
        print(f"Will save relabeled file to {variant['ret_file']}")
    
        #offline_trajs = get_trajectory_for_offline("kuhn_poker_cfr_expert_vs_random_results.json")
        task = ToyOfflineEnv("kuhn_poker_cfr_expert_vs_random_results.json")
        env = task.env_cls()
        trajs = task.trajs
        print("==============")
        print(trajs[0])
        env_params = {
            "task": task, 
            "max_ep_len": 5, 
            "env_targets": list(np.arange(0, 6.01, 0.5)), 
            "scale": 5.0, 
            "action_type": "discrete"
        }

        generate_maxmin(
            env, 
            trajs, 
            variant['config'], 
            variant['ret_file'], 
            variant['device'], 
            variant['n_cpu'], 
            is_simple_model=variant['is_simple_maxmin_model'],
            is_toy=(variant['env_name'] == 'toy'),
            run_implicit=variant['run_implicit']
        )

        for test_adv in test_advs:   
            variant['test_adv'] = test_adv
            if variant['env_name'] in {'halfcheetah', 'hopper', 'walker2d'}:
                env.reset_adv_agent(variant['test_adv'], variant['device'])

            experiment(
                trajs,
                env,
                env_params['max_ep_len'],
                env_params['env_targets'],
                env_params['scale'],
                env_params['action_type'],
                variant=vars(args),
                offline_file=variant['offline_file']
            )

# Trajectory(obs=[array([1., 0., 0., 0., 0., 0., 0.]), 
# array([0., 0., 0., 1., 0., 0., 0.])], 
# actions=[0, 0], rewards=[0, 6],
#  infos=[{'adv': np.int64(2)}, {'adv': 0}], 
#  policy_infos=[PolicyInfo(), PolicyInfo()])
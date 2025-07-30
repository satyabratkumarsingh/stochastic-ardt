
import os
import json
import pyspiel
import pickle
import random
from open_spiel.python.algorithms.exploitability import exploitability, nash_conv

def calculate_exploitability(game, solver):
    """Calculate exploitability of the current policy."""
    average_policy = solver.average_policy()
    exploitability_value = exploitability(game, average_policy)
    return exploitability_value

def calculate_ne_gap(game, solver):
    """Calculate Nash Equilibrium Gap."""
    average_policy = solver.average_policy()
    ne_gap = nash_conv(game, average_policy)
    return ne_gap


def calculate_winning_rate(game, solver1, solver2, num_simulations=1000):
    """Simulate games and calculate winning rate for both players."""
    wins = [0, 0]
    for _ in range(num_simulations):
        state = game.new_initial_state()
        while not state.is_terminal():
            if state.is_chance_node():
                outcomes_with_probs = state.chance_outcomes()
                actions, probs = zip(*outcomes_with_probs)
                chosen_action = random.choices(actions, probs)[0]
                state.apply_action(chosen_action)
            else:
                current_player = state.current_player()
                solver = solver1 if current_player == 0 else solver2
                policy = solver.average_policy()
                action_probs = policy.action_probabilities(state)

                legal_actions = list(action_probs.keys())
                probabilities = list(action_probs.values())

                chosen_action = random.choices(legal_actions, weights=probabilities, k=1)[0]
                state.apply_action(chosen_action)
        final_returns = state.returns()
        if final_returns[0] > final_returns[1]:
            wins[0] += 1
        elif final_returns[1] > final_returns[0]:
            wins[1] += 1

    return [w / num_simulations for w in wins]

def save_solver(game_name, solver, episode, player_id, save_dir):
    """Save the solver's average policy."""
    os.makedirs(save_dir, exist_ok=True)
    policy = solver.average_policy()  # Keep as TabularPolicy
    if not hasattr(policy, 'states'):
        raise TypeError("Policy must be a TabularPolicy object")
    with open(f"{save_dir}/{game_name}_player{player_id}_ep{episode}.pkl", "wb") as f:
        pickle.dump(policy, f)

def load_solver(policy_path, game_name="kuhn_poker"):
    """Load a policy from a JSON file and convert it to an OpenSpiel TabularPolicy."""
    game = pyspiel.load_game(game_name)
    policy = pyspiel.TabularPolicy(game)
    
    try:
        with open(policy_path, 'r') as f:
            policy_dict = json.load(f)
        
        for state_str, action_probs in policy_dict.items():
            policy.set_action_probabilities(state_str, {i: prob for i, prob in enumerate(action_probs)})
        return policy
    except Exception as e:
        print(f"Error loading policy from {policy_path}: {e}")
        raise

def save_final_file(game_name, method, results, suffix, save_dir="gameplay_data"):
    """Save gameplay data to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{save_dir}/{game_name}_{method}_{suffix}_gameplay.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved gameplay data to {filename}")

def log_exploitability(game_name, episode, method, value):
    """Log exploitability value."""
    print(f"[{game_name}] Episode {episode} ({method}): Exploitability = {value}")

def log_ne_gap(game_name, episode, method, gap1, gap2):
    """Log Nash equilibrium gap."""
    print(f"[{game_name}] Episode {episode} ({method}): NE Gap Solver1 = {gap1}, Solver2 = {gap2}")

def log_winning_rate(game_name, episode, method, rate1, rate2):
    """Log winning rates."""
    print(f"[{game_name}] Episode {episode} ({method}): Win Rate Solver1 = {rate1}, Solver2 = {rate2}")

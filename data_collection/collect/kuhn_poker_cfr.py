import os
import pyspiel
import random
import json
import pickle

NUM_GAMES = 100000

def load_solver(policy_path):
    """Load a saved policy from a pickle file."""
    with open(policy_path, "rb") as f:
        policy = pickle.load(f)
    if not hasattr(policy, 'states'):
        raise TypeError("Loaded policy must cref. TabularPolicy object")
    return policy

def save_final_file(game_name, algorithm, results, suffix, save_dir="policies"):
    """Save gameplay results to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = f"{save_dir}/{game_name}_{algorithm}_{suffix}.json"
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {file_path}")

def simulate_games(game_name, solver1_path, solver2_path, num_games=NUM_GAMES):
    game = pyspiel.load_game(game_name)
    policy1 = load_solver(solver1_path)
    policy2 = load_solver(solver2_path)
    players = [policy1, policy2]
    gameplay_data = []

    for episode_id in range(num_games):
        state = game.new_initial_state()
        str_states = []
        num_states = []
        player_ids = []
        str_actions = []
        num_actions = []
        rewards = []

        while not state.is_terminal():
            current_player = state.current_player()
            print(f"Episode {episode_id}, Current Player: {current_player}")
            print(f"State: {state}")

            if current_player == -1:  # Chance node
                chance_outcomes = state.chance_outcomes()
                action, probability = random.choice(chance_outcomes)
                print(f"Chance node: sampled action {action} with probability {probability}")
                state.apply_action(action)
                continue

            if current_player < 0 or current_player >= game.num_players():
                print(f"Invalid current_player {current_player} at episode {episode_id}. Skipping game.")
                break

            try:
                num_states.append(state.information_state_string(current_player))
                player_ids.append(current_player)
                action_probs = players[current_player].action_probabilities(state)
                print(f"Action probabilities for player {current_player}: {action_probs}")

                if action_probs:
                    action = max(action_probs, key=action_probs.get)
                    try:
                        action_string = state.action_to_string(current_player, action)
                        str_actions.append(action_string)
                        print(f"Action string: {action_string}")
                    except Exception as e:
                        print(f"Error converting action to string: {e}")
                        str_actions.append(None)
                else:
                    print(f"No action probabilities for player {current_player}")
                    str_actions.append(None)

                num_actions.append(action)
                state.apply_action(action)
                str_states.append(str(state))
            except Exception as e:
                print(f"Error during simulation: {e}")
                break

        rewards = state.returns()
        gameplay_data.append({
            "episode_id": episode_id,
            "str_states": str_states,
            "num_states": num_states,
            "player_ids": player_ids,
            "str_actions": str_actions,
            "num_actions": num_actions,
            "rewards": rewards
        })

    return gameplay_data

def play_game():
    # CFR for Kuhn Poker
    solver1_path = "../train/policies/kuhn_poker_player1_ep10000.pkl"
    solver2_path = "../train/policies/kuhn_poker_player2_ep10000.pkl"
    results = simulate_games("kuhn_poker", solver1_path, solver2_path)
    save_final_file("kuhn_poker", "cfr", results, "expert_vs_expert")

    # MCCFR for Leduc Poker (uncomment after training MCCFR solvers)
    # solver1_path = "policies/leduc_poker_player1_ep5000.pkl"
    # solver2_path = "policies/leduc_poker_player2_ep5000.pkl"
    # results = simulate_games("leduc_poker", solver1_path, solver2_path)
    # save_final_file("leduc_poker", "mccfr", results, "expert_vs_expert")

def main():
    play_game()

if __name__ == "__main__":
    main()
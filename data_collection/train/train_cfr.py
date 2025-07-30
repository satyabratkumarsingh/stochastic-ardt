# src/train_cfr.py
from open_spiel.python.algorithms import cfr
import pyspiel
from utils import calculate_exploitability, calculate_ne_gap, calculate_winning_rate, save_solver, log_exploitability, log_ne_gap, log_winning_rate

def train_cfr(game_name="kuhn_poker", num_episodes=10000, print_freq=10, save_freq=1000, save_dir="policies"):
    """Train two CFR solvers for a given game and save their policies."""
    game = pyspiel.load_game(game_name)
    
    solver1 = cfr.CFRSolver(game)
    solver2 = cfr.CFRSolver(game)

    for ep in range(num_episodes):
        if ep % print_freq == 0:
            print(f"Episode: {ep}")
            exploitability_value1 = calculate_exploitability(game, solver1)
            exploitability_value2 = calculate_exploitability(game, solver2)
            log_exploitability(game_name, ep, 'cfr', exploitability_value1)
            log_exploitability(game_name, ep, 'cfr', exploitability_value2)

        if ep > 0 and ep % save_freq == 0:
            save_solver(game_name, solver1, ep, 1, save_dir)
            save_solver(game_name, solver2, ep, 2, save_dir)

        solver1.evaluate_and_update_policy()
        solver2.evaluate_and_update_policy()

    # Final metrics
    ne_gap1 = calculate_ne_gap(game, solver1)
    ne_gap2 = calculate_ne_gap(game, solver2)
    log_ne_gap(game_name, ep, 'cfr', ne_gap1, ne_gap2)

    winning_rates = calculate_winning_rate(game, solver1, solver2)
    log_winning_rate(game_name, ep, 'cfr', winning_rates[0], winning_rates[1])

    # Save final policies
    save_solver(game_name, solver1, num_episodes, 1, save_dir)
    save_solver(game_name, solver2, num_episodes, 2, save_dir)

if __name__ == '__main__':
    train_cfr()
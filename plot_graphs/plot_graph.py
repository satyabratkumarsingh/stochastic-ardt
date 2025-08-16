import matplotlib.pyplot as plt
import numpy as np

# Data
cql = {
    "normal_case": {
        "target_-2.0": {"mean": 0.046},
        "target_-1.5": {"mean": 0.012},
        "target_-1.0": {"mean": 0.008},
        "target_-0.5": {"mean": -0.002},
        "target_0.0": {"mean": 0.012},
        "target_0.5": {"mean": 0.015},
        "target_1.0": {"mean": 0.018},
        "target_1.5": {"mean": 0.015},
        "target_2.0": {"mean": 0.008}
    },
    "worst_case": {
        "target_-2.0": {"mean": -0.98},
        "target_-1.5": {"mean": -1.02},
        "target_-1.0": {"mean": -1.01},
        "target_-0.5": {"mean": -0.99},
        "target_0.0": {"mean": -1.00},
        "target_0.5": {"mean": -1.02},
        "target_1.0": {"mean": -1.01},
        "target_1.5": {"mean": -1.05},
        "target_2.0": {"mean": -0.95}
    }
}

implicit_q = {
    "normal_case": {
        "target_-2.0": {"mean": 1.0},
        "target_-1.0": {"mean": 1.0},
        "target_0.0": {"mean": 1.0},
        "target_1.0": {"mean": 1.0},
        "target_2.0": {"mean": 0.25}
    },
    "worst_case": {
        "target_-2.0": {"mean": 0.5},
        "target_-1.0": {"mean": 0.5},
        "target_0.0": {"mean": 0.5},
        "target_1.0": {"mean": 0.5},
        "target_2.0": {"mean": 0.1}
    }
}

min_max = {
    "normal_case": {
        "target_-2.0": {"mean": 0.26},
        "target_-1.5": {"mean": 0.19},
        "target_-1.0": {"mean": 0.21},
        "target_-0.5": {"mean": 0.35},
        "target_0.0": {"mean": 0.02},
        "target_0.5": {"mean": 0.02},
        "target_1.0": {"mean": 0.04},
        "target_1.5": {"mean": 0.03},
        "target_2.0": {"mean": 0.01}
    },
    "worst_case": {
        "target_-2.0": {"mean": -0.26},
        "target_-1.5": {"mean": -0.30},
        "target_-1.0": {"mean": -0.32},
        "target_-0.5": {"mean": -0.42},
        "target_0.0": {"mean": -1.00},
        "target_0.5": {"mean": -1.01},
        "target_1.0": {"mean": -1.00},
        "target_1.5": {"mean": -1.04},
        "target_2.0": {"mean": -0.92}
    }
}

nash_equilibrium = -0.0476

# Helper function to extract data
def extract_data(data_dict):
    targets = []
    normal_means = []
    worst_means = []
    
    for key, value in data_dict["normal_case"].items():
        target = float(key.split("_")[1])
        targets.append(target)
        normal_means.append(value["mean"])
    
    for key, value in data_dict["worst_case"].items():
        target = float(key.split("_")[1])
        worst_means.append(value["mean"])
    
    # Sort by target values
    sorted_indices = np.argsort(targets)
    targets = [targets[i] for i in sorted_indices]
    normal_means = [normal_means[i] for i in sorted_indices]
    worst_means = [worst_means[i] for i in sorted_indices]
    
    return targets, normal_means, worst_means

# Create the plot
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Model Performance Comparison: Normal vs. Worst-Case', fontsize=16, fontweight='bold')

# Colors and styles
colors = {'normal': '#2E8B57', 'worst': '#DC143C'}  # Sea Green and Crimson
markers = {'normal': 'o', 'worst': 'x'}

# Plot 1: CQL
targets_cql, normal_cql, worst_cql = extract_data(cql)
axes[0].plot(targets_cql, normal_cql, color=colors['normal'], marker=markers['normal'], 
             linewidth=2, markersize=8, label='CQL Normal Case')
axes[0].plot(targets_cql, worst_cql, color=colors['worst'], marker=markers['worst'], 
             linewidth=2, markersize=8, label='CQL Worst Case')
axes[0].axhline(y=nash_equilibrium, color='purple', linestyle='--', alpha=0.7, 
                label=f'Nash Equilibrium ({nash_equilibrium:.4f})')
axes[0].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[0].plot([-2, 2], [-2, 2], color='gray', linestyle='--', alpha=0.5, label='Ideal Performance')
axes[0].set_title('CQL Performance', fontweight='bold')
axes[0].set_xlabel('Target Return')
axes[0].set_ylabel('Achieved Mean Return')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Plot 2: Implicit Q-Learning
targets_iq, normal_iq, worst_iq = extract_data(implicit_q)
axes[1].plot(targets_iq, normal_iq, color=colors['normal'], marker=markers['normal'], 
             linewidth=2, markersize=8, label='IQL Normal Case')
axes[1].plot(targets_iq, worst_iq, color=colors['worst'], marker=markers['worst'], 
             linewidth=2, markersize=8, label='IQL Worst Case')
axes[1].axhline(y=nash_equilibrium, color='purple', linestyle='--', alpha=0.7, 
                label=f'Nash Equilibrium ({nash_equilibrium:.4f})')
axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[1].plot([-2, 2], [-2, 2], color='gray', linestyle='--', alpha=0.5, label='Ideal Performance')
axes[1].set_title('Implicit Q-Learning Performance', fontweight='bold')
axes[1].set_xlabel('Target Return')
axes[1].set_ylabel('Achieved Mean Return')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Plot 3: MinMax
targets_mm, normal_mm, worst_mm = extract_data(min_max)
axes[2].plot(targets_mm, normal_mm, color=colors['normal'], marker=markers['normal'], 
             linewidth=2, markersize=8, label='MinMax Normal Case')
axes[2].plot(targets_mm, worst_mm, color=colors['worst'], marker=markers['worst'], 
             linewidth=2, markersize=8, label='MinMax Worst Case')
axes[2].axhline(y=nash_equilibrium, color='purple', linestyle='--', alpha=0.7, 
                label=f'Nash Equilibrium ({nash_equilibrium:.4f})')
axes[2].axhline(y=0, color='gray', linestyle='-', alpha=0.3)
axes[2].plot([-2, 2], [-2, 2], color='gray', linestyle='--', alpha=0.5, label='Ideal Performance')
axes[2].set_title('MinMax Performance', fontweight='bold')
axes[2].set_xlabel('Target Return')
axes[2].set_ylabel('Achieved Mean Return')
axes[2].grid(True, alpha=0.3)
axes[2].legend()

# Adjust layout
plt.tight_layout()

# Save the figure
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('model_performance_comparison.pdf', bbox_inches='tight')
print("Graphs saved as 'model_performance_comparison.png' and 'model_performance_comparison.pdf'")

# Display statistics
print("Performance Summary:")
print("=" * 50)
print(f"Nash Equilibrium Return: {nash_equilibrium:.4f}")
print()
for name, data in [("CQL", cql), ("Implicit Q-Learning", implicit_q), ("MinMax", min_max)]:
    normal_avg = np.mean(list(data["normal_case"][k]["mean"] for k in data["normal_case"]))
    worst_avg = np.mean(list(data["worst_case"][k]["mean"] for k in data["worst_case"]))
    print(f"{name}:")
    print(f"  Normal Case Average: {normal_avg:.4f}")
    print(f"  Worst Case Average: {worst_avg:.4f}")
    print(f"  Performance Gap: {normal_avg - worst_avg:.4f}")
    print()

plt.show()
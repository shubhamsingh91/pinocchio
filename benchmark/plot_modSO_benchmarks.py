#!/usr/bin/env python3
"""
Plot SO Modified Derivatives Benchmarks
Generates bar charts for ID and FD second-order derivative timings.
4 bars per model: Case 3 (analytical), Case 1 (full SO AD), Case 2a (AD full FO), Case 2b (AD mod FO)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc

# Use mathtext (no LaTeX needed)
rc('text', usetex=False)
rc('font', family='serif', size=14)
rc('mathtext', fontset='cm')

MODEL_NAMES = ['double_pendulum', 'ur3_robot', 'hyq_f', 'atlas_f', 'talos_full_v2_f']
MODEL_LABELS = ['Double Pend (2)', r'UR$_3$ (6)', 'HyQ (18)', 'ATLAS (36)', 'Talos (50)']


def load_timing_data(data_dir):
    """Load timing data from .txt files in data_dir.
    Each file has 4 lines: case3, case1, case2a, case2b (in microseconds)."""
    case3 = []
    case1 = []
    case2a = []
    case2b = []
    for name in MODEL_NAMES:
        fpath = os.path.join(data_dir, name + '.txt')
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found, using NaN")
            case3.append(np.nan)
            case1.append(np.nan)
            case2a.append(np.nan)
            case2b.append(np.nan)
            continue
        with open(fpath) as f:
            vals = [float(line.strip()) for line in f if line.strip()]
        case3.append(vals[0])
        case1.append(vals[1])
        case2a.append(vals[2])
        case2b.append(vals[3])
    return np.array(case3), np.array(case1), np.array(case2a), np.array(case2b)


def plot_benchmark(data_dir, title_prefix, output_file):
    """Create grouped bar chart for one benchmark (ID or FD)."""
    case3, case1, case2a, case2b = load_timing_data(data_dir)

    fig, ax = plt.subplots(figsize=(8, 5))

    n_models = len(MODEL_NAMES)
    x = np.arange(n_models)
    width = 0.18

    bars3  = ax.bar(x - 1.5*width, case3,  width, label=r'Analytical (Case 3)',       color='green')
    bars2b = ax.bar(x - 0.5*width, case2b, width, label=r'AD over mod FO (Case 2b)',  color='red')
    bars2a = ax.bar(x + 0.5*width, case2a, width, label=r'AD over full FO (Case 2a)', color='blue')
    bars1  = ax.bar(x + 1.5*width, case1,  width, label=r'Full SO AD (Case 1)',        color='magenta')

    ax.set_xlabel(r'DOF ($n$)')
    ax.set_ylabel(r'Run-time ($\mu$s)')
    ax.set_title(title_prefix + r' SO Derivatives: Timing Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_yscale('log')
    ax.set_ylim(0.5, 2e5)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, which='both', axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def print_summary(data_dir, label):
    """Print timing summary and speedups."""
    case3, case1, case2a, case2b = load_timing_data(data_dir)

    print(f"\n{'='*60}")
    print(f"  {label} SO Derivatives: Timing Summary (microseconds)")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Case 3':>10} {'Case 1':>10} {'Case 2a':>10} {'Case 2b':>10}")
    print(f"{'':<20} {'Analytical':>10} {'Full AD':>10} {'AD+Full':>10} {'AD+Mod':>10}")
    print('-'*60)
    for i, name in enumerate(MODEL_NAMES):
        print(f"{name:<20} {case3[i]:>10.2f} {case1[i]:>10.2f} {case2a[i]:>10.2f} {case2b[i]:>10.2f}")

    print(f"\nSpeedups (CasADi / Analytical):")
    print(f"{'Model':<20} {'Case 1':>10} {'Case 2a':>10} {'Case 2b':>10}")
    print('-'*50)
    for i, name in enumerate(MODEL_NAMES):
        print(f"{name:<20} {case1[i]/case3[i]:>10.1f}x {case2a[i]/case3[i]:>10.1f}x {case2b[i]/case3[i]:>10.1f}x")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    id_dir = os.path.join(script_dir, 'data', 'modID_SO')
    fd_dir = os.path.join(script_dir, 'data', 'modFD_SO')

    if os.path.isdir(id_dir):
        plot_benchmark(id_dir, 'Modified ID', os.path.join(script_dir, 'modID_SO_timing.png'))
        print_summary(id_dir, 'Modified ID')

    if os.path.isdir(fd_dir):
        plot_benchmark(fd_dir, 'Modified FD', os.path.join(script_dir, 'modFD_SO_timing.png'))
        print_summary(fd_dir, 'Modified FD')

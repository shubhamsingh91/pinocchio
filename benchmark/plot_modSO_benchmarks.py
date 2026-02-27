#!/usr/bin/env python3
"""
Plot SO Modified Derivatives Benchmarks
Generates bar charts for ID and FD second-order derivative timings.
6 bars per model:
  1. FO analytical (cyan)
  2. Full SO analytical (orange) — T-Ro tensor approach
  3. Mod SO analytical (green) — our new algorithm
  4. Case 1: Full SO AD codegen (magenta)
  5. Case 2a: AD over full FO codegen (blue)
  6. Case 2b: AD over mod FO codegen (red)

Supports gcc/clang data directories via --id-dir / --fd-dir args.
"""

import os
import sys
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
    Each file has 6 lines: FO, Full SO, Mod SO, Case1, Case2a, Case2b (in microseconds).
    -1 means unavailable."""
    fo = []
    full_so = []
    mod_so = []
    case1 = []
    case2a = []
    case2b = []
    for name in MODEL_NAMES:
        fpath = os.path.join(data_dir, name + '.txt')
        if not os.path.exists(fpath):
            print(f"Warning: {fpath} not found, using NaN")
            fo.append(np.nan)
            full_so.append(np.nan)
            mod_so.append(np.nan)
            case1.append(np.nan)
            case2a.append(np.nan)
            case2b.append(np.nan)
            continue
        with open(fpath) as f:
            vals = [float(line.strip()) for line in f if line.strip()]
        if len(vals) >= 6:
            fo.append(vals[0])
            full_so.append(vals[1])
            mod_so.append(vals[2])
            case1.append(vals[3])
            case2a.append(vals[4])
            case2b.append(vals[5])
        elif len(vals) == 4:
            # Legacy 4-line format: case3, case1, case2a, case2b
            fo.append(np.nan)
            full_so.append(np.nan)
            mod_so.append(vals[0])
            case1.append(vals[1])
            case2a.append(vals[2])
            case2b.append(vals[3])
        else:
            print(f"Warning: {fpath} has {len(vals)} lines, expected 6")
            fo.append(np.nan)
            full_so.append(np.nan)
            mod_so.append(np.nan)
            case1.append(np.nan)
            case2a.append(np.nan)
            case2b.append(np.nan)
    return (np.array(fo), np.array(full_so), np.array(mod_so),
            np.array(case1), np.array(case2a), np.array(case2b))


def plot_benchmark(data_dir, title_prefix, output_file, ylim_max=None):
    """Create grouped bar chart for one benchmark (ID or FD)."""
    fo, full_so, mod_so, case1, case2a, case2b = load_timing_data(data_dir)

    # Replace negative values with NaN (missing data)
    for arr in [fo, full_so, mod_so, case1, case2a, case2b]:
        arr[arr < 0] = np.nan

    fig, ax = plt.subplots(figsize=(10, 5.5))

    n_models = len(MODEL_NAMES)
    x = np.arange(n_models)
    width = 0.13

    # Derive short tag from title_prefix (e.g. "Modified ID" -> "ID", "Modified FD" -> "FD")
    tag = title_prefix.replace('Modified ', '')  # "ID" or "FD"

    ax.bar(x - 2.5*width, fo,      width, label=f'FO analytical',                   color='cyan')
    ax.bar(x - 1.5*width, full_so, width, label=f'Full {tag} SO analytical (T-Ro)', color='orange')
    ax.bar(x - 0.5*width, mod_so,  width, label=f'Mod {tag} SO analytical (ours)',  color='green')
    ax.bar(x + 0.5*width, case1,   width, label=f'Full SO AD codegen',              color='magenta')
    ax.bar(x + 1.5*width, case2a,  width, label=f'AD over full FO codegen',         color='blue')
    ax.bar(x + 2.5*width, case2b,  width, label=f'AD over mod FO codegen',          color='red')

    ax.set_xlabel(r'DOF ($n$)')
    ax.set_ylabel(r'Run-time ($\mu$s)')
    ax.set_title(title_prefix + r' SO Derivatives: Timing Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS)
    ax.set_yscale('log')
    if ylim_max is not None:
        ax.set_ylim(0.5, ylim_max)
    else:
        all_vals = np.concatenate([fo, full_so, mod_so, case1, case2a, case2b])
        max_val = np.nanmax(all_vals)
        ax.set_ylim(0.5, max_val * 3)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(True, which='both', axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved: {output_file}")
    plt.close()


def print_summary(data_dir, label):
    """Print timing summary and speedups."""
    fo, full_so, mod_so, case1, case2a, case2b = load_timing_data(data_dir)

    print(f"\n{'='*80}")
    print(f"  {label} SO Derivatives: Timing Summary (microseconds)")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'FO':>8} {'Full SO':>10} {'Mod SO':>10} {'Case 1':>10} {'Case 2a':>10} {'Case 2b':>10}")
    print('-'*80)
    for i, name in enumerate(MODEL_NAMES):
        def fmt(v):
            return f"{v:>10.2f}" if not np.isnan(v) and v >= 0 else f"{'N/A':>10}"
        print(f"{name:<20} {fmt(fo[i])} {fmt(full_so[i])} {fmt(mod_so[i])} {fmt(case1[i])} {fmt(case2a[i])} {fmt(case2b[i])}")

    print(f"\nSpeedups vs Mod SO analytical:")
    print(f"{'Model':<20} {'FO':>8} {'Full SO':>10} {'Case 1':>10} {'Case 2a':>10} {'Case 2b':>10}")
    print('-'*70)
    for i, name in enumerate(MODEL_NAMES):
        def ratio(v):
            if np.isnan(v) or v < 0 or np.isnan(mod_so[i]) or mod_so[i] <= 0:
                return f"{'N/A':>10}"
            return f"{v/mod_so[i]:>10.1f}x"
        print(f"{name:<20} {ratio(fo[i])} {ratio(full_so[i])} {ratio(case1[i])} {ratio(case2a[i])} {ratio(case2b[i])}")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Default data directories
    id_dir = os.path.join(script_dir, 'data', 'modID_SO')
    fd_dir = os.path.join(script_dir, 'data', 'modFD_SO')
    suffix = ''

    # Parse optional args for gcc/clang support
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--id-dir' and i + 1 < len(args):
            id_dir = args[i + 1]
            i += 2
        elif args[i] == '--fd-dir' and i + 1 < len(args):
            fd_dir = args[i + 1]
            i += 2
        elif args[i] == '--suffix' and i + 1 < len(args):
            suffix = '_' + args[i + 1]
            i += 2
        else:
            i += 1

    if os.path.isdir(id_dir):
        plot_benchmark(id_dir, 'Modified ID',
                       os.path.join(script_dir, f'modID_SO_timing{suffix}.png'))
        print_summary(id_dir, 'Modified ID')

    if os.path.isdir(fd_dir):
        plot_benchmark(fd_dir, 'Modified FD',
                       os.path.join(script_dir, f'modFD_SO_timing{suffix}.png'))
        print_summary(fd_dir, 'Modified FD')

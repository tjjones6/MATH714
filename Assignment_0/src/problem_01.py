"""
Problem 01 — Math 714 Assignment 0
Recursive sequence:
    x_{k+1} = a x_k^2 + b x_k

Adds plotting utilities:
- Trajectories near fixed points
- Phase portrait
- Stability function |f'(x)|
- Stability heatmap in (a,b)-space
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt


# ============================================================
# Recurrence definition
# ============================================================

def recurrence_step(x, a, b):
    try:
        return a * x**2 + b * x
    except OverflowError:
        return np.inf


# ============================================================
# Fixed points
# ============================================================

def fixed_points(a, b):
    if a == 0:
        if b == 1:
            return ["All real numbers (identity map)"]
        return [0.0]
    return [0.0, (1 - b) / a]


# ============================================================
# Stability analysis
# ============================================================

def stability(a, b, fp):
    derivative = 2 * a * fp + b
    return abs(derivative) < 1, derivative


# ============================================================
# Sequence iteration
# ============================================================

def iterate_sequence(x0, a, b, max_iter=200, tol=1e-12, blowup=1e12):
    xs = [x0]
    x = x0

    for _ in range(max_iter):
        x_next = recurrence_step(x, a, b)
        xs.append(x_next)

        if abs(x_next) > blowup or np.isnan(x_next):
            return xs, None

        if abs(x_next - x) < tol:
            return xs, x_next

        x = x_next

    return xs, None


# ============================================================
# Convergence test near fixed points
# ============================================================

def test_convergence_near_fixed_points(a, b, perturb=1e-3):
    results = []
    fps = fixed_points(a, b)

    for fp in fps:
        if isinstance(fp, str):
            results.append((fp, None, []))
            continue

        x0 = fp + perturb
        traj, limit = iterate_sequence(x0, a, b)
        results.append((fp, limit, traj))

    return results


# ============================================================
# Plotting utilities
# ============================================================

def plot_trajectory(traj, fp, a, b):
    plt.figure(figsize=(6,4))
    plt.plot(traj, marker='o')
    plt.title(f"Trajectory near fixed point {fp}\na={a}, b={b}")
    plt.xlabel("Iteration k")
    plt.ylabel("x_k")
    plt.grid(True)
    plt.tight_layout()


def plot_phase_portrait(a, b):
    xs = np.linspace(-2, 2, 400)
    ys = recurrence_step(xs, a, b)

    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, label="x_{k+1} = a x_k^2 + b x_k")
    plt.plot(xs, xs, '--', label="x_{k+1} = x_k (identity)")
    plt.title(f"Phase Portrait\nRecurrence map for a={a}, b={b}")
    plt.xlabel("x_k")
    plt.ylabel("x_{k+1}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_stability(a, b):
    xs = np.linspace(-2, 2, 400)
    deriv = 2 * a * xs + b

    plt.figure(figsize=(6,4))
    plt.plot(xs, np.abs(deriv))
    plt.axhline(1.0, color='red', linestyle='--', label="|f'(x)| = 1")
    plt.title(f"Stability Function |f'(x)|\na={a}, b={b}")
    plt.xlabel("x")
    plt.ylabel("|f'(x)|")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


# ============================================================
# Stability heatmap
# ============================================================

def plot_stability_heatmap():
    a_vals = np.linspace(-2, 2, 400)
    b_vals = np.linspace(-2, 2, 400)

    A, B = np.meshgrid(a_vals, b_vals)

    # Stability conditions:
    # fp0 = 0 stable if |b| < 1
    stable_fp0 = (np.abs(B) < 1)

    # fp1 = (1-b)/a stable if |2 - b| < 1 (independent of a)
    stable_fp1 = (np.abs(2 - B) < 1)

    # Combine regions:
    # 0 = unstable
    # 1 = fp0 stable
    # 2 = fp1 stable
    # 3 = both stable
    heatmap = stable_fp0.astype(int) + 2 * stable_fp1.astype(int)

    plt.figure(figsize=(7,6))
    plt.imshow(
        heatmap,
        extent=[a_vals.min(), a_vals.max(), b_vals.min(), b_vals.max()],
        origin='lower',
        cmap='viridis',
        aspect='auto'
    )
    plt.colorbar(label="Stability Region Code\n0=unstable, 1=fp0 stable, 2=fp1 stable, 3=both stable")
    plt.title("Stability Heatmap in (a,b)-space")
    plt.xlabel("a")
    plt.ylabel("b")
    plt.tight_layout()


# ============================================================
# CLI argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Problem 01 recurrence analysis")
    parser.add_argument("--a", type=float, default=0.5)
    parser.add_argument("--b", type=float, default=0.3)
    parser.add_argument("--plot", action="store_true", help="Enable trajectory/phase/stability plots")
    parser.add_argument("--heatmap", action="store_true", help="Plot stability heatmap in (a,b)-space")
    return parser.parse_args()


# ============================================================
# Main driver
# ============================================================

def main():
    args = parse_args()
    a = args.a
    b = args.b

    print("\n=== Problem 01 ===")
    print(f"Parameters: a = {a}, b = {b}")

    fps = fixed_points(a, b)
    print("\nFixed points and stability:")
    for fp in fps:
        if isinstance(fp, str):
            print(f"  {fp}")
            continue
        stable, deriv = stability(a, b, fp)
        print(f"  fp = {fp: .6f}, f'(fp) = {deriv: .6f}, stable = {stable}")

    print("\nTesting convergence near fixed points:")
    results = test_convergence_near_fixed_points(a, b)

    for fp, limit, traj in results:
        print(f"  Start near {fp}: converges to {limit}")
        if args.plot and isinstance(fp, float):
            plot_trajectory(traj, fp, a, b)

    if args.plot:
        plot_phase_portrait(a, b)
        plot_stability(a, b)

    if args.heatmap:
        plot_stability_heatmap()

    if args.plot or args.heatmap:
        plt.show()

    print("\nDone.\n")


if __name__ == "__main__":
    main()

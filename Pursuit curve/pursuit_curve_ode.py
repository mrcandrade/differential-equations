"""
Pursuit Curve - Classical ODE Solution
=======================================

Problem: A pursuer (P) at (a, 0) chases a prey (E) that moves along the
y-axis with constant velocity v_e. The pursuer always points directly at
the prey and moves with velocity v_p.

Resulting ODE:
    x * y'' = -k * sqrt(1 + (y')^2)

where k = v_e / v_p (velocity ratio).

Initial conditions: y(a) = 0, y'(a) = 0
"""

import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def solve_pursuit_curve(a: float, k: float, x_end: float = 0.01, n_points: int = 2000):
    """
    Solves the pursuit curve ODE.

    Parameters
    ----------
    a : float
        Initial position of the pursuer on the x-axis (x0 = a, y0 = 0).
    k : float
        Velocity ratio v_prey / v_pursuer.
    x_end : float
        Final x value for integration (close to 0, since x decreases).
    n_points : int
        Number of evaluation points.

    Returns
    -------
    x, y : arrays with the pursuit curve.
    """
    # ODE: x * y'' = -k * sqrt(1 + y'^2)
    # Convert to first-order system:
    #   u1 = y,  u2 = y'
    #   u1' = u2
    #   u2' = -k * sqrt(1 + u2^2) / x

    def ode_system(x, u):
        y, yp = u
        if abs(x) < 1e-12:
            return [yp, 0.0]
        ypp = -k * np.sqrt(1 + yp**2) / x
        return [yp, ypp]

    x_span = (a, x_end)
    x_eval = np.linspace(a, x_end, n_points)

    sol = solve_ivp(
        ode_system, x_span, [0.0, 0.0],
        t_eval=x_eval,
        method='RK45',
        rtol=1e-10,
        atol=1e-12,
        max_step=a / 500
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    return sol.t, sol.y[0], sol.y[1]


def analytical_solution(x, a, k):
    """Analytical solution (exists for k != 1 and k == 1)."""
    r = x / a
    if abs(k - 1.0) < 1e-10:
        # k = 1
        y = (a / 2.0) * (r**2 / 2.0 - np.log(r)) - a / 4.0
    else:
        y = (a / 2.0) * (r**(1 - k) / (1 - k) - r**(1 + k) / (1 + k)) + a * k / (k**2 - 1)
    return y


def compute_prey_position(x, y, yp):
    """
    Computes the prey position along the y-axis corresponding to each
    point (x, y) on the pursuit curve.

    The prey is at (0, y_e) where y_e = y - x * y' (from the pursuit condition).
    Uses the exact derivative from the ODE solver.
    """
    y_prey = y - x * yp
    return y_prey


def plot_pursuit_curve(a, k, x_end=0.01, show_analytical=True, animate=False):
    """Plots the pursuit curve with full visualization."""

    x_num, y_num, yp_num = solve_pursuit_curve(a, k, x_end)

    # Prey position
    y_prey = compute_prey_position(x_num, y_num, yp_num)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f'Pursuit Curve  |  a = {a}, k = v_prey/v_pursuer = {k}',
                 fontsize=14, fontweight='bold')

    # --- Left panel: pursuit curve + prey ---
    ax1 = axes[0]
    ax1.plot(x_num, y_num, 'b-', linewidth=2, label='Pursuer')
    ax1.plot(np.zeros_like(y_prey), y_prey, 'r--', linewidth=2, label='Prey (y-axis)')
    ax1.plot(a, 0, 'bo', markersize=10, label=f'Pursuer start ({a}, 0)')
    ax1.plot(0, 0, 'ro', markersize=10, label='Prey start (0, 0)')

    # Some lines of sight
    step = max(1, len(x_num) // 10)
    for i in range(0, len(x_num), step):
        ax1.plot([x_num[i], 0], [y_num[i], y_prey[i]],
                 'gray', alpha=0.3, linewidth=0.8)

    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title('Trajectory in the Plane', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # --- Right panel: numerical vs analytical comparison ---
    ax2 = axes[1]
    ax2.plot(x_num, y_num, 'b-', linewidth=2, label='Numerical (RK45)')

    if show_analytical:
        x_an = np.linspace(x_end, a, 2000)
        y_an = analytical_solution(x_an, a, k)
        ax2.plot(x_an, y_an, 'r--', linewidth=2, label='Analytical')

        # Error
        y_an_interp = analytical_solution(x_num, a, k)
        error = np.abs(y_num - y_an_interp)
        ax2_twin = ax2.twinx()
        ax2_twin.semilogy(x_num, error + 1e-16, 'g-', alpha=0.5, linewidth=1)
        ax2_twin.set_ylabel('Absolute error', color='green', fontsize=11)
        ax2_twin.tick_params(axis='y', labelcolor='green')

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title('Numerical vs Analytical', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pursuit_curve_ode.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # --- Animation (optional) ---
    if animate:
        fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
        ax_anim.set_xlim(-0.5, a + 0.5)
        y_max = max(np.max(y_num), np.max(y_prey))
        ax_anim.set_ylim(-0.5, y_max + 0.5)
        ax_anim.set_aspect('equal')
        ax_anim.grid(True, alpha=0.3)
        ax_anim.set_title(f'Pursuit Animation  |  k = {k}', fontsize=13)

        trail_p, = ax_anim.plot([], [], 'b-', linewidth=1.5, alpha=0.5)
        dot_p, = ax_anim.plot([], [], 'bo', markersize=8)
        trail_e, = ax_anim.plot([], [], 'r-', linewidth=1.5, alpha=0.5)
        dot_e, = ax_anim.plot([], [], 'ro', markersize=8)
        line_sight, = ax_anim.plot([], [], 'gray', alpha=0.4, linewidth=1)

        def init():
            trail_p.set_data([], [])
            dot_p.set_data([], [])
            trail_e.set_data([], [])
            dot_e.set_data([], [])
            line_sight.set_data([], [])
            return trail_p, dot_p, trail_e, dot_e, line_sight

        skip = max(1, len(x_num) // 300)

        def update(frame):
            i = frame * skip
            if i >= len(x_num):
                i = len(x_num) - 1
            trail_p.set_data(x_num[:i+1], y_num[:i+1])
            dot_p.set_data([x_num[i]], [y_num[i]])
            trail_e.set_data(np.zeros(i+1), y_prey[:i+1])
            dot_e.set_data([0], [y_prey[i]])
            line_sight.set_data([x_num[i], 0], [y_num[i], y_prey[i]])
            return trail_p, dot_p, trail_e, dot_e, line_sight

        n_frames = len(x_num) // skip
        anim = FuncAnimation(fig_anim, update, init_func=init,
                             frames=n_frames, interval=20, blit=True)
        plt.show()

    return x_num, y_num


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pursuit Curve - ODE Solution')
    parser.add_argument('-a', '--position', type=float, default=1.0,
                        help='Initial position a (default: 1.0)')
    parser.add_argument('-k', '--ratio', type=float, default=0.5,
                        help='Velocity ratio k (default: 0.5)')
    parser.add_argument('--animate', action='store_true',
                        help='Show pursuit animation')
    args = parser.parse_args()

    a, k = args.position, args.ratio

    print("=" * 60)
    print("  PURSUIT CURVE - ODE Solution")
    print("=" * 60)
    print(f"\nSolving with a={a}, k={k}...")

    if k >= 1:
        print("NOTE: k >= 1 means the prey is as fast (or faster) than the pursuer.")
        print("      The pursuer will NEVER catch the prey.\n")

    x, y = plot_pursuit_curve(a, k, animate=args.animate)

    print(f"\nSolution computed: {len(x)} points")
    print(f"Final y (x->0): {y[-1]:.6f}")

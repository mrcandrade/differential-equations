"""
Pursuit Curve - Comparison: Classical ODE vs PINN
==================================================

Main script that runs both methods and compares the results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

from pursuit_curve_ode import solve_pursuit_curve, analytical_solution, compute_prey_position
from pursuit_curve_pinn import PursuitPINNSolver


def run_comparison(a=1.0, k=0.5, epochs=15000, n_collocation=500):
    """Runs both solvers and compares them."""

    x_min = 0.05  # Lower limit (avoid singularity at x=0)

    print("=" * 60)
    print("  PURSUIT CURVE")
    print(f"  a = {a}  |  k = v_prey/v_pursuer = {k}")
    print("=" * 60)

    # -------------------------------------------------------
    # 1. Classical solution (RK45)
    # -------------------------------------------------------
    print("\n[1/3] Solving ODE with RK45...")
    t0 = time.time()
    x_ode, y_ode, yp_ode = solve_pursuit_curve(a, k, x_end=x_min)
    t_ode = time.time() - t0
    print(f"      Completed in {t_ode:.3f}s ({len(x_ode)} points)")

    # -------------------------------------------------------
    # 2. Analytical solution
    # -------------------------------------------------------
    print("[2/3] Computing analytical solution...")
    x_anal = np.linspace(x_min, a, 2000)
    y_anal = analytical_solution(x_anal, a, k)

    # -------------------------------------------------------
    # 3. PINN
    # -------------------------------------------------------
    print(f"[3/3] Training PINN ({epochs} epochs)...\n")
    solver = PursuitPINNSolver(a, k, x_min=x_min,
                                hidden_layers=4, neurons=64,
                                activation='tanh')
    t0 = time.time()
    solver.train(epochs=epochs, n_collocation=n_collocation, lr=1e-3)
    t_pinn = time.time() - t0

    # PINN predictions
    x_eval = np.linspace(x_min, a, 1000)
    y_pinn = solver.predict(x_eval)
    y_pred, dy_pred, d2y_pred = solver.predict_with_derivatives(x_eval)

    # Prey position (via PINN)
    y_prey_pinn = y_pred - x_eval * dy_pred

    # Prey position (via ODE)
    y_prey_ode = compute_prey_position(x_ode, y_ode, yp_ode)

    # ODE residual from PINN
    residual = x_eval * d2y_pred + k * np.sqrt(1 + dy_pred**2)

    # Errors
    y_anal_eval = analytical_solution(x_eval, a, k)
    error_pinn = np.abs(y_pinn - y_anal_eval)

    y_anal_ode = analytical_solution(x_ode, a, k)
    error_ode = np.abs(y_ode - y_anal_ode)

    # -------------------------------------------------------
    # Metrics
    # -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n{'Metric':<35} {'RK45':>12} {'PINN':>12}")
    print("-" * 60)
    print(f"{'Computation time (s)':<35} {t_ode:>12.3f} {t_pinn:>12.1f}")
    print(f"{'Mean error (vs analytical)':<35} {np.mean(error_ode):>12.2e} {np.mean(error_pinn):>12.2e}")
    print(f"{'Max error (vs analytical)':<35} {np.max(error_ode):>12.2e} {np.max(error_pinn):>12.2e}")
    print(f"{'Mean ODE residual':<35} {'--':>12} {np.mean(np.abs(residual)):>12.2e}")
    print(f"{'y(a) [expected: 0]':<35} {y_ode[0]:>12.2e} {y_pinn[-1]:>12.2e}")

    # -------------------------------------------------------
    # Final comparison plot
    # -------------------------------------------------------
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Pursuit Curve  --  a = {a}, k = {k}\n'
                 f'Comparison: Classical ODE vs PINN',
                 fontsize=15, fontweight='bold')

    # 1. Pursuit trajectory (ODE)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(x_ode, y_ode, 'b-', linewidth=2, label='Pursuer')
    ax1.plot(np.zeros_like(y_prey_ode), y_prey_ode, 'r--', linewidth=2, label='Prey')
    ax1.plot(a, 0, 'bo', markersize=10)
    ax1.plot(0, 0, 'ro', markersize=10)
    step = max(1, len(x_ode) // 10)
    for i in range(0, len(x_ode), step):
        ax1.plot([x_ode[i], 0], [y_ode[i], y_prey_ode[i]], 'gray', alpha=0.3, lw=0.8)
    ax1.set_title('Trajectory (RK45)', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend(fontsize=9)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # 2. Trajectory (PINN)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(x_eval, y_pinn, 'b-', linewidth=2, label='Pursuer')
    ax2.plot(np.zeros_like(y_prey_pinn), y_prey_pinn, 'r--', linewidth=2, label='Prey')
    ax2.plot(a, 0, 'bo', markersize=10)
    ax2.plot(0, 0, 'ro', markersize=10)
    step = max(1, len(x_eval) // 10)
    for i in range(0, len(x_eval), step):
        ax2.plot([x_eval[i], 0], [y_pinn[i], y_prey_pinn[i]], 'gray', alpha=0.3, lw=0.8)
    ax2.set_title('Trajectory (PINN)', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend(fontsize=9)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)

    # 3. Overlay of all 3 solutions
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(x_ode, y_ode, 'b-', linewidth=2.5, label='RK45')
    ax3.plot(x_eval, y_pinn, 'r--', linewidth=2, label='PINN')
    ax3.plot(x_anal, y_anal, 'g:', linewidth=2, label='Analytical')
    ax3.set_title('Solution Comparison', fontsize=12)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Absolute error
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.semilogy(x_ode, error_ode + 1e-16, 'b-', linewidth=1.5, label='RK45')
    ax4.semilogy(x_eval, error_pinn + 1e-16, 'r-', linewidth=1.5, label='PINN')
    ax4.set_title('Absolute Error (vs Analytical)', fontsize=12)
    ax4.set_xlabel('x')
    ax4.set_ylabel('|error|')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # 5. ODE residual (PINN)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(x_eval, residual, 'purple', linewidth=1.5)
    ax5.axhline(0, color='k', ls='--', alpha=0.3)
    ax5.set_title('ODE Residual (PINN)', fontsize=12)
    ax5.set_xlabel('x')
    ax5.set_ylabel("x*y'' + k*sqrt(1+y'^2)")
    ax5.grid(True, alpha=0.3)

    # 6. Training history
    ax6 = fig.add_subplot(2, 3, 6)
    losses = solver.losses_history
    ax6.semilogy(losses['total'], 'k-', lw=1, label='Total', alpha=0.7)
    ax6.semilogy(losses['ode'], 'b-', lw=1, label='ODE', alpha=0.7)
    ax6.set_title('PINN Training', fontsize=12)
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Loss')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pursuit_curve_comparison.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nPlots saved to pursuit_curve_comparison.png")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pursuit Curve - ODE vs PINN Comparison')
    parser.add_argument('-a', '--position', type=float, default=1.0,
                        help='Initial position a (default: 1.0)')
    parser.add_argument('-k', '--ratio', type=float, default=0.5,
                        help='Velocity ratio k (default: 0.5)')
    parser.add_argument('-e', '--epochs', type=int, default=15000,
                        help='PINN epochs (default: 15000)')
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PURSUIT CURVE - ODE vs PINN")
    print("=" * 60)

    run_comparison(a=args.position, k=args.ratio, epochs=args.epochs)

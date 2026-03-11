"""
Pursuit Curve - PINN (Physics-Informed Neural Network)
=======================================================

Solves the pursuit curve ODE using a Physics-Informed Neural Network (PINN).

ODE:  x * y''(x) + k * sqrt(1 + y'(x)^2) = 0
ICs:  y(a) = 0,  y'(a) = 0

The neural network learns y(x) directly, and the loss function penalizes
the ODE residual at collocation points. Initial conditions y(a) = 0 and
y'(a) = 0 are enforced exactly via a hard constraint: y_hat(x) = (x-a)^2 * NN(x).
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

from pursuit_curve_ode import solve_pursuit_curve, analytical_solution

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# =====================================================================
# Neural Network
# =====================================================================
class PursuitPINN(nn.Module):
    """Neural network to approximate y(x) in the pursuit curve.

    Uses hard constraint: y_hat(x) = (x - a)^2 * NN(x)
    This automatically satisfies y(a) = 0 and y'(a) = 0.
    """

    def __init__(self, hidden_layers=4, neurons=64, activation='tanh',
                 a=1.0, x_min=0.05):
        super().__init__()
        self.a = a
        self.x_min = x_min
        layers = []
        in_dim = 1

        act_fn = {
            'tanh': nn.Tanh,
            'sin': SinActivation,
            'gelu': nn.GELU,
        }.get(activation, nn.Tanh)

        for i in range(hidden_layers):
            layers.append(nn.Linear(in_dim, neurons))
            layers.append(act_fn())
            in_dim = neurons

        layers.append(nn.Linear(neurons, 1))
        self.net = nn.Sequential(*layers)

        # Xavier initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Normalize input to [0, 1] for better training stability
        x_norm = (x - self.x_min) / (self.a - self.x_min)
        # Hard constraint: (x - a)^2 * NN(x_norm) ensures y(a) = 0 and y'(a) = 0
        return (x - self.a) ** 2 * self.net(x_norm)


class SinActivation(nn.Module):
    """Sinusoidal activation - works well for PINNs."""
    def forward(self, x):
        return torch.sin(x)


# =====================================================================
# PINN Training
# =====================================================================
class PursuitPINNSolver:
    """Solves the pursuit curve via PINN."""

    def __init__(self, a: float, k: float, x_min: float = 0.05,
                 hidden_layers: int = 4, neurons: int = 64,
                 activation: str = 'tanh', device: str = 'auto'):

        self.a = a
        self.k = k
        self.x_min = x_min

        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = PursuitPINN(hidden_layers, neurons, activation,
                                    a=a, x_min=x_min).to(self.device)

        self.losses_history = {'total': [], 'ode': []}

    def _compute_derivatives(self, x):
        """Computes y, y', y'' using PyTorch autograd."""
        y = self.model(x)

        # First derivative
        dy_dx = torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True
        )[0]

        # Second derivative
        d2y_dx2 = torch.autograd.grad(
            dy_dx, x, grad_outputs=torch.ones_like(dy_dx),
            create_graph=True, retain_graph=True
        )[0]

        return y, dy_dx, d2y_dx2

    def _loss_ode(self, x_col):
        """ODE residual loss: x*y'' + k*sqrt(1 + y'^2) = 0"""
        y, dy, d2y = self._compute_derivatives(x_col)
        residual = x_col * d2y + self.k * torch.sqrt(1 + dy**2)
        return torch.mean(residual**2)

    def _generate_collocation(self, n_collocation, extra_points=None):
        """Generates collocation points: uniform + log-spaced near singularity."""
        n_uniform = int(n_collocation * 0.6)
        n_log = n_collocation - n_uniform

        # Uniform part
        x_uniform = np.linspace(self.x_min, self.a, n_uniform)

        # Log-spaced near x_min (denser where ODE is stiff)
        x_log = np.logspace(np.log10(self.x_min),
                            np.log10(self.x_min + 0.3 * (self.a - self.x_min)),
                            n_log)

        x_col = np.unique(np.concatenate([x_uniform, x_log]))

        # Append adaptive (RAR) points if provided
        if extra_points is not None and len(extra_points) > 0:
            x_col = np.unique(np.concatenate([x_col, extra_points]))

        return x_col

    def _find_high_residual_points(self, n_candidates=2000, n_add=50):
        """Residual-Based Adaptive Resampling: finds points with highest ODE residual."""
        self.model.eval()
        x_cand = np.linspace(self.x_min, self.a, n_candidates)
        x_t = torch.tensor(x_cand.reshape(-1, 1), dtype=torch.float32,
                           requires_grad=True, device=self.device)
        y, dy, d2y = self._compute_derivatives(x_t)
        residual = (x_t * d2y + self.k * torch.sqrt(1 + dy**2)).detach().cpu().numpy().flatten()
        residual_abs = np.abs(residual)

        # Select top-n_add points by residual magnitude
        top_idx = np.argsort(residual_abs)[-n_add:]
        return x_cand[top_idx]

    def train(self, epochs: int = 10000, n_collocation: int = 500,
              lr: float = 1e-3, lbfgs_iters: int = 500,
              rar_interval: int = 2000, rar_points: int = 50,
              verbose: bool = True):
        """Trains the PINN with Adam + RAR followed by L-BFGS refinement.

        Uses cosine annealing LR and residual-based adaptive resampling.
        """

        # Reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max(1, epochs // 5), T_mult=2, eta_min=1e-6
        )

        # Initial collocation points (uniform + log-spaced)
        adaptive_points = None
        x_col_np = self._generate_collocation(n_collocation)

        start_time = time.time()
        best_loss = float('inf')
        best_state = None

        # === Phase 1: Adam with Cosine Annealing + RAR ===
        for epoch in range(1, epochs + 1):
            self.model.train()

            # Residual-Based Adaptive Resampling
            if rar_interval > 0 and epoch > 1 and epoch % rar_interval == 0:
                new_pts = self._find_high_residual_points(n_add=rar_points)
                if adaptive_points is None:
                    adaptive_points = new_pts
                else:
                    adaptive_points = np.unique(np.concatenate([adaptive_points, new_pts]))
                x_col_np = self._generate_collocation(n_collocation, adaptive_points)
                if verbose:
                    print(f"[RAR]   Epoch {epoch}: added {rar_points} points "
                          f"(total collocation: {len(x_col_np)})")

            # Add small noise for regularization
            noise = np.random.uniform(-0.005 * (self.a - self.x_min),
                                       0.005 * (self.a - self.x_min),
                                       len(x_col_np))
            x_noisy = np.clip(x_col_np + noise, self.x_min, self.a)

            x_col = torch.tensor(x_noisy.reshape(-1, 1), dtype=torch.float32,
                                 requires_grad=True, device=self.device)

            loss_ode = self._loss_ode(x_col)

            optimizer.zero_grad()
            loss_ode.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(epoch + epoch / epochs)

            self.losses_history['total'].append(loss_ode.item())
            self.losses_history['ode'].append(loss_ode.item())

            if loss_ode.item() < best_loss:
                best_loss = loss_ode.item()
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if verbose and epoch % 1000 == 0:
                elapsed = time.time() - start_time
                current_lr = optimizer.param_groups[0]['lr']
                print(f"[Adam]  Epoch {epoch:6d}/{epochs} | "
                      f"Loss: {loss_ode.item():.2e} | "
                      f"lr: {current_lr:.1e} | "
                      f"Time: {elapsed:.1f}s")

        # Restore best Adam model before L-BFGS
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # === Phase 2: L-BFGS refinement ===
        if lbfgs_iters > 0:
            if verbose:
                print(f"\n[L-BFGS] Starting refinement ({lbfgs_iters} iterations)...")

            x_col_fixed = torch.tensor(x_col_np.reshape(-1, 1), dtype=torch.float32,
                                        requires_grad=True, device=self.device)

            optimizer_lbfgs = torch.optim.LBFGS(
                self.model.parameters(),
                lr=1.0,
                max_iter=20,
                history_size=50,
                line_search_fn='strong_wolfe'
            )

            for it in range(1, lbfgs_iters + 1):
                def closure():
                    optimizer_lbfgs.zero_grad()
                    loss = self._loss_ode(x_col_fixed)
                    loss.backward()
                    return loss

                loss = optimizer_lbfgs.step(closure)

                self.losses_history['total'].append(loss.item())
                self.losses_history['ode'].append(loss.item())

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

                if verbose and it % 100 == 0:
                    elapsed = time.time() - start_time
                    print(f"[L-BFGS] Iter {it:5d}/{lbfgs_iters} | "
                          f"Loss: {loss.item():.2e} | "
                          f"Time: {elapsed:.1f}s")

            if best_state is not None:
                self.model.load_state_dict(best_state)

        total_time = time.time() - start_time
        if verbose:
            print(f"\nTraining completed in {total_time:.1f}s")
            print(f"Best loss: {best_loss:.2e}")

        return self.losses_history

    def predict(self, x_np):
        """Predicts y(x) using the trained model."""
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32,
                               device=self.device)
            y_t = self.model(x_t)
        return y_t.cpu().numpy().flatten()

    def predict_with_derivatives(self, x_np):
        """Predicts y, y', y'' using the trained model."""
        self.model.eval()
        x_t = torch.tensor(x_np.reshape(-1, 1), dtype=torch.float32,
                           requires_grad=True, device=self.device)
        y, dy, d2y = self._compute_derivatives(x_t)
        return (y.detach().cpu().numpy().flatten(),
                dy.detach().cpu().numpy().flatten(),
                d2y.detach().cpu().numpy().flatten())

    def save_model(self, path):
        """Saves the trained model to disk."""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Loads a trained model from disk."""
        self.model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))


# =====================================================================
# Visualization
# =====================================================================
def plot_results(solver, a, k, x_min):
    """Plots results comparing PINN vs reference solution."""

    # Evaluation points
    x_eval = np.linspace(x_min, a, 1000)

    # PINN prediction
    y_pinn = solver.predict(x_eval)

    # Reference (RK45)
    x_ref, y_ref, _ = solve_pursuit_curve(a, k, x_end=x_min)

    # Analytical solution
    y_anal = analytical_solution(x_eval, a, k)

    # ODE residual from PINN
    y_pred, dy_pred, d2y_pred = solver.predict_with_derivatives(x_eval)
    residual = x_eval * d2y_pred + k * np.sqrt(1 + dy_pred**2)

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'PINN - Pursuit Curve  |  a={a}, k={k}',
                 fontsize=14, fontweight='bold')

    # 1. Compared curves
    ax = axes[0, 0]
    ax.plot(x_eval, y_pinn, 'b-', linewidth=2, label='PINN')
    ax.plot(x_ref, y_ref, 'r--', linewidth=2, label='RK45')
    ax.plot(x_eval, y_anal, 'g:', linewidth=2, label='Analytical')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Comparison: PINN vs RK45 vs Analytical')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Absolute error
    ax = axes[0, 1]
    error_anal = np.abs(y_pinn - y_anal)
    ax.semilogy(x_eval, error_anal + 1e-16, 'b-', linewidth=1.5, label='|PINN - Analytical|')
    y_ref_interp = np.interp(x_eval, x_ref, y_ref)
    error_rk = np.abs(y_pinn - y_ref_interp)
    ax.semilogy(x_eval, error_rk + 1e-16, 'r--', linewidth=1.5, label='|PINN - RK45|')
    ax.set_xlabel('x')
    ax.set_ylabel('Absolute error')
    ax.set_title('PINN Error')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. ODE residual
    ax = axes[1, 0]
    ax.plot(x_eval, residual, 'purple', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('Residual')
    ax.set_title("ODE Residual: x*y'' + k*sqrt(1+y'^2)")
    ax.grid(True, alpha=0.3)

    # 4. Loss history
    ax = axes[1, 1]
    losses = solver.losses_history
    ax.semilogy(losses['total'], 'k-', linewidth=1, label='Total', alpha=0.7)
    ax.semilogy(losses['ode'], 'b-', linewidth=1, label='ODE', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pursuit_curve_pinn.png'), dpi=150, bbox_inches='tight')
    plt.show()

    # Metrics
    print("\n" + "=" * 50)
    print("PINN METRICS")
    print("=" * 50)
    print(f"Mean error (vs analytical):  {np.mean(error_anal):.2e}")
    print(f"Max error (vs analytical):   {np.max(error_anal):.2e}")
    print(f"Mean ODE residual:           {np.mean(np.abs(residual)):.2e}")
    print(f"Max ODE residual:            {np.max(np.abs(residual)):.2e}")
    print(f"IC - y(a):                   {y_pinn[-1]:.2e} (expected: 0)")


def plot_pursuit_trajectory(solver, a, k, x_min):
    """Plots the pursuit trajectory using the PINN solution."""
    x_eval = np.linspace(x_min, a, 1000)
    y_pinn = solver.predict(x_eval)

    # Prey position
    y_pred, dy_pred, _ = solver.predict_with_derivatives(x_eval)
    y_prey = y_pred - x_eval * dy_pred

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_eval, y_pinn, 'b-', linewidth=2.5, label='Pursuer (PINN)')
    ax.plot(np.zeros_like(y_prey), y_prey, 'r--', linewidth=2, label='Prey')
    ax.plot(a, 0, 'bo', markersize=12, zorder=5)
    ax.plot(0, 0, 'ro', markersize=12, zorder=5)

    # Lines of sight
    step = max(1, len(x_eval) // 12)
    for i in range(0, len(x_eval), step):
        ax.plot([x_eval[i], 0], [y_pinn[i], y_prey[i]],
                'gray', alpha=0.3, linewidth=0.8)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'Pursuit Trajectory (PINN)  |  a={a}, k={k}',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pursuit_curve_pinn_trajectory.png'), dpi=150, bbox_inches='tight')
    plt.show()


# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Pursuit Curve - PINN Solver')
    parser.add_argument('-a', '--position', type=float, default=1.0,
                        help='Initial position a (default: 1.0)')
    parser.add_argument('-k', '--ratio', type=float, default=0.5,
                        help='Velocity ratio k = v_prey/v_pursuer (default: 0.5)')
    parser.add_argument('-e', '--epochs', type=int, default=15000,
                        help='Training epochs (default: 15000)')
    parser.add_argument('-n', '--collocation', type=int, default=500,
                        help='Collocation points (default: 500)')
    parser.add_argument('--lbfgs', type=int, default=500,
                        help='L-BFGS iterations (default: 500)')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save trained model')
    args = parser.parse_args()

    a, k = args.position, args.ratio
    x_min = 0.05

    print("=" * 60)
    print("  PURSUIT CURVE - PINN")
    print("=" * 60)
    print(f"\nDevice: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print(f"Parameters: a={a}, k={k}, x_min={x_min}")
    print(f"Training: {args.epochs} epochs, {args.collocation} collocation points\n")

    solver = PursuitPINNSolver(a, k, x_min=x_min,
                                hidden_layers=4, neurons=64,
                                activation='tanh')

    solver.train(epochs=args.epochs, n_collocation=args.collocation,
                 lr=1e-3, lbfgs_iters=args.lbfgs)

    if args.save:
        solver.save_model(args.save)
        print(f"Model saved to {args.save}")

    plot_results(solver, a, k, x_min)
    plot_pursuit_trajectory(solver, a, k, x_min)

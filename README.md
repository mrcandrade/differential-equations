# Pursuit Curves: Classical ODE, Physics-Informed Neural Networks, and 3D Simulation

> A comprehensive computational study of pursuit curves solved via classical numerical methods (Runge-Kutta), Physics-Informed Neural Networks (PINNs), and extended to a full 3D missile/drone pursuit simulator with multiple guidance strategies.

---


## 1. Abstract

This project presents a complete computational investigation of **pursuit curves** — a classical problem in differential equations — attacked from three perspectives: (i) the classical numerical solution via the Runge-Kutta-Fehlberg (RK45) adaptive method, (ii) an alternative solution using a **Physics-Informed Neural Network (PINN)** trained to satisfy the governing ODE and its initial conditions, and (iii) a generalized **3D pursuit simulator** implementing real-world missile guidance strategies (Pure Pursuit, Proportional Navigation, Lead Pursuit) against configurable prey maneuvers in three-dimensional space. The analytical closed-form solution serves as the ground truth for error quantification. All equations, algorithms, network architectures, and implementation details are documented in full.

---

## 2. Introduction

The **pursuit problem** is one of the oldest problems in differential equations, first studied by Pierre Bouguer in 1732 and later generalized by Leonardo Euler. The problem asks: given a prey moving along a known path and a pursuer that always moves directly toward the prey, what is the trajectory of the pursuer?

This seemingly simple question gives rise to a nonlinear second-order ODE that admits analytical solutions only in special cases. The problem has direct applications in:

- **Missile guidance and interception** — the mathematical foundation of proportional navigation
- **Autonomous drone pursuit** — UAV tracking and interception algorithms
- **Robotics** — path planning and moving-target tracking
- **Biology** — predator-prey chasing behavior modeling
- **Coast Guard and naval interception** — optimal intercept course calculation

This project explores the pursuit curve through three complementary lenses:

1. **Classical ODE solver**: Using SciPy's `solve_ivp` with the RK45 adaptive method — the standard numerical approach to initial value problems.

2. **Physics-Informed Neural Network**: A neural network that learns the solution $y(x)$ by minimizing a loss function that encodes the ODE physics and boundary conditions — no training data required.

3. **3D Pursuit Simulator**: A generalization to three dimensions with realistic constraints (limited acceleration, multiple guidance laws, diverse evasion maneuvers) and real-time Matplotlib animation.

---

## 3. Mathematical Formulation of the Pursuit Curve

### 3.1 Problem Statement

Consider a two-dimensional plane. A **prey** $E$ (evader) starts at the origin $(0, 0)$ and moves upward along the $y$-axis with constant speed $v_e$. A **pursuer** $P$ starts at position $(a, 0)$ on the $x$-axis and moves with constant speed $v_p$, always heading directly toward the current position of the prey.

We seek the trajectory $y(x)$ of the pursuer, where $x$ is the pursuer's $x$-coordinate (which decreases from $a$ toward $0$ as the pursuer approaches the $y$-axis).

**Assumptions:**
- The prey moves in a straight line along the $y$-axis
- The pursuer always points directly toward the prey (pure pursuit)
- Both speeds are constant: $v_e$ and $v_p$
- The pursuer starts at $(a, 0)$ and the prey at $(0, 0)$

### 3.2 Derivation of the Governing ODE

At time $t$, the prey is at position $(0, v_e t)$ and the pursuer is at $(x, y)$ where $y = y(x)$.

**Pursuit condition:** The tangent to the pursuer's path at $(x, y)$ passes through the prey's current position $(0, v_e t)$. By the slope of the line connecting $(x, y)$ to $(0, v_e t)$:

$$y'(x) = \frac{y - v_e t}{x - 0} = \frac{y - v_e t}{x}$$

Solving for the prey's position:

$$v_e t = y - x \, y'$$

Differentiating both sides with respect to $x$ (using the chain rule and noting $\frac{dt}{dx} = \frac{dt}{ds}\frac{ds}{dx}$):

$$v_e \frac{dt}{dx} = y' - (y' + x \, y'') = -x \, y''$$

**Arc length relation:** The pursuer travels along its curve with speed $v_p$, so:

$$\frac{ds}{dt} = -v_p$$

The negative sign arises because $x$ decreases as time increases (the pursuer moves toward the $y$-axis). The arc length element is:

$$ds = -\sqrt{1 + (y')^2} \, dx$$

(negative because $dx < 0$ while $ds > 0$). Therefore:

$$\frac{dt}{dx} = \frac{dt}{ds} \cdot \frac{ds}{dx} = \frac{-1}{v_p} \cdot \left(-\sqrt{1 + (y')^2}\right) = \frac{\sqrt{1 + (y')^2}}{v_p}$$

Substituting into the differentiated pursuit condition:

$$v_e \cdot \frac{\sqrt{1 + (y')^2}}{v_p} = -x \, y''$$

Defining the **velocity ratio** $k = \dfrac{v_e}{v_p}$:

$$\boxed{x \, y''(x) = -k \sqrt{1 + [y'(x)]^2}}$$

This is a **second-order nonlinear ODE**. The initial conditions come from the setup:

$$y(a) = 0, \qquad y'(a) = 0$$

The condition $y'(a) = 0$ means the pursuer initially moves horizontally (perpendicular to the prey's path), which is geometrically correct.

### 3.3 Analytical Solution

The ODE $x \, y'' = -k\sqrt{1 + (y')^2}$ can be solved by the substitution $p = y'$:

$$x \, p' = -k\sqrt{1 + p^2}$$

This is separable:

$$\frac{dp}{\sqrt{1+p^2}} = -\frac{k}{x} dx$$

Integrating both sides:

$$\sinh^{-1}(p) = -k \ln x + C_1$$

Using the initial condition $p(a) = y'(a) = 0$:

$$0 = -k \ln a + C_1 \implies C_1 = k \ln a$$

So:

$$\sinh^{-1}(p) = k \ln\left(\frac{a}{x}\right) = -k \ln\left(\frac{x}{a}\right)$$

Let $r = x/a$, then:

$$p = y' = \sinh\left(-k \ln r\right) = -\sinh(k \ln r)$$

Using the identity $\sinh(\theta) = \frac{e^\theta - e^{-\theta}}{2}$:

$$y' = -\frac{r^k - r^{-k}}{2} = \frac{r^{-k} - r^k}{2}$$

Integrating with respect to $x = a \cdot r$, so $dx = a \, dr$:

$$y = \frac{a}{2} \int \left(r^{-k} - r^k\right) dr = \frac{a}{2}\left[\frac{r^{1-k}}{1-k} - \frac{r^{1+k}}{1+k}\right] + C_2$$

Applying $y(a) = 0$ (i.e., $r = 1$):

$$0 = \frac{a}{2}\left[\frac{1}{1-k} - \frac{1}{1+k}\right] + C_2$$

$$C_2 = -\frac{a}{2} \cdot \frac{(1+k) - (1-k)}{(1-k)(1+k)} = -\frac{a}{2} \cdot \frac{2k}{1-k^2} = \frac{ak}{k^2-1}$$

**Final analytical solution for $k \neq 1$:**

$$\boxed{y(x) = \frac{a}{2}\left[\frac{(x/a)^{1-k}}{1-k} - \frac{(x/a)^{1+k}}{1+k}\right] + \frac{ak}{k^2 - 1}}$$

**Special case $k = 1$ (equal speeds):**

When $k = 1$, we need L'Hopital or direct integration:

$$y' = \frac{r^{-1} - r}{2}$$

$$y = \frac{a}{2}\left[-\ln r - \frac{r^2}{2}\right] + C_2$$

Applying $y(a) = 0$ ($r = 1$, $\ln 1 = 0$):

$$C_2 = \frac{a}{4}$$

$$\boxed{y(x) = \frac{a}{2}\left[\frac{(x/a)^2}{2} - \ln\left(\frac{x}{a}\right)\right] - \frac{a}{4}}$$

**Physical interpretation of $k$:**

| Condition | Meaning | Outcome |
|-----------|---------|---------|
| $k < 1$ | Pursuer is faster | Pursuer catches prey at a finite point on the $y$-axis |
| $k = 1$ | Equal speeds | Pursuer asymptotically approaches but never reaches the prey; $y \to \infty$ as $x \to 0$ |
| $k > 1$ | Prey is faster | Pursuer never catches the prey; the separation increases |

### 3.4 Prey Position Recovery

Given the pursuer's trajectory $(x, y(x))$, the prey's position at the corresponding instant can be recovered from the pursuit condition derived earlier:

$$y_E = y - x \, y'$$

where $y_E$ is the $y$-coordinate of the prey (its $x$-coordinate is always $0$).

In the code, $y'$ is estimated numerically using `numpy.gradient`, which computes central finite differences:

$$y'(x_i) \approx \frac{y_{i+1} - y_{i-1}}{x_{i+1} - x_{i-1}}$$

---

## 4. Classical Numerical Solution (RK45)

### 4.1 Reduction to a First-Order System

The second-order ODE:

$$x \, y'' = -k\sqrt{1 + (y')^2}$$

cannot be directly solved by standard ODE solvers, which expect first-order systems. We introduce the substitution:

$$u_1 = y, \qquad u_2 = y'$$

yielding the first-order system:

$$\begin{cases} u_1' = u_2 \\ u_2' = \dfrac{-k\sqrt{1 + u_2^2}}{x} \end{cases}$$

with initial conditions $u_1(a) = 0$, $u_2(a) = 0$.

**Singularity handling:** When $x \to 0$, the term $u_2' = -k\sqrt{1+u_2^2}/x$ diverges. In the implementation, a guard clause returns $u_2' = 0$ when $|x| < 10^{-12}$ to prevent division by zero. The integration is also stopped at $x_{\text{end}} \approx 0.01$ rather than at $x = 0$ exactly.

### 4.2 The Runge-Kutta-Fehlberg Method

The code uses SciPy's `solve_ivp` with `method='RK45'`, which implements the **Dormand-Prince** variant of the Runge-Kutta-Fehlberg method — an embedded pair of order 4 and 5.

Given a system $\mathbf{u}' = \mathbf{f}(x, \mathbf{u})$, each step computes six stages:

$$\mathbf{k}_i = \mathbf{f}\left(x_n + c_i h, \; \mathbf{u}_n + h \sum_{j=1}^{i-1} a_{ij} \mathbf{k}_j\right), \quad i = 1, \ldots, 6$$

The fourth-order solution estimate is:

$$\mathbf{u}_{n+1}^{(4)} = \mathbf{u}_n + h \sum_{i=1}^{6} b_i \mathbf{k}_i$$

The fifth-order estimate is:

$$\mathbf{u}_{n+1}^{(5)} = \mathbf{u}_n + h \sum_{i=1}^{6} b_i^* \mathbf{k}_i$$

The difference provides an error estimate:

$$\boldsymbol{\varepsilon} = \mathbf{u}_{n+1}^{(5)} - \mathbf{u}_{n+1}^{(4)}$$

The step size $h$ is adaptively controlled to maintain:

$$\|\boldsymbol{\varepsilon}\| \leq \text{atol} + \text{rtol} \cdot |\mathbf{u}_{n+1}|$$

### 4.3 Implementation Details

The function `solve_pursuit_curve(a, k, x_end, n_points)` in `pursuit_curve_ode.py` implements the solver with the following configuration:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `method` | `'RK45'` | Adaptive Runge-Kutta (4th/5th order Dormand-Prince) |
| `rtol` | $10^{-10}$ | Relative tolerance — high precision to match analytical solution |
| `atol` | $10^{-12}$ | Absolute tolerance — captures small-scale features near $x \to 0$ |
| `max_step` | $a / 500$ | Prevents large steps that could skip stiff regions |
| `x_span` | $(a, x_{\text{end}})$ | Integration direction: from $x = a$ toward $x \approx 0$ |
| `n_points` | $2000$ | Number of uniformly spaced evaluation points via `t_eval` |

**Note:** The integration proceeds in the direction of decreasing $x$ (from $a$ to $x_{\text{end}}$), which is the physically meaningful direction since the pursuer moves toward the $y$-axis.

### 4.4 Visualization of the ODE Solution

The function `plot_pursuit_curve()` generates a figure with two panels:

**Left panel — Trajectory in the plane:**
- The pursuer's path $(x, y)$ plotted as a solid blue curve
- The prey's path $(0, y_E)$ plotted on the $y$-axis as a red dashed line
- Lines of sight (gray lines) connecting pursuer to prey at uniformly spaced intervals, illustrating the "always pointing at the prey" constraint
- Start markers for both agents

**Right panel — Numerical vs Analytical:**
- Overlaid curves of the numerical solution (RK45) and the closed-form analytical solution
- Absolute error $|y_{\text{num}} - y_{\text{anal}}|$ plotted on a logarithmic secondary axis, typically showing machine-precision agreement ($\sim 10^{-12}$ or better)

**Optional animation:** When `animate=True`, a `FuncAnimation` renders the pursuit in real time. The animation uses frame skipping (`skip = max(1, len(x_num) // 300)`) to limit the total number of frames to approximately 300, ensuring smooth playback. Each frame updates:
- The pursuer and prey trail paths (growing lines)
- Current position markers (moving dots)
- The line of sight connecting them

---

## 5. Physics-Informed Neural Networks (PINNs)

### 5.1 Theoretical Foundation

Physics-Informed Neural Networks, introduced by Raissi, Perdikaris, and Karniadakis (2019), are a paradigm for solving differential equations by training a neural network whose loss function encodes the governing equations.

The key insight is that neural networks are universal function approximators (Universal Approximation Theorem, Cybenko 1989), and modern automatic differentiation (AD) frameworks like PyTorch can compute exact derivatives of the network output with respect to its inputs. This means we can:

1. Let a neural network $\hat{y}_\theta(x)$ represent the unknown solution
2. Compute $\hat{y}'_\theta(x)$ and $\hat{y}''_\theta(x)$ via automatic differentiation
3. Substitute into the ODE to compute the **residual**
4. Minimize the residual (plus boundary/initial condition errors) using gradient descent

**No training data is needed** — the physics itself provides the supervision signal.

The general framework for a differential equation $\mathcal{N}\lbrack y\rbrack(x) = 0$ with conditions $\mathcal{B}\lbrack y\rbrack(x_b) = 0$ is:

$$\mathcal{L}(\theta) = w_{\text{PDE}} \cdot \underbrace{\frac{1}{N_r}\sum_{i=1}^{N_r} \left|\mathcal{N}\lbrack\hat{y}_\theta\rbrack(x_i)\right|^2}_{\text{Residual loss}} + w_{\text{BC}} \cdot \underbrace{\frac{1}{N_b}\sum_{j=1}^{N_b} \left|\mathcal{B}\lbrack\hat{y}_\theta\rbrack(x_j)\right|^2}_{\text{Boundary/IC loss}}$$

where $\{x_i\}$ are **collocation points** distributed in the domain and $\{x_j\}$ are boundary/initial condition points.

### 5.2 Network Architecture

The neural network `PursuitPINN` in `pursuit_curve_pinn.py` has the following architecture:

```
Input (1) → [Linear(1, 64) → Tanh] × 4 → Linear(64, 1) → Output (1)
```

In detail:

| Layer | Type | Input dim | Output dim | Activation |
|-------|------|-----------|------------|------------|
| 1 | `nn.Linear` | 1 | 64 | `Tanh` |
| 2 | `nn.Linear` | 64 | 64 | `Tanh` |
| 3 | `nn.Linear` | 64 | 64 | `Tanh` |
| 4 | `nn.Linear` | 64 | 64 | `Tanh` |
| 5 (output) | `nn.Linear` | 64 | 1 | None |

**Total parameters:** $1 \times 64 + 64 + 64 \times 64 + 64 + 64 \times 64 + 64 + 64 \times 64 + 64 + 64 \times 1 + 1 = \mathbf{12,\!865}$

**Weight initialization:** All linear layers use **Xavier normal initialization** (`nn.init.xavier_normal_`), which draws weights from:

$$W \sim \mathcal{N}\left(0, \; \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)$$

This keeps the variance of activations roughly constant across layers, preventing vanishing or exploding gradients. Biases are initialized to zero.

**Activation functions:** Three options are available:
- **Tanh** (default): $\sigma(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$. Smooth, bounded in $(-1, 1)$, infinitely differentiable — critical for PINNs since we compute second derivatives.
- **Sinusoidal** (`SinActivation`): $\sigma(z) = \sin(z)$. Periodic activation that can better capture oscillatory solutions. Custom `nn.Module` implementation.
- **GELU**: $\sigma(z) = z \cdot \Phi(z)$ where $\Phi$ is the standard Gaussian CDF. Smooth alternative popular in transformers.

**Why Tanh for PINNs?** The activation function must be at least $C^2$ (twice continuously differentiable) since the ODE involves $y''$. ReLU fails this requirement (its second derivative is zero everywhere except at the origin where it's undefined). Tanh satisfies $C^\infty$ smoothness.

### 5.3 Loss Function Design

The total loss for the pursuit curve PINN consists of two components:

$$\mathcal{L}_{\text{total}} = w_{\text{ODE}} \cdot \mathcal{L}_{\text{ODE}} + w_{\text{IC}} \cdot \mathcal{L}_{\text{IC}}$$

**ODE residual loss** — enforces the differential equation at $N$ collocation points $\{x_i\}_{i=1}^{N}$:

$$\mathcal{L}_{\text{ODE}} = \frac{1}{N}\sum_{i=1}^{N} \left[x_i \, \hat{y}''(x_i) + k\sqrt{1 + [\hat{y}'(x_i)]^2}\right]^2$$

This is the mean squared residual of the ODE. If the network perfectly satisfies the ODE at all collocation points, this loss is zero.

**Initial conditions loss** — enforces $y(a) = 0$ and $y'(a) = 0$:

$$\mathcal{L}_{\text{IC}} = [\hat{y}(a)]^2 + [\hat{y}'(a)]^2$$

**Loss weights:** In the implementation, $w_{\text{ODE}} = 1.0$ and $w_{\text{IC}} = 50.0$. The high weight on the initial conditions is critical because:
- The IC loss involves only a single point, while the ODE loss averages over 500 points
- Without sufficient weighting, the optimizer may satisfy the ODE residual while drifting from the correct initial conditions, producing a valid but wrong solution (satisfying the ODE with different ICs)

### 5.4 Automatic Differentiation

The core mechanism enabling PINNs is **automatic differentiation** (AD). Unlike numerical differentiation (finite differences) or symbolic differentiation, AD computes exact derivatives by applying the chain rule to elementary operations recorded during the forward pass.

In PyTorch, this is implemented via the `torch.autograd` module. The `_compute_derivatives` method works as follows:

1. **Forward pass:** Compute $\hat{y} = \text{NN}(x)$ where $x$ has `requires_grad=True`
2. **First derivative:** Call `torch.autograd.grad(y, x, ...)` to get $\hat{y}' = \frac{\partial \hat{y}}{\partial x}$. The flags `create_graph=True` and `retain_graph=True` ensure the computation graph is preserved for higher-order differentiation.
3. **Second derivative:** Call `torch.autograd.grad(dy_dx, x, ...)` to get $\hat{y}'' = \frac{\partial^2 \hat{y}}{\partial x^2}$.

The `grad_outputs=torch.ones_like(y)` parameter is necessary because `torch.autograd.grad` computes vector-Jacobian products. For scalar-valued functions evaluated at multiple points (batch), passing ones is equivalent to computing the gradient at each point independently.

**Computational graph retention:** The `create_graph=True` flag is essential — it ensures that the derivative computation itself is part of the computational graph, so when we backpropagate through the loss (which depends on $\hat{y}''$), gradients can flow all the way back through the network parameters.

### 5.5 Training Strategy

The training loop in `PursuitPINNSolver.train()` employs several techniques:

**Optimizer: Adam**

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

Adam combines momentum ($m_t$) with adaptive per-parameter learning rates ($v_t$). Default PyTorch values: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Learning rate schedule:** `ReduceLROnPlateau` monitors the total loss and halves the learning rate when no improvement is seen for 500 consecutive epochs. Minimum learning rate: $10^{-6}$.

**Gradient clipping:** `clip_grad_norm_(parameters, max_norm=1.0)` prevents gradient explosion, which can occur when the ODE residual produces large gradients near the singularity at $x \to 0$.

**Collocation point perturbation:** Each epoch adds small random noise to the base collocation points:

$$x_i^{(\text{noisy})} = \text{clip}\left(x_i + \mathcal{U}\left(-0.01(a - x_{\min}), \; 0.01(a - x_{\min})\right), \; x_{\min}, \; a\right)$$

This acts as a regularization technique, preventing the network from "memorizing" the residual at specific points and encouraging generalization across the entire domain.

**Best model checkpointing:** The model state dictionary is saved whenever the total loss reaches a new minimum. After training, the best state is restored — this prevents the final model from being a suboptimal state if the loss oscillated near the end of training.

**Default hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Epochs | 15,000 |
| Collocation points | 500 |
| Learning rate | $10^{-3}$ |
| Scheduler patience | 500 epochs |
| LR reduction factor | 0.5 |
| Minimum LR | $10^{-6}$ |
| Gradient clip norm | 1.0 |
| Domain | $[x_{\min}, a] = [0.05, a]$ |

### 5.6 PINN Applied to the Pursuit Curve

The specific application to the pursuit curve proceeds as follows:

1. **Domain:** $x \in [x_{\min}, a]$ where $x_{\min} = 0.05$. We avoid $x = 0$ because the ODE is singular there (division by $x$).

2. **Network input/output:** The network takes $x \in \mathbb{R}$ and outputs $\hat{y}(x) \in \mathbb{R}$.

3. **Loss computation per epoch:**
   - Sample 500 collocation points in $[x_{\min}, a]$ with small random perturbation
   - Compute $\hat{y}, \hat{y}', \hat{y}''$ at all collocation points via AD
   - Compute $\mathcal{L}_{\text{ODE}} = \text{mean}\left[\left(x \hat{y}'' + k\sqrt{1+(\hat{y}')^2}\right)^2\right]$
   - Compute $\hat{y}(a), \hat{y}'(a)$ via AD at the single point $x = a$
   - Compute $\mathcal{L}_{\text{IC}} = [\hat{y}(a)]^2 + [\hat{y}'(a)]^2$
   - Compute $\mathcal{L}_{\text{total}} = 1.0 \cdot \mathcal{L}_{\text{ODE}} + 50.0 \cdot \mathcal{L}_{\text{IC}}$

4. **Prediction:** After training, `predict(x_np)` evaluates the network at arbitrary $x$ values. `predict_with_derivatives(x_np)` returns $(\hat{y}, \hat{y}', \hat{y}'')$ using AD, enabling computation of the ODE residual and prey position.

5. **Reference comparison:** The file also contains `solve_ode_reference()` and `analytical_solution()` for direct comparison using the same evaluation points.

### 5.7 Visualization of the PINN Solution

The `plot_results()` function generates a 2x2 figure:

1. **Top-left — Solution comparison:** PINN, RK45, and analytical solutions overlaid. Visual agreement indicates the PINN has learned the correct solution.

2. **Top-right — Absolute error:** $|\hat{y}_{\text{PINN}} - y_{\text{analytical}}|$ and $|\hat{y}_{\text{PINN}} - y_{\text{RK45}}|$ plotted on a log scale. Typical PINN errors range from $10^{-4}$ to $10^{-2}$, several orders of magnitude larger than RK45 ($\sim 10^{-12}$).

3. **Bottom-left — ODE residual:** The quantity $x\hat{y}'' + k\sqrt{1+(\hat{y}')^2}$ plotted across the domain. A perfectly trained PINN would show this identically zero; in practice, small nonzero residuals remain.

4. **Bottom-right — Training history:** Log-scale plot of $\mathcal{L}_{\text{total}}$, $\mathcal{L}_{\text{ODE}}$, and $\mathcal{L}_{\text{IC}}$ versus epoch. Typical behavior: rapid initial decrease, followed by a long plateau with occasional drops when the learning rate is reduced.

The `plot_pursuit_trajectory()` function generates the pursuit trajectory plot using the PINN solution, including:
- The pursuer's path from $(a, 0)$
- The prey's position computed as $y_E = \hat{y} - x\hat{y}'$ using AD-computed derivatives
- Lines of sight connecting pursuer to prey at regular intervals

**Printed metrics:**
- Mean and max absolute error vs. analytical solution
- Mean and max ODE residual magnitude
- Value of $\hat{y}(a)$ (should be $0$, tests IC satisfaction)

---

## 6. Comparison: RK45 vs PINN vs Analytical

### 6.1 Methodology

The script `main.py` runs a head-to-head comparison:

1. Solve the ODE using `solve_pursuit_curve()` (RK45)
2. Evaluate the closed-form `analytical_solution()`
3. Train a PINN using `PursuitPINNSolver`
4. Compare all three at 1000 evaluation points on $[x_{\min}, a]$

### 6.2 Metrics

The comparison table includes:

| Metric | Description |
|--------|-------------|
| Computation time | Wall-clock time for RK45 solve vs. PINN training |
| Mean error vs. analytical | $\frac{1}{N}\sum|y_{\text{method}} - y_{\text{analytical}}|$ |
| Max error vs. analytical | $\max|y_{\text{method}} - y_{\text{analytical}}|$ |
| Mean ODE residual (PINN only) | $\frac{1}{N}\sum|x\hat{y}'' + k\sqrt{1+(\hat{y}')^2}|$ |
| $y(a)$ | Should be $0$ — tests initial condition satisfaction |

**Typical results (a=1, k=0.5, 15000 epochs):**

- RK45: sub-millisecond solve time, error $\sim 10^{-12}$
- PINN: ~30-60s training time, error $\sim 10^{-3}$ to $10^{-4}$

The PINN trades precision for generality — it provides a differentiable, mesh-free approximation that could generalize to more complex scenarios where analytical/numerical solutions are unavailable.

### 6.3 Comparative Plots

The comparison figure (saved as `pursuit_curve_comparison.png`) contains six panels:

1. **Pursuit trajectory (RK45)** — with pursuer path, prey path, and lines of sight
2. **Pursuit trajectory (PINN)** — same layout using PINN predictions
3. **Solution overlay** — all three solutions (RK45, PINN, analytical) on one plot
4. **Absolute error** — log-scale comparison of both methods' errors
5. **ODE residual (PINN)** — how well the PINN satisfies the ODE
6. **Training history** — loss curves over epochs

---

## 7. 3D Pursuit Simulator

The file `simulador_perseguicao_3d.py` extends the 2D pursuit problem to a realistic 3D scenario with finite maneuverability, multiple guidance strategies, and diverse prey behaviors.

### 7.1 Simulation Architecture

The simulator is built around two dataclasses:

**`DroneConfig`** — defines each agent:

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Label for plots and displays |
| `position` | `np.ndarray` (3,) | Initial 3D position $(x, y, z)$ in meters |
| `speed` | `float` | Constant speed magnitude in m/s |
| `max_acceleration` | `float` | Maximum acceleration in m/s² (maneuverability limit) |
| `color` | `str` | Matplotlib color for visualization |
| `size` | `float` | Marker size in scatter plots |

**`SimConfig`** — defines simulation parameters:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dt` | `float` | 0.02 | Time step in seconds |
| `t_max` | `float` | 30.0 | Maximum simulation time |
| `intercept_distance` | `float` | 2.0 | Distance threshold for interception (meters) |
| `strategy` | `str` | `"pure_pursuit"` | Guidance law selection |
| `prey_maneuver` | `str` | `"straight"` | Prey behavior pattern |
| `nav_constant` | `float` | 4.0 | $N$ for proportional navigation |

### 7.2 Pursuit Strategies

Three classical missile guidance strategies are implemented:

#### 7.2.1 Pure Pursuit

The simplest guidance law: the pursuer velocity vector always points directly at the target's current position.

$$\vec{v}_P = v_P \cdot \frac{\vec{r}_T - \vec{r}_P}{\|\vec{r}_T - \vec{r}_P\|}$$

where $\vec{r}_P$ and $\vec{r}_T$ are the 3D position vectors of the pursuer and target, respectively.

**Characteristics:**
- Simple implementation
- Produces a **tail-chase** trajectory (the pursuer always lags behind)
- Inefficient against fast or maneuvering targets
- Results in a curved approach path even against straight-line targets

#### 7.2.2 Proportional Navigation (PN)

The most widely used missile guidance law. PN commands an acceleration proportional to the rotation rate of the **Line of Sight (LOS)** between pursuer and target.

The algorithm:

1. Compute the LOS vector: $\vec{r} = \vec{r}_T - \vec{r}_P$

2. Compute the relative velocity: $\vec{v}_{\text{rel}} = \vec{v}_T - v_P \hat{r}$

3. Compute the closing velocity: $V_c = -\vec{v}_{\text{rel}} \cdot \hat{r}$

4. Compute the LOS rotation rate vector:

$$\vec{\omega}_{\text{LOS}} = \frac{\vec{r} \times \vec{v}_{\text{rel}}}{\|\vec{r}\|^2}$$

5. Compute the commanded acceleration:

$$\vec{a}_{\text{cmd}} = N \cdot V_c \cdot (\vec{\omega}_{\text{LOS}} \times \hat{r})$$

where $N$ is the **navigation constant** (typically $3 \leq N \leq 5$; default $N = 4$).

6. Compute the commanded velocity:

$$\vec{v}_{\text{cmd}} = v_P \hat{r} + 0.1 \cdot \vec{a}_{\text{cmd}}$$

then normalize to maintain constant speed.

**Characteristics:**
- Produces a **near-straight-line** intercept path against constant-velocity targets
- Optimal for minimizing the required lateral acceleration
- The factor $0.1$ on $\vec{a}_{\text{cmd}}$ acts as a gain that controls how aggressively the correction is applied

#### 7.2.3 Lead Pursuit

A predictive strategy that estimates where the target will be in the future and steers toward that predicted position.

1. Estimate time to interception: $t_{\text{est}} = \frac{\|\vec{r}_T - \vec{r}_P\|}{v_P}$

2. Predict future target position:

$$\vec{r}_{\text{future}} = \vec{r}_T + \vec{v}_T \cdot t_{\text{est}} \cdot 0.7$$

The factor $0.7$ is a smoothing coefficient — using $1.0$ can cause oscillations since the target continues to move while the pursuer adjusts.

3. Steer toward the predicted position:

$$\vec{v}_P = v_P \cdot \frac{\vec{r}_{\text{future}} - \vec{r}_P}{\|\vec{r}_{\text{future}} - \vec{r}_P\|}$$

**Characteristics:**
- More efficient than pure pursuit against straight-line targets
- Can struggle against highly maneuvering targets (prediction becomes inaccurate)
- The $0.7$ factor trades interception speed for stability

### 7.3 Prey Maneuvers

Five prey maneuver patterns are implemented, all returning a velocity vector $\vec{v}_T \in \mathbb{R}^3$ given the current time $t$, position $\vec{r}$, and speed magnitude $v_e$:

#### 7.3.1 Straight Line (`maneuver_straight`)

$$\vec{d} = (1.0, \; 0.3, \; 0.05)$$
$$\vec{v}_T = v_e \cdot \frac{\vec{d}}{\|\vec{d}\|}$$

Constant-direction flight with a slight lateral and vertical drift.

#### 7.3.2 Circular (`maneuver_circular`)

$$\vec{d}(t) = \begin{pmatrix} -\sin(\omega t) \\ \cos(\omega t) \\ 0.02 \end{pmatrix}, \quad \omega = 0.5 \text{ rad/s}$$

Circular orbit in the XY plane with a slight climb. One full revolution every $T = 2\pi/\omega \approx 12.6$ seconds.

#### 7.3.3 Zigzag (`maneuver_zigzag`)

$$\vec{d}(t) = \begin{pmatrix} 1.0 \\ \text{sign}(\sin(2\pi f t)) \cdot 0.8 \\ 0.3 \sin(\pi f t) \end{pmatrix}, \quad f = 0.8 \text{ Hz}$$

Evasive maneuver with sharp lateral reversals using the `sign` function for instantaneous direction changes. The $z$-component adds a sinusoidal altitude variation. One reversal cycle every $1/f = 1.25$ seconds.

#### 7.3.4 Ascending Spiral (`maneuver_spiral`)

$$\vec{d}(t) = \begin{pmatrix} \cos(\omega t) \\ \sin(\omega t) \\ 0.3 + 0.1 t \end{pmatrix}, \quad \omega = 0.6 \text{ rad/s}$$

Circular motion in XY combined with an increasing climb rate. The climb component $0.3 + 0.1t$ ensures the spiral radius in the vertical plane increases with time.

#### 7.3.5 Pseudo-Random (`maneuver_random`)

$$\vec{d}(t) = \begin{pmatrix} \cos(0.3t) + 0.5\sin(0.7t) \\ \sin(0.4t) + 0.3\cos(1.1t) \\ 0.2\sin(0.5t) + 0.1\cos(0.9t) \end{pmatrix}$$

Sum of sinusoids at **incommensurate frequencies** (0.3, 0.7, 0.4, 1.1, 0.5, 0.9 rad/s). Since these frequencies are not rational multiples of each other, the motion appears random and never exactly repeats. This is a deterministic simulation approach that avoids the non-reproducibility of true random noise.

All maneuver functions normalize the direction vector $\vec{d}$ to unit length and multiply by the speed $v_e$ to produce the velocity vector.

### 7.4 Simulation Engine

The `simulate()` function implements a first-order Euler integration loop:

```
for each time step i = 0, 1, ..., n_steps:
    1. Compute prey velocity: v_T = maneuver_fn(t, pos_T, speed_T)
    2. Update prey position:  pos_T += v_T * dt
    3. Compute pursuer velocity: v_P = strategy_fn(pos_P, pos_T, speed_P, v_T)
    4. Apply acceleration limit (see Section 7.5)
    5. Update pursuer position: pos_P += v_P * dt
    6. Compute distance: d = ||pos_T - pos_P||
    7. If d <= intercept_distance: INTERCEPTION → stop
```

The Euler method is chosen for simplicity and because the simulation priority is physical intuition and real-time animation, not high-order accuracy. With $dt = 0.02$ s and typical speeds of 100-250 m/s, the position accuracy per step is on the order of $\frac{1}{2}a(\Delta t)^2 \sim 0.02$ m, below the interception threshold.

**Output dictionary:**

| Key | Type | Description |
|-----|------|-------------|
| `traj_pursuer` | `ndarray (N, 3)` | Pursuer trajectory |
| `traj_prey` | `ndarray (N, 3)` | Prey trajectory |
| `times` | `ndarray (N,)` | Time at each step |
| `distances` | `ndarray (N,)` | Pursuer-prey distance at each step |
| `intercepted` | `bool` | Whether interception occurred |
| `t_intercept` | `float` | Time of interception (or `t_max`) |
| `final_idx` | `int` | Index of the final simulation step |

### 7.5 Acceleration Limiting

Real missiles and drones have finite maneuverability. The simulator enforces this constraint:

1. Compute the desired velocity $\vec{v}_P^{\text{desired}}$ from the guidance law
2. Compute the previous velocity: $\vec{v}_P^{\text{prev}} = \frac{\vec{r}_P(t) - \vec{r}_P(t - \Delta t)}{\Delta t}$
3. Compute the required acceleration: $\vec{a} = \frac{\vec{v}_P^{\text{desired}} - \vec{v}_P^{\text{prev}}}{\Delta t}$
4. If $\|\vec{a}\| > a_{\max}$, clip the acceleration:

$$\vec{a}_{\text{clipped}} = a_{\max} \cdot \frac{\vec{a}}{\|\vec{a}\|}$$

5. Apply the clipped acceleration:

$$\vec{v}_P = \vec{v}_P^{\text{prev}} + \vec{a}_{\text{clipped}} \cdot \Delta t$$

6. **Renormalize** to maintain constant speed:

$$\vec{v}_P = v_P \cdot \frac{\vec{v}_P}{\|\vec{v}_P\|}$$

This ensures the pursuer maintains its constant speed while limiting how quickly it can change direction — a physical constraint representing structural $g$-limits and aerodynamic turning capability.

### 7.6 3D Animation System

The `animate_3d()` function creates a real-time 3D Matplotlib animation:

**Setup:**
- Computes axis limits from all trajectory data with a 50m margin
- Creates a `fig` with a single `Axes3D` subplot
- Initializes plot elements: trails (growing lines), current-position markers (scatter), line of sight (dashed line)
- Marks start positions with star markers
- If intercepted, marks the interception point with a yellow star

**Frame selection:** To maintain smooth playback, the animation uses frame skipping:

```python
frames = list(range(0, n_frames, anim_speed))
```

With `anim_speed=5`, every 5th simulation step is rendered, giving a $5\times$ speedup. The final frame is always included.

**Per-frame update:**
1. Extend trails up to the current frame index
2. Move scatter markers to current positions (via `_offsets3d`)
3. Update line of sight between pursuer and prey
4. Update info text overlay showing: strategy, maneuver, time, distance, speeds, status
5. Rotate camera: `ax.view_init(elev=25, azim=30 + frame * 0.3)` — a slowly rotating perspective

**GIF export:** When `save_gif=True`, the animation is saved using the Pillow writer at 30 FPS, 100 DPI. The GIF is saved to `results/pursuit_3d.gif`.

### 7.7 Metrics Dashboard

The `plot_metrics()` function generates a 2x2 figure:

1. **Distance vs Time** — The pursuer-prey distance over time. A horizontal green line marks the interception radius. An orange vertical line marks the interception time (if it occurred). Monotonically decreasing distance indicates successful pursuit.

2. **Top View (XY)** — Bird's-eye view of both trajectories projected onto the XY plane. Star markers indicate start positions; a yellow star marks interception. Equal aspect ratio for undistorted spatial representation.

3. **Altitude vs Time** — The $z$-coordinate of both agents over time. Useful for visualizing vertical pursuit dynamics, especially against spiral and zigzag maneuvers.

4. **Pursuer Maneuverability** — The angular rate of the pursuer's velocity vector in degrees per second:

$$\dot{\alpha}(t) = \frac{\arccos(\hat{v}(t) \cdot \hat{v}(t + \Delta t))}{\Delta t}$$

High angular rates indicate aggressive maneuvering. This metric reveals the "agility cost" of each guidance strategy.

The figure is saved to `results/pursuit_metrics.png`.

---

## 8. Project Structure

```
EQDIFF/
├── README.md                          # This document
├── requirements.txt                   # Python dependencies
├── results/                           # Output directory (auto-created)
│   ├── pursuit_curve_ode.png          # ODE solver output
│   ├── pursuit_curve_pinn.png         # PINN solver output
│   ├── pursuit_curve_pinn_trajectory.png
│   ├── pursuit_curve_comparison.png   # Comparison figure (from main.py)
│   ├── pursuit_metrics.png            # 3D simulator metrics
│   └── pursuit_3d.gif                 # 3D animation (optional)
├── Pursuit curve/
│   ├── pursuit_curve_ode.py           # Classical ODE solver (RK45)
│   ├── pursuit_curve_pinn.py          # PINN solver (PyTorch)
│   ├── main.py                        # Comparison script
│   └── simulador_perseguicao_3d.py    # 3D pursuit simulator
└── .venv/                             # Python virtual environment
```

---

## 9. Installation and Usage

### Prerequisites

- Python 3.9+ (tested with 3.11)
- pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd EQDIFF

# Create and activate virtual environment
python -m venv .venv

# Windows:
.\.venv\Scripts\Activate.ps1

# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Scripts

**Solve with classical ODE (RK45):**
```bash
python "Pursuit curve/pursuit_curve_ode.py"
```
Prompts for initial position $a$, velocity ratio $k$, and animation toggle.

**Solve with PINN:**
```bash
python "Pursuit curve/pursuit_curve_pinn.py"
```
Prompts for $a$, $k$, training epochs, and collocation points.

**Compare both methods:**
```bash
python "Pursuit curve/main.py"
```
Runs RK45, analytical, and PINN solutions side by side with metrics.

**3D pursuit simulator:**
```bash
python "Pursuit curve/simulador_perseguicao_3d.py"
```
Interactive menu with custom configuration or pre-built demo scenarios.

---

## 10. Dependencies and Technical Stack

| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | ≥ 2.0 | Neural network framework for PINN (autograd, `nn.Module`, optimizers) |
| **NumPy** | ≥ 1.24 | Array operations, linear algebra, gradient computation |
| **SciPy** | ≥ 1.10 | `solve_ivp` — adaptive ODE integration (RK45 method) |
| **Matplotlib** | ≥ 3.7 | All plotting: 2D curves, 3D scatter/line, `FuncAnimation`, GIF export |
| **Pillow** | ≥ 9.0 | GIF writer backend for Matplotlib animations |
| **seaborn** | ≥ 0.12 | Enhanced plot styling (optional) |

---

## 11. References

1. **Bouguer, P.** (1732). "Sur de nouvelles courbes auxquelles on peut donner le nom de lignes de poursuite." *Mémoires de l'Académie Royale des Sciences*, Paris.

2. **Nahin, P. J.** (2012). *Chases and Escapes: The Mathematics of Pursuit and Evasion*. Princeton University Press.

3. **Raissi, M., Perdikaris, P., & Karniadakis, G. E.** (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." *Journal of Computational Physics*, 378, 686-707.

4. **Cybenko, G.** (1989). "Approximation by superpositions of a sigmoidal function." *Mathematics of Control, Signals and Systems*, 2(4), 303-314.

5. **Dormand, J. R., & Prince, P. J.** (1980). "A family of embedded Runge-Kutta formulae." *Journal of Computational and Applied Mathematics*, 6(1), 19-26.

6. **Kingma, D. P., & Ba, J.** (2015). "Adam: A method for stochastic optimization." *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.

7. **Glorot, X., & Bengio, Y.** (2010). "Understanding the difficulty of training deep feedforward neural networks." *Proceedings of the 13th International Conference on Artificial Intelligence and Statistics (AISTATS)*.

8. **Zarchan, P.** (2012). *Tactical and Strategic Missile Guidance*. 6th edition, AIAA.

9. **Shneydor, N. A.** (1998). *Missile Guidance and Pursuit: Kinematics, Dynamics and Control*. Horwood Publishing.

10. **Lu, L., Meng, X., Mao, Z., & Karniadakis, G. E.** (2021). "DeepXDE: A deep learning library for solving differential equations." *SIAM Review*, 63(1), 208-228.

---

*Generated as part of the EQDIFF project — Differential Equations solved with classical and machine-learning methods.*

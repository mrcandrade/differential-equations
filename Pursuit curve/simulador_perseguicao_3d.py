"""
3D PURSUIT SIMULATOR -- MISSILES / DRONES
====================================================
Simulates 3D pursuit with real-time animation.
Allows configuring pursuer and prey characteristics.

Pursuit strategies:
  - Pure Pursuit: always points directly at the target
  - Proportional Navigation: anticipates the target's future position
  - Lead Pursuit: computes interception point

Prey maneuvers:
  - Straight line
  - Circular curve
  - Zigzag evasion
  - Ascending spiral
  - Random maneuver
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
import time

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ===========================================================================
# Drone configuration
# ===========================================================================

@dataclass
class DroneConfig:
    """Configuration for a drone (pursuer or prey)."""
    name: str = "Drone"
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    speed: float = 100.0               # m/s
    max_acceleration: float = 50.0     # m/s^2 (maneuverability)
    color: str = "red"
    size: float = 80


@dataclass
class SimConfig:
    """Simulation configuration."""
    dt: float = 0.02                    # time step (s)
    t_max: float = 30.0                 # maximum time (s)
    intercept_distance: float = 2.0     # distance to consider interception (m)
    strategy: str = "pure_pursuit"      # pure_pursuit, prop_nav, lead_pursuit
    prey_maneuver: str = "straight"     # straight, circular, zigzag, spiral, random
    nav_constant: float = 4.0          # proportional navigation constant


# ===========================================================================
# Prey maneuvers (generate flight direction at each instant)
# ===========================================================================

def maneuver_straight(t: float, pos: np.ndarray, speed: float) -> np.ndarray:
    """Straight line flight in the +X, +Y direction."""
    direction = np.array([1.0, 0.3, 0.05])
    return direction / np.linalg.norm(direction) * speed


def maneuver_circular(t: float, pos: np.ndarray, speed: float) -> np.ndarray:
    """Circular curve in the XY plane with slight climb."""
    omega = 0.5  # rad/s
    direction = np.array([
        -np.sin(omega * t),
        np.cos(omega * t),
        0.02
    ])
    return direction / np.linalg.norm(direction) * speed


def maneuver_zigzag(t: float, pos: np.ndarray, speed: float) -> np.ndarray:
    """Zigzag evasion."""
    freq = 0.8
    direction = np.array([
        1.0,
        np.sign(np.sin(2 * np.pi * freq * t)) * 0.8,
        0.3 * np.sin(np.pi * freq * t)
    ])
    return direction / np.linalg.norm(direction) * speed


def maneuver_spiral(t: float, pos: np.ndarray, speed: float) -> np.ndarray:
    """Ascending spiral."""
    omega = 0.6
    direction = np.array([
        np.cos(omega * t),
        np.sin(omega * t),
        0.3 + 0.1 * t
    ])
    return direction / np.linalg.norm(direction) * speed


def maneuver_random(t: float, pos: np.ndarray, speed: float) -> np.ndarray:
    """Smooth pseudo-random maneuver using sinusoids."""
    direction = np.array([
        np.cos(0.3 * t) + 0.5 * np.sin(0.7 * t),
        np.sin(0.4 * t) + 0.3 * np.cos(1.1 * t),
        0.2 * np.sin(0.5 * t) + 0.1 * np.cos(0.9 * t)
    ])
    return direction / np.linalg.norm(direction) * speed


MANEUVERS = {
    "straight": maneuver_straight,
    "circular": maneuver_circular,
    "zigzag": maneuver_zigzag,
    "spiral": maneuver_spiral,
    "random": maneuver_random,
}


# ===========================================================================
# Pursuit strategies
# ===========================================================================

def pure_pursuit(pos_p: np.ndarray, pos_t: np.ndarray,
                 speed_p: float, **kwargs) -> np.ndarray:
    """Pure Pursuit: points directly at the target."""
    direction = pos_t - pos_p
    dist = np.linalg.norm(direction)
    if dist < 1e-10:
        return np.zeros(3)
    return (direction / dist) * speed_p


def proportional_navigation(pos_p: np.ndarray, pos_t: np.ndarray,
                             speed_p: float, vel_t: np.ndarray = None,
                             N: float = 4.0, **kwargs) -> np.ndarray:
    """Proportional Navigation: uses the line-of-sight rotation rate."""
    r = pos_t - pos_p
    dist = np.linalg.norm(r)
    if dist < 1e-10:
        return np.zeros(3)

    r_hat = r / dist

    # Closing velocity
    vel_p_dir = r_hat  # approximate current direction
    if vel_t is not None:
        v_rel = vel_t - vel_p_dir * speed_p
        v_closing = -np.dot(v_rel, r_hat)

        # LOS (Line of Sight) rotation rate
        omega_los = np.cross(r, v_rel) / (dist ** 2)

        # Proportional navigation acceleration
        a_cmd = N * v_closing * np.cross(omega_los, r_hat)

        # Commanded velocity = base direction + correction
        dt = kwargs.get('dt', 0.02)
        vel_cmd = r_hat * speed_p + a_cmd * dt
    else:
        vel_cmd = r_hat * speed_p

    norm = np.linalg.norm(vel_cmd)
    if norm < 1e-10:
        return r_hat * speed_p
    return (vel_cmd / norm) * speed_p


def lead_pursuit(pos_p: np.ndarray, pos_t: np.ndarray,
                 speed_p: float, vel_t: np.ndarray = None, **kwargs) -> np.ndarray:
    """Lead Pursuit: computes future interception point with iterative refinement."""
    if vel_t is None:
        return pure_pursuit(pos_p, pos_t, speed_p)

    r = pos_t - pos_p
    dist = np.linalg.norm(r)
    if dist < 1e-10:
        return np.zeros(3)

    # Iterative Newton refinement for interception time (3 iterations)
    t_est = dist / speed_p
    for _ in range(3):
        future_pos = pos_t + vel_t * t_est
        new_dist = np.linalg.norm(future_pos - pos_p)
        t_est = new_dist / speed_p

    future_pos = pos_t + vel_t * t_est
    direction = future_pos - pos_p
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return np.zeros(3)
    return (direction / norm) * speed_p


STRATEGIES = {
    "pure_pursuit": pure_pursuit,
    "prop_nav": proportional_navigation,
    "lead_pursuit": lead_pursuit,
}


# ===========================================================================
# Simulation engine
# ===========================================================================

def simulate(pursuer: DroneConfig, prey: DroneConfig,
             config: SimConfig) -> dict:
    """Runs the simulation and returns the trajectories."""

    maneuver_fn = MANEUVERS[config.prey_maneuver]
    strategy_fn = STRATEGIES[config.strategy]

    n_steps = int(config.t_max / config.dt)

    # Store trajectories
    traj_p = np.zeros((n_steps + 1, 3))  # pursuer
    traj_t = np.zeros((n_steps + 1, 3))  # target (prey)
    times = np.zeros(n_steps + 1)
    distances = np.zeros(n_steps + 1)

    traj_p[0] = pursuer.position.copy()
    traj_t[0] = prey.position.copy()
    distances[0] = np.linalg.norm(traj_t[0] - traj_p[0])

    intercepted = False
    t_intercept = config.t_max
    final_idx = n_steps

    # CPA (Closest Point of Approach) tracking
    min_dist = distances[0]
    t_min_dist = 0.0

    vel_t_current = np.zeros(3)
    vel_p_current = np.zeros(3)  # Explicit velocity state

    for i in range(n_steps):
        t = i * config.dt
        times[i] = t

        # --- Move prey ---
        vel_t_current = maneuver_fn(t, traj_t[i], prey.speed)
        traj_t[i + 1] = traj_t[i] + vel_t_current * config.dt

        # --- Move pursuer ---
        vel_p_cmd = strategy_fn(
            traj_p[i], traj_t[i],
            pursuer.speed,
            vel_t=vel_t_current,
            N=config.nav_constant,
            dt=config.dt
        )

        # Limit acceleration (maneuverability) using velocity state
        if i > 0:
            accel = (vel_p_cmd - vel_p_current) / config.dt
            accel_mag = np.linalg.norm(accel)
            if accel_mag > pursuer.max_acceleration:
                accel = accel * (pursuer.max_acceleration / accel_mag)
                vel_p_cmd = vel_p_current + accel * config.dt
                # Renormalize to maintain constant speed
                norm_v = np.linalg.norm(vel_p_cmd)
                if norm_v > 1e-10:
                    vel_p_cmd = vel_p_cmd / norm_v * pursuer.speed

        vel_p_current = vel_p_cmd
        traj_p[i + 1] = traj_p[i] + vel_p_current * config.dt

        # --- Check interception ---
        dist = np.linalg.norm(traj_t[i + 1] - traj_p[i + 1])
        distances[i + 1] = dist

        # Track CPA
        if dist < min_dist:
            min_dist = dist
            t_min_dist = (i + 1) * config.dt

        if dist <= config.intercept_distance:
            intercepted = True
            t_intercept = (i + 1) * config.dt
            final_idx = i + 1
            times[i + 1] = t_intercept
            break

    times[final_idx] = final_idx * config.dt

    return {
        "traj_pursuer": traj_p[:final_idx + 1],
        "traj_prey": traj_t[:final_idx + 1],
        "times": times[:final_idx + 1],
        "distances": distances[:final_idx + 1],
        "intercepted": intercepted,
        "t_intercept": t_intercept,
        "final_idx": final_idx,
        "min_distance": min_dist,
        "t_min_distance": t_min_dist,
    }


# ===========================================================================
# 3D Animation
# ===========================================================================

def animate_3d(result: dict, pursuer: DroneConfig, prey: DroneConfig,
               config: SimConfig, anim_speed: int = 5, save_gif: bool = False):
    """Creates a 3D animation of the pursuit."""

    traj_p = result["traj_pursuer"]
    traj_t = result["traj_prey"]

    # Compute axis limits
    all_pos = np.vstack([traj_p, traj_t])
    margin = 50
    x_lim = [all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin]
    y_lim = [all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin]
    z_lim = [all_pos[:, 2].min() - margin, all_pos[:, 2].max() + margin]

    # Ensure z_lim is not degenerate
    if z_lim[1] - z_lim[0] < 10:
        z_lim = [z_lim[0] - 50, z_lim[1] + 50]

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot elements
    trail_p, = ax.plot([], [], [], color=pursuer.color, lw=1.5, alpha=0.5)
    trail_t, = ax.plot([], [], [], color=prey.color, lw=1.5, alpha=0.5)
    point_p = ax.scatter([], [], [], color=pursuer.color,
                         s=pursuer.size, marker='^',
                         edgecolors='black', zorder=5)
    point_t = ax.scatter([], [], [], color=prey.color,
                         s=prey.size, marker='o',
                         edgecolors='black', zorder=5)
    line_los, = ax.plot([], [], [], 'k--', alpha=0.3, lw=0.8)

    # Start points
    ax.scatter(*traj_p[0], color=pursuer.color, s=100, marker='*',
               edgecolors='black', label=f'{pursuer.name} (start)')
    ax.scatter(*traj_t[0], color=prey.color, s=100, marker='*',
               edgecolors='black', label=f'{prey.name} (start)')

    # Interception point
    if result["intercepted"]:
        ax.scatter(*traj_p[-1], color='yellow', s=200, marker='*',
                   edgecolors='red', linewidths=2, label='INTERCEPTION!', zorder=10)

    info_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes,
                          fontsize=10, verticalalignment='top',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='black',
                                    alpha=0.8, edgecolor='lime'),
                          color='lime')

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_zlabel('Z (m)', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)

    strategy_label = {
        "pure_pursuit": "Pure Pursuit",
        "prop_nav": "Proportional Navigation",
        "lead_pursuit": "Lead Pursuit"
    }
    maneuver_label = {
        "straight": "Straight Line",
        "circular": "Circular Curve",
        "zigzag": "Zigzag",
        "spiral": "Ascending Spiral",
        "random": "Random Maneuver"
    }

    n_frames = len(traj_p)
    frames = list(range(0, n_frames, anim_speed))
    if frames[-1] != n_frames - 1:
        frames.append(n_frames - 1)

    def init():
        trail_p.set_data_3d([], [], [])
        trail_t.set_data_3d([], [], [])
        return trail_p, trail_t, point_p, point_t, line_los, info_text

    def update(frame):
        i = frames[frame]

        # Trails
        trail_p.set_data_3d(traj_p[:i+1, 0], traj_p[:i+1, 1], traj_p[:i+1, 2])
        trail_t.set_data_3d(traj_t[:i+1, 0], traj_t[:i+1, 1], traj_t[:i+1, 2])

        # Current positions
        point_p._offsets3d = ([traj_p[i, 0]], [traj_p[i, 1]], [traj_p[i, 2]])
        point_t._offsets3d = ([traj_t[i, 0]], [traj_t[i, 1]], [traj_t[i, 2]])

        # Line of sight
        line_los.set_data_3d(
            [traj_p[i, 0], traj_t[i, 0]],
            [traj_p[i, 1], traj_t[i, 1]],
            [traj_p[i, 2], traj_t[i, 2]]
        )

        # Info
        dist = result["distances"][i]
        t = result["times"][i]
        status = "INTERCEPTED!" if (result["intercepted"] and i == len(traj_p) - 1) else "Pursuing..."

        info = (
            f"--- PURSUIT SIMULATION ---\n"
            f"Strategy:   {strategy_label.get(config.strategy, config.strategy)}\n"
            f"Maneuver:   {maneuver_label.get(config.prey_maneuver, config.prey_maneuver)}\n"
            f"Time:       {t:.2f}s\n"
            f"Distance:   {dist:.1f}m\n"
            f"Pursuer V:  {pursuer.speed:.0f} m/s\n"
            f"Prey V:     {prey.speed:.0f} m/s\n"
            f"Status:     {status}"
        )
        info_text.set_text(info)

        # Smooth camera rotation
        ax.view_init(elev=25, azim=30 + frame * 0.3)

        return trail_p, trail_t, point_p, point_t, line_los, info_text

    title = (f"3D Pursuit: {pursuer.name} vs {prey.name}\n"
             f"Strategy: {strategy_label.get(config.strategy, config.strategy)}")
    ax.set_title(title, fontsize=13, fontweight='bold')

    anim = animation.FuncAnimation(
        fig, update, init_func=init,
        frames=len(frames), interval=30, blit=False
    )

    if save_gif:
        print("Saving GIF (this may take a while)...")
        gif_path = os.path.join(RESULTS_DIR, "pursuit_3d.gif")
        anim.save(gif_path, writer='pillow', fps=30, dpi=100)
        print(f"GIF saved: {gif_path}")

    plt.tight_layout()
    plt.show()


def plot_metrics(result: dict, pursuer: DroneConfig,
                 prey: DroneConfig, config: SimConfig):
    """Plots simulation metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    times = result["times"]
    traj_p = result["traj_pursuer"]
    traj_t = result["traj_prey"]

    # 1. Distance vs Time
    axes[0, 0].plot(times, result["distances"], 'r-', lw=2)
    axes[0, 0].axhline(y=config.intercept_distance, color='g', ls='--',
                        label=f'Intercept radius ({config.intercept_distance}m)')
    if result["intercepted"]:
        axes[0, 0].axvline(x=result["t_intercept"], color='orange',
                           ls=':', label=f'Interception t={result["t_intercept"]:.2f}s')
    # Mark Closest Point of Approach
    axes[0, 0].plot(result["t_min_distance"], result["min_distance"], 'k*', ms=12,
                    label=f'CPA: {result["min_distance"]:.1f}m @ t={result["t_min_distance"]:.1f}s')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Distance (m)')
    axes[0, 0].set_title('Pursuer-Prey Distance')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. XY trajectory (top view)
    axes[0, 1].plot(traj_p[:, 0], traj_p[:, 1], color=pursuer.color,
                    lw=2, label=pursuer.name)
    axes[0, 1].plot(traj_t[:, 0], traj_t[:, 1], color=prey.color,
                    lw=2, label=prey.name)
    axes[0, 1].plot(traj_p[0, 0], traj_p[0, 1], 'k*', ms=12)
    axes[0, 1].plot(traj_t[0, 0], traj_t[0, 1], 'k*', ms=12)
    if result["intercepted"]:
        axes[0, 1].plot(traj_p[-1, 0], traj_p[-1, 1], 'y*', ms=15,
                        markeredgecolor='red', markeredgewidth=2)
    axes[0, 1].set_xlabel('X (m)')
    axes[0, 1].set_ylabel('Y (m)')
    axes[0, 1].set_title('Top View (XY)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect('equal')

    # 3. Altitude vs Time
    axes[1, 0].plot(times, traj_p[:, 2], color=pursuer.color,
                    lw=2, label=pursuer.name)
    axes[1, 0].plot(times, traj_t[:, 2], color=prey.color,
                    lw=2, label=prey.name)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Altitude Z (m)')
    axes[1, 0].set_title('Altitude vs Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Angular rate (pursuer direction change rate)
    if len(traj_p) > 2:
        vel_p = np.diff(traj_p, axis=0) / config.dt
        vel_dirs = vel_p / (np.linalg.norm(vel_p, axis=1, keepdims=True) + 1e-10)
        ang_changes = np.zeros(len(vel_dirs) - 1)
        for j in range(len(vel_dirs) - 1):
            cos_a = np.clip(np.dot(vel_dirs[j], vel_dirs[j+1]), -1, 1)
            ang_changes[j] = np.degrees(np.arccos(cos_a)) / config.dt
        axes[1, 1].plot(times[1:-1], ang_changes, 'purple', lw=1, alpha=0.7)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Angular rate (deg/s)')
        axes[1, 1].set_title('Pursuer Maneuverability')
        axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"Metrics -- {pursuer.name} vs {prey.name}", fontsize=14,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pursuit_metrics.png"), dpi=150, bbox_inches='tight')
    plt.show()


# ===========================================================================
# User interface
# ===========================================================================

def input_float(prompt: str, default: float) -> float:
    val = input(prompt).strip()
    if val == "":
        return default
    return float(val)


def input_int(prompt: str, default: int) -> int:
    val = input(prompt).strip()
    if val == "":
        return default
    return int(val)


def menu():
    print("\n" + "="*60)
    print("  3D PURSUIT SIMULATOR -- MISSILES / DRONES")
    print("="*60)

    # --- Pursuer ---
    print("\n--- PURSUER (missile/drone) ---")
    p_vel = input_float("  Speed (m/s) [200]: ", 200.0)
    p_accel = input_float("  Max acceleration (m/s^2) [80]: ", 80.0)
    p_x = input_float("  Initial position X [0]: ", 0.0)
    p_y = input_float("  Initial position Y [0]: ", 0.0)
    p_z = input_float("  Initial position Z [100]: ", 100.0)

    pursuer = DroneConfig(
        name="Missile",
        position=np.array([p_x, p_y, p_z]),
        speed=p_vel,
        max_acceleration=p_accel,
        color="red",
        size=100
    )

    # --- Prey ---
    print("\n--- PREY (target/enemy drone) ---")
    t_vel = input_float("  Speed (m/s) [120]: ", 120.0)
    t_x = input_float("  Initial position X [500]: ", 500.0)
    t_y = input_float("  Initial position Y [200]: ", 200.0)
    t_z = input_float("  Initial position Z [150]: ", 150.0)

    prey = DroneConfig(
        name="Target",
        position=np.array([t_x, t_y, t_z]),
        speed=t_vel,
        max_acceleration=100.0,
        color="blue",
        size=80
    )

    # --- Strategy ---
    print("\n--- PURSUIT STRATEGY ---")
    print("  1. Pure Pursuit (points directly at target)")
    print("  2. Proportional Navigation (anticipates position)")
    print("  3. Lead Pursuit (computes interception)")
    est = input("  Choose [1]: ").strip() or "1"
    strategy = {"1": "pure_pursuit", "2": "prop_nav", "3": "lead_pursuit"}.get(est, "pure_pursuit")

    # --- Prey maneuver ---
    print("\n--- PREY MANEUVER ---")
    print("  1. Straight line")
    print("  2. Circular curve")
    print("  3. Zigzag (evasion)")
    print("  4. Ascending spiral")
    print("  5. Random maneuver")
    man = input("  Choose [1]: ").strip() or "1"
    maneuver = {"1": "straight", "2": "circular", "3": "zigzag",
                "4": "spiral", "5": "random"}.get(man, "straight")

    # --- Simulation ---
    print("\n--- SIMULATION PARAMETERS ---")
    t_max = input_float("  Maximum time (s) [30]: ", 30.0)
    dt = input_float("  Time step (s) [0.02]: ", 0.02)
    anim_spd = input_int("  Animation speed (1-20) [5]: ", 5)
    save = input("  Save GIF? (y/n) [n]: ").strip().lower() == "y"

    config = SimConfig(
        dt=dt,
        t_max=t_max,
        intercept_distance=3.0,
        strategy=strategy,
        prey_maneuver=maneuver,
    )

    # --- Execute ---
    print("\n" + "-"*60)
    print("  Simulating...")
    print("-"*60)

    t0 = time.time()
    result = simulate(pursuer, prey, config)
    dt_sim = time.time() - t0

    # Results
    print(f"\n{'='*50}")
    print(f"  SIMULATION RESULT")
    print(f"{'='*50}")
    print(f"  Simulation time: {dt_sim:.3f}s")
    if result["intercepted"]:
        print(f"  INTERCEPTION at t = {result['t_intercept']:.2f}s!")
        dist_p = np.sum(np.linalg.norm(np.diff(result["traj_pursuer"], axis=0), axis=1))
        dist_t = np.sum(np.linalg.norm(np.diff(result["traj_prey"], axis=0), axis=1))
        print(f"  Distance traveled (pursuer): {dist_p:.1f}m")
        print(f"  Distance traveled (prey):    {dist_t:.1f}m")
    else:
        print(f"  Did not intercept in {config.t_max}s")
        print(f"  Final distance: {result['distances'][-1]:.1f}m")
    print(f"  Closest approach: {result['min_distance']:.2f}m at t={result['t_min_distance']:.2f}s")

    # Plot
    plot_metrics(result, pursuer, prey, config)
    animate_3d(result, pursuer, prey, config,
               anim_speed=anim_spd, save_gif=save)


# ===========================================================================
# Pre-configured scenarios
# ===========================================================================

def demo_scenario(scenario: int = 1):
    """Ready-made scenarios for quick demonstration."""

    scenarios = {
        1: {
            "desc": "Fast missile vs slow target in straight line",
            "pursuer": DroneConfig("Missile", np.array([0., 0., 100.]), 250, 100, "red"),
            "prey": DroneConfig("Target", np.array([500., 200., 150.]), 80, 50, "blue"),
            "config": SimConfig(strategy="pure_pursuit", prey_maneuver="straight"),
        },
        2: {
            "desc": "Proportional navigation vs zigzag evasion",
            "pursuer": DroneConfig("Missile", np.array([0., 0., 100.]), 200, 80, "red"),
            "prey": DroneConfig("Target", np.array([400., 100., 120.]), 150, 60, "dodgerblue"),
            "config": SimConfig(strategy="prop_nav", prey_maneuver="zigzag"),
        },
        3: {
            "desc": "Lead pursuit vs ascending spiral",
            "pursuer": DroneConfig("Missile", np.array([0., 0., 50.]), 180, 70, "darkred"),
            "prey": DroneConfig("Drone", np.array([300., 300., 100.]), 140, 80, "cyan"),
            "config": SimConfig(strategy="lead_pursuit", prey_maneuver="spiral"),
        },
        4: {
            "desc": "Slow pursuer vs prey with random maneuver",
            "pursuer": DroneConfig("Drone", np.array([0., 0., 200.]), 130, 60, "orangered"),
            "prey": DroneConfig("Target", np.array([400., 0., 180.]), 120, 90, "limegreen"),
            "config": SimConfig(strategy="prop_nav", prey_maneuver="random", t_max=40),
        },
    }

    if scenario not in scenarios:
        print("Invalid scenario!")
        return

    c = scenarios[scenario]
    print(f"\nScenario {scenario}: {c['desc']}")
    result = simulate(c["pursuer"], c["prey"], c["config"])

    if result["intercepted"]:
        print(f"Interception at t = {result['t_intercept']:.2f}s!")
    else:
        print(f"Did not intercept. Final distance: {result['distances'][-1]:.1f}m")
    print(f"Closest approach: {result['min_distance']:.2f}m at t={result['t_min_distance']:.2f}s")

    plot_metrics(result, c["pursuer"], c["prey"], c["config"])
    animate_3d(result, c["pursuer"], c["prey"], c["config"])


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='3D Pursuit Simulator')
    parser.add_argument('--demo', type=int, default=None, choices=[1, 2, 3, 4],
                        help='Run demo scenario (1-4)')
    parser.add_argument('--strategy', type=str, default=None,
                        choices=['pure_pursuit', 'prop_nav', 'lead_pursuit'])
    parser.add_argument('--maneuver', type=str, default=None,
                        choices=['straight', 'circular', 'zigzag', 'spiral', 'random'])
    parser.add_argument('--t-max', type=float, default=30.0)
    parser.add_argument('--save-gif', action='store_true')
    args = parser.parse_args()

    if args.demo:
        demo_scenario(args.demo)
    elif args.strategy and args.maneuver:
        # Quick non-interactive run
        pursuer = DroneConfig("Missile", np.array([0., 0., 100.]), 200, 80, "red", 100)
        prey = DroneConfig("Target", np.array([500., 200., 150.]), 120, 60, "blue", 80)
        config = SimConfig(strategy=args.strategy, prey_maneuver=args.maneuver,
                           t_max=args.t_max)
        result = simulate(pursuer, prey, config)
        if result["intercepted"]:
            print(f"Interception at t = {result['t_intercept']:.2f}s!")
        else:
            print(f"Did not intercept. Final distance: {result['distances'][-1]:.1f}m")
        print(f"Closest approach: {result['min_distance']:.2f}m at t={result['t_min_distance']:.2f}s")
        plot_metrics(result, pursuer, prey, config)
        animate_3d(result, pursuer, prey, config, save_gif=args.save_gif)
    else:
        # Interactive menu
        print("=" * 60)
        print("  3D PURSUIT SIMULATOR -- MISSILES/DRONES")
        print("=" * 60)
        print("  1. Custom configuration")
        print("  2. Demo: Fast missile vs slow target")
        print("  3. Demo: Prop. Navigation vs zigzag")
        print("  4. Demo: Lead Pursuit vs spiral")
        print("  5. Demo: Slow pursuer vs random maneuver")

        choice = input("\nChoose (1-5): ").strip() or "1"

        if choice == "1":
            menu()
        elif choice in ["2", "3", "4", "5"]:
            demo_scenario(int(choice) - 1)
        else:
            print("Invalid option!")

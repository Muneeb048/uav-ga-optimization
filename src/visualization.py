"""Plots, live UAV simulation window, and CSV outputs."""
import csv
import numpy as np
import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle


UAV_COLORS = ["#1f77b4", "#2ca02c", "#d62728"]
SENSOR_RADIUS = 2.0


def plot_fitness(history, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    generations = np.arange(1, len(history["best"]) + 1)
    ax.plot(generations, history["best"], label="Best fitness", linewidth=2)
    ax.plot(generations, history["average"], label="Average fitness", linewidth=1.5)
    ax.set(title="Genetic Algorithm Fitness Convergence",
           xlabel="Generation", ylabel="Fitness")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_routes(best, environment, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    _draw_map(ax, environment)
    for i, plan in enumerate(best.plans):
        route = np.vstack([environment.starts[i], plan.waypoints])
        ax.plot(route[:, 0], route[:, 1], marker="o", markersize=3,
                linewidth=1.8, color=UAV_COLORS[i % len(UAV_COLORS)],
                label=f"UAV {i+1}")
        ax.scatter(*environment.starts[i], marker="s", s=70,
                   color=UAV_COLORS[i % len(UAV_COLORS)], zorder=5)
    ax.set(title="Evolved UAV Flight Paths", xlabel="X grid coordinate",
           ylabel="Y grid coordinate", xlim=(-0.5, environment.size-0.5),
           ylim=(-0.5, environment.size-0.5), aspect="equal")
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_coverage(best, environment, out_path):
    coverage = np.zeros((environment.size, environment.size), dtype=float)
    xx, yy = np.ogrid[:environment.size, :environment.size]
    for i, plan in enumerate(best.plans):
        route = [environment.starts[i]] + [tuple(map(int, p)) for p in plan.waypoints]
        for j, (a, b) in enumerate(zip(route[:-1], route[1:])):
            if not plan.sensors[j]:
                continue
            for x, y in environment.line_cells(a, b):
                mask = (xx-x)**2 + (yy-y)**2 <= 2**2
                coverage[mask] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(coverage.T, origin="lower", cmap="viridis")
    bx, by = np.where(environment.no_fly)
    ax.scatter(bx, by, marker="s", s=10, color="white", alpha=0.7, label="No-fly zone")
    tx, ty = np.where(environment.priority > 0)
    ax.scatter(tx, ty, marker="*", s=65, color="red", edgecolor="white", label="Target")
    ax.set(title="UAV Sensor Coverage Heatmap", xlabel="X grid coordinate", ylabel="Y grid coordinate")
    fig.colorbar(im, ax=ax, label="Number of sensor passes")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_comparison(ga_history, random_curve, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x_random = np.arange(1, len(random_curve) + 1)
    ax.plot(x_random, random_curve, label="Random search (best so far)", alpha=0.85)
    ga_x = np.linspace(1, len(random_curve), len(ga_history["best"]))
    ax.plot(ga_x, ga_history["best"], label="GA (best so far)", linewidth=2)
    ax.set(title="GA vs Random Search (evaluation-budget view)",
           xlabel="Approximate candidate evaluation budget", ylabel="Best fitness so far")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_history_csv(history, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness", "average_fitness"])
        for i, (best, avg) in enumerate(zip(history["best"], history["average"]), start=1):
            writer.writerow([i, best, avg])


def plot_sensitivity(rows, out_path):
    labels = [r["label"] for r in rows]
    scores = [r["best_fitness"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, scores)
    ax.set(title="Hyperparameter Sensitivity Experiment",
           xlabel="Configuration", ylabel="Best final fitness")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _draw_map(ax, environment):
    ax.imshow(environment.hazard.T, origin="lower", cmap="YlOrRd",
              extent=(-0.5, environment.size - 0.5, -0.5, environment.size - 0.5),
              alpha=0.55, vmin=0, vmax=1)
    for x, y in zip(*np.where(environment.no_fly)):
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="black", zorder=2))
    tx, ty = np.where(environment.priority > 0)
    ax.scatter(tx, ty, marker="*", s=110, color="gold", edgecolor="black",
               linewidths=0.6, label="Priority target", zorder=4)


def _trajectories(best, environment):
    drones = []
    max_t = 0.0
    for i, plan in enumerate(best.plans):
        route = [environment.starts[i]] + [tuple(map(int, p)) for p in plan.waypoints]
        times = [0.0]
        xs = [float(route[0][0])]
        ys = [float(route[0][1])]
        sensors = [int(plan.sensors[0]) if len(plan.sensors) else 0]
        t = 0.0
        for j, (a, b) in enumerate(zip(route[:-1], route[1:])):
            dist = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            speed = float(np.clip(plan.speeds[j], 0.2, 1.0))
            duration = dist / (0.5 + speed) if dist > 0 else 0.05
            sensor = int(plan.sensors[j])
            steps = max(int(np.ceil(duration / 0.35)), 2)
            for k in range(1, steps + 1):
                frac = k / steps
                times.append(t + duration * frac)
                xs.append(a[0] + (b[0] - a[0]) * frac)
                ys.append(a[1] + (b[1] - a[1]) * frac)
                sensors.append(sensor)
            t += duration
        drones.append({
            "t": np.asarray(times, dtype=float),
            "x": np.asarray(xs, dtype=float),
            "y": np.asarray(ys, dtype=float),
            "s": np.asarray(sensors, dtype=float),
            "path": np.vstack([environment.starts[i], plan.waypoints]),
        })
        max_t = max(max_t, t)
    return drones, max(max_t, 1.0)


def _state_at(drone, t):
    times = drone["t"]
    t = float(np.clip(t, times[0], times[-1]))
    x = float(np.interp(t, times, drone["x"]))
    y = float(np.interp(t, times, drone["y"]))
    idx = int(np.searchsorted(times, t, side="right") - 1)
    idx = max(0, min(idx, len(drone["s"]) - 1))
    sensor_on = bool(drone["s"][idx] >= 0.5)
    return x, y, sensor_on


def animate_mission(best, environment, block=True):
    """Open a live window of the three UAVs flying the evolved routes."""
    drones, max_t = _trajectories(best, environment)
    n_frames = 240
    frame_times = np.linspace(0.0, max_t, n_frames)
    metrics = best.metrics or {}

    fig, ax = plt.subplots(figsize=(9.2, 8.2))
    try:
        fig.canvas.manager.set_window_title("UAV Genetic Algorithm — Live Simulation")
    except Exception:
        pass

    _draw_map(ax, environment)
    for i, drone in enumerate(drones):
        path = drone["path"]
        ax.plot(path[:, 0], path[:, 1], color=UAV_COLORS[i], linewidth=1.2,
                alpha=0.35, zorder=3)
        ax.scatter(*environment.starts[i], marker="s", s=55,
                   color=UAV_COLORS[i], zorder=5)

    trails = []
    markers = []
    sensors = []
    for i in range(len(drones)):
        color = UAV_COLORS[i % len(UAV_COLORS)]
        trail, = ax.plot([], [], color=color, linewidth=2.4, zorder=6, label=f"UAV {i+1}")
        marker, = ax.plot([], [], marker="o", markersize=11, color=color,
                          markeredgecolor="white", markeredgewidth=1.2, zorder=8)
        ring = Circle((0, 0), SENSOR_RADIUS, fill=False, linestyle="--",
                      linewidth=1.4, color=color, alpha=0.0, zorder=7)
        ax.add_patch(ring)
        trails.append(trail)
        markers.append(marker)
        sensors.append(ring)

    ax.set(xlim=(-0.5, environment.size - 0.5),
           ylim=(-0.5, environment.size - 0.5),
           aspect="equal",
           xlabel="X grid coordinate",
           ylabel="Y grid coordinate")
    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("Live UAV mission")
    fig.tight_layout()

    def _update(frame_idx):
        t = frame_times[frame_idx]
        for i, drone in enumerate(drones):
            x, y, sensor_on = _state_at(drone, t)
            mask = drone["t"] <= t
            trails[i].set_data(drone["x"][mask], drone["y"][mask])
            markers[i].set_data([x], [y])
            sensors[i].center = (x, y)
            sensors[i].set_alpha(0.85 if sensor_on else 0.0)
        fitness = metrics.get("fitness", float("nan"))
        coverage = metrics.get("coverage", float("nan"))
        title.set_text(
            f"Live UAV mission  |  t={t:6.1f}  |  fitness={fitness:.3f}  |  coverage={coverage:.1%}"
        )
        return trails + markers + sensors + [title]

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=40,
                         blit=False, repeat=True)
    fig._uav_animation = anim  # keep a reference so GC does not stop it
    print("\nA simulation window should now be open.")
    print("Watch the three UAVs fly. Close that window to continue.")
    if block:
        plt.show()
    return anim


def show_result_plots(image_paths):
    """Show the saved result images in one window so they are not easy to miss."""
    existing = [(title, path) for title, path in image_paths]
    if not existing:
        return
    cols = 2
    rows = int(np.ceil(len(existing) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 5.5 * rows))
    try:
        fig.canvas.manager.set_window_title("UAV Genetic Algorithm — Result Plots")
    except Exception:
        pass
    axes = np.atleast_1d(axes).ravel()
    for ax, (title, path) in zip(axes, existing):
        image = plt.imread(path)
        ax.imshow(image)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(existing):]:
        ax.axis("off")
    fig.tight_layout()
    print("Result plots window is open. Close it to finish.")
    plt.show()

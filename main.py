"""Run the UAV Genetic Algorithm assignment simulation."""
import os
import sys
import json
import csv
import subprocess


def _ensure_packages():
    """Install numpy/matplotlib automatically so one command is enough."""
    try:
        import numpy  # noqa: F401
        import matplotlib  # noqa: F401
        return
    except ImportError:
        pass
    req = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    print("Required packages are missing. Installing numpy and matplotlib...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--user", "-r", req])
    os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_packages()

from src.environment import Environment  # noqa: E402
from src.genetic_algorithm import GeneticAlgorithm, random_search  # noqa: E402
from src.visualization import (  # noqa: E402
    plot_fitness, plot_routes, plot_coverage, plot_comparison,
    save_history_csv, plot_sensitivity, animate_mission, show_result_plots
)

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)


def run_sensitivity(base_seed=200):
    """Small one-factor-at-a-time experiments for report evidence."""
    rows = []
    configs = [
        ("pop=10", 10, 5, 0.15, 3),
        ("pop=15", 15, 5, 0.15, 3),
        ("pop=20", 20, 5, 0.15, 3),
        ("mut=0.05", 12, 5, 0.05, 3),
        ("mut=0.15", 12, 5, 0.15, 3),
        ("mut=0.30", 12, 5, 0.30, 3),
        ("tour=2", 12, 5, 0.15, 2),
        ("tour=3", 12, 5, 0.15, 3),
        ("tour=5", 12, 5, 0.15, 5),
    ]
    for idx, (label, pop, gens, mut, tour) in enumerate(configs):
        env = Environment(seed=7)
        ga = GeneticAlgorithm(env, population_size=pop, generations=gens,
                              mutation_rate=mut, tournament_size=tour,
                              seed=base_seed + idx)
        best = ga.run(verbose=False)
        rows.append({
            "label": label, "population_size": pop, "generations": gens,
            "mutation_rate": mut, "tournament_size": tour,
            "best_fitness": round(float(best.fitness), 6),
            "coverage": round(float(best.metrics["coverage"]), 6),
            "energy": round(float(best.metrics["energy"]), 4),
            "time": round(float(best.metrics["time"]), 4),
            "hazard": round(float(best.metrics["hazard"]), 4),
            "reported_feasible": bool(best.metrics["feasible"])
        })
        print(f"Sensitivity {label}: best={best.fitness:.4f}")
    with open(os.path.join(RESULTS, "sensitivity_results.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_sensitivity(rows, os.path.join(RESULTS, "sensitivity.png"))
    return rows


def main():
    env = Environment(size=50, num_drones=3, num_waypoints=8, seed=7)
    ga = GeneticAlgorithm(env, population_size=30, generations=50,
                          mutation_rate=0.15, crossover_rate=0.8,
                          tournament_size=3, elite_count=2, seed=21)
    best = ga.run(verbose=True)

    fitness_path = os.path.join(RESULTS, "fitness_curve.png")
    routes_path = os.path.join(RESULTS, "uav_paths.png")
    coverage_path = os.path.join(RESULTS, "coverage_heatmap.png")
    comparison_path = os.path.join(RESULTS, "comparison.png")

    plot_fitness(ga.history, fitness_path)
    plot_routes(best, env, routes_path)
    plot_coverage(best, env, coverage_path)
    save_history_csv(ga.history, os.path.join(RESULTS, "fitness_history.csv"))

    # Show the live flight immediately so the simulation is not delayed
    # by the random-search baseline.
    animate_mission(best, env)

    random_best, random_curve, _ = random_search(env, evaluations=2536, seed=99)
    plot_comparison(ga.history, random_curve, comparison_path)

    summary = {
        "ga_best_fitness": float(best.fitness),
        "ga_metrics": best.metrics,
        "random_search_best_fitness": float(random_best.fitness),
        "ga_evaluations": int(ga.evaluations),
        "random_search_evaluations": 2536,
        "note": "Treat results as simulation output, not proof of global optimality."
    }
    with open(os.path.join(RESULTS, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\nRandom-search best fitness:", round(random_best.fitness, 4))
    print("Saved plots and result files in:", RESULTS)

    show_result_plots([
        ("Fitness convergence", fitness_path),
        ("Evolved UAV paths", routes_path),
        ("Coverage heatmap", coverage_path),
        ("GA vs random search", comparison_path),
    ])

    if os.environ.get("RUN_SENSITIVITY", "0") == "1":
        run_sensitivity()


if __name__ == "__main__":
    main()

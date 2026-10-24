"""Normalized multi-objective fitness evaluation."""
import numpy as np
from .constraints import repair_chromosome

WEIGHTS = {
    "coverage": 0.35,
    "detection": 0.25,
    "energy": 0.15,
    "time": 0.10,
    "hazard": 0.15,
}


def evaluate(chromosome, environment, repair=True):
    if repair:
        repair_chromosome(chromosome, environment)

    target_coords = np.argwhere(environment.priority > 0)
    target_weights = np.asarray([environment.priority[x, y] for x, y in target_coords])
    target_total = float(target_weights.sum()) or 1.0
    detected = np.zeros(len(target_coords), dtype=bool)
    total_distance = total_energy = total_time = hazard_exposure = 0.0

    for drone_idx, plan in enumerate(chromosome.plans):
        route = [environment.starts[drone_idx]] + [tuple(map(int, p)) for p in plan.waypoints]
        for j, (a, b) in enumerate(zip(route[:-1], route[1:])):
            d = float(np.hypot(b[0]-a[0], b[1]-a[1]))
            speed = float(np.clip(plan.speeds[j], 0.2, 1.0))
            sensor = int(plan.sensors[j])
            total_distance += d
            total_energy += d * (1.0 + 0.35 * speed**2 + 0.12 * sensor)
            total_time += d / (0.5 + speed)
            cells = environment.line_cells(a, b)
            for x, y in cells:
                if environment.in_bounds((x, y)):
                    hazard_exposure += float(environment.hazard[x, y])
                    if sensor and len(target_coords):
                        dist2 = (target_coords[:, 0] - x) ** 2 + (target_coords[:, 1] - y) ** 2
                        detected |= dist2 <= 4

    detected_priority = float(target_weights[detected].sum())
    coverage = detected_priority / target_total
    # In this model, detection is sensor-enabled weighted target coverage.
    detection = coverage
    energy_norm = min(total_energy / (3 * 230.0), 1.0)
    time_norm = min(total_time / (3 * 300.0), 1.0)
    hazard_norm = min(hazard_exposure / (3 * environment.num_waypoints * 8.0), 1.0)

    penalty = 0.0
    for drone_idx, plan in enumerate(chromosome.plans):
        route = [environment.starts[drone_idx]] + [tuple(map(int, p)) for p in plan.waypoints]
        dist = sum(np.hypot(route[k+1][0]-route[k][0], route[k+1][1]-route[k][1])
                   for k in range(len(route)-1))
        if dist > 180.0:
            penalty += 0.15 + 0.001 * (dist - 180.0)
        for a, b in zip(route[:-1], route[1:]):
            if any(environment.is_blocked(c) for c in environment.line_cells(a, b)[1:]):
                penalty += 0.25

    score = (WEIGHTS["coverage"] * coverage
             + WEIGHTS["detection"] * detection
             - WEIGHTS["energy"] * energy_norm
             - WEIGHTS["time"] * time_norm
             - WEIGHTS["hazard"] * hazard_norm
             - penalty)
    chromosome.fitness = float(score)
    chromosome.metrics = {
        "fitness": float(score), "coverage": float(coverage),
        "detection": float(detection), "energy": float(total_energy),
        "energy_normalized": float(energy_norm), "time": float(total_time),
        "time_normalized": float(time_norm), "hazard": float(hazard_exposure),
        "hazard_normalized": float(hazard_norm), "distance": float(total_distance),
        "feasible": bool(penalty == 0.0), "penalty": float(penalty)
    }
    return chromosome.fitness

"""Constraint repair and feasibility checks."""
import numpy as np


def repair_chromosome(chromosome, environment):
    for drone_idx, plan in enumerate(chromosome.plans):
        plan.waypoints = np.asarray(
            [environment.safe_point(p) for p in plan.waypoints], dtype=int
        )
        route = environment.repair_route(environment.starts[drone_idx], plan.waypoints)
        plan.waypoints = np.asarray(route[1:], dtype=int)
        plan.speeds = np.clip(plan.speeds, 0.2, 1.0)
        plan.sensors = np.asarray(plan.sensors, dtype=int)
        plan.sensors = np.clip(plan.sensors, 0, 1)


def validate_chromosome(chromosome, environment, max_range=180.0, battery_budget=230.0):
    """Return (feasible, list of readable violations)."""
    issues = []
    for i, plan in enumerate(chromosome.plans):
        route = [environment.starts[i]] + [tuple(map(int, p)) for p in plan.waypoints]
        distance = sum(
            np.hypot(route[j+1][0]-route[j][0], route[j+1][1]-route[j][1])
            for j in range(len(route)-1)
        )
        for a, b in zip(route[:-1], route[1:]):
            if any(environment.is_blocked(c) for c in environment.line_cells(a, b)[1:]):
                issues.append(f"UAV {i+1}: route intersects a no-fly cell")
                break
        if distance > max_range:
            issues.append(f"UAV {i+1}: range {distance:.1f} exceeds {max_range:.1f}")
        # A lightweight energy estimate, consistent with fitness calculation.
        energy = 0.0
        for j in range(len(plan.waypoints)):
            a = route[j]
            b = route[j+1]
            d = np.hypot(b[0]-a[0], b[1]-a[1])
            energy += d * (1 + 0.35 * plan.speeds[j]**2 + 0.12 * plan.sensors[j])
        if energy > battery_budget:
            issues.append(f"UAV {i+1}: energy {energy:.1f} exceeds {battery_budget:.1f}")
    return len(issues) == 0, issues

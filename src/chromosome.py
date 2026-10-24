"""Chromosome representation: waypoint coordinates, speed and sensor schedule."""
from dataclasses import dataclass
import numpy as np


@dataclass
class DronePlan:
    waypoints: np.ndarray  # shape: (num_waypoints, 2), integer grid coordinates
    speeds: np.ndarray     # normalized speeds in [0.2, 1.0]
    sensors: np.ndarray    # binary sensor activation values

    def copy(self):
        return DronePlan(self.waypoints.copy(), self.speeds.copy(), self.sensors.copy())


@dataclass
class Chromosome:
    plans: list
    fitness: float = float("-inf")
    metrics: dict | None = None

    def copy(self):
        return Chromosome([p.copy() for p in self.plans], self.fitness,
                          None if self.metrics is None else dict(self.metrics))

    @staticmethod
    def random(environment, rng):
        plans = []
        for start in environment.starts:
            points = []
            for _ in range(environment.num_waypoints):
                points.append([
                    rng.integers(0, environment.size),
                    rng.integers(0, environment.size)
                ])
            plans.append(DronePlan(
                waypoints=np.asarray(points, dtype=int),
                speeds=rng.uniform(0.2, 1.0, environment.num_waypoints),
                sensors=rng.integers(0, 2, environment.num_waypoints, dtype=int)
            ))
        return Chromosome(plans)

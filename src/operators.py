"""Genetic operators customized for multi-segment UAV chromosomes."""
import numpy as np
from .chromosome import Chromosome
from .constraints import repair_chromosome
from .fitness import evaluate


def tournament_selection(population, rng, tournament_size=3):
    contestants = rng.choice(population, size=min(tournament_size, len(population)), replace=False)
    return max(contestants, key=lambda c: c.fitness)


def crossover(parent_a, parent_b, rng, environment):
    """Segment crossover: swap a waypoint interval and corresponding settings."""
    child_a, child_b = parent_a.copy(), parent_b.copy()
    for pa, pb, ca, cb in zip(parent_a.plans, parent_b.plans,
                              child_a.plans, child_b.plans):
        n = len(pa.waypoints)
        if n >= 2:
            left, right = sorted(rng.choice(np.arange(n), size=2, replace=False))
            right += 1
            ca.waypoints[left:right], cb.waypoints[left:right] = (
                pb.waypoints[left:right].copy(), pa.waypoints[left:right].copy()
            )
            ca.speeds[left:right], cb.speeds[left:right] = (
                pb.speeds[left:right].copy(), pa.speeds[left:right].copy()
            )
            ca.sensors[left:right], cb.sensors[left:right] = (
                pb.sensors[left:right].copy(), pa.sensors[left:right].copy()
            )
    repair_chromosome(child_a, environment)
    repair_chromosome(child_b, environment)
    evaluate(child_a, environment, repair=False)
    evaluate(child_b, environment, repair=False)
    return child_a, child_b


def mutate(chromosome, rng, environment, mutation_rate=0.15):
    for plan in chromosome.plans:
        for j in range(len(plan.waypoints)):
            if rng.random() < mutation_rate:
                # Local coordinate mutation promotes spatial exploration.
                plan.waypoints[j] += rng.integers(-5, 6, size=2)
            if rng.random() < mutation_rate:
                plan.speeds[j] += rng.normal(0, 0.15)
            if rng.random() < mutation_rate:
                plan.sensors[j] = 1 - int(plan.sensors[j])
    repair_chromosome(chromosome, environment)
    evaluate(chromosome, environment, repair=False)
    return chromosome

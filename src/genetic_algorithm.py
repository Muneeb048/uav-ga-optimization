"""Object-oriented genetic algorithm and random-search baseline."""
import time
import numpy as np
from .chromosome import Chromosome
from .fitness import evaluate
from .operators import tournament_selection, crossover, mutate
from .constraints import repair_chromosome, validate_chromosome


class GeneticAlgorithm:
    def __init__(self, environment, population_size=50, generations=100,
                 mutation_rate=0.15, crossover_rate=0.8, tournament_size=3,
                 elite_count=2, seed=21):
        self.env = environment
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.elite_count = elite_count
        self.rng = np.random.default_rng(seed)
        self.history = {"best": [], "average": []}
        self.evaluations = 0

    def initialize(self):
        pop = [Chromosome.random(self.env, self.rng) for _ in range(self.population_size)]
        for c in pop:
            repair_chromosome(c, self.env)
            evaluate(c, self.env, repair=False)
        self.evaluations += len(pop)
        return pop

    def run(self, verbose=True):
        started = time.perf_counter()
        population = self.initialize()
        best = max(population, key=lambda c: c.fitness).copy()

        for generation in range(self.generations):
            population.sort(key=lambda c: c.fitness, reverse=True)
            current_best = population[0]
            if current_best.fitness > best.fitness:
                best = current_best.copy()
            self.history["best"].append(best.fitness)
            self.history["average"].append(float(np.mean([c.fitness for c in population])))

            next_pop = [c.copy() for c in population[:self.elite_count]]
            while len(next_pop) < self.population_size:
                p1 = tournament_selection(population, self.rng, self.tournament_size)
                p2 = tournament_selection(population, self.rng, self.tournament_size)
                if self.rng.random() < self.crossover_rate:
                    c1, c2 = crossover(p1, p2, self.rng, self.env)
                    self.evaluations += 2
                else:
                    c1, c2 = p1.copy(), p2.copy()
                mutate(c1, self.rng, self.env, self.mutation_rate)
                self.evaluations += 1
                next_pop.append(c1)
                if len(next_pop) < self.population_size:
                    mutate(c2, self.rng, self.env, self.mutation_rate)
                    self.evaluations += 1
                    next_pop.append(c2)
            population = next_pop
            if verbose and (generation == 0 or (generation + 1) % 10 == 0):
                print(f"Generation {generation+1:3d}/{self.generations}: "
                      f"best={best.fitness:.4f}, avg={self.history['average'][-1]:.4f}")

        population.sort(key=lambda c: c.fitness, reverse=True)
        if population[0].fitness > best.fitness:
            best = population[0].copy()
        elapsed = time.perf_counter() - started
        feasible, issues = validate_chromosome(best, self.env)
        print(f"\nGA finished in {elapsed:.2f}s; evaluations={self.evaluations}")
        print(f"Best fitness: {best.fitness:.4f}")
        print(f"Metrics: {best.metrics}")
        print(f"Feasible under strict range/battery checks: {feasible}")
        if issues:
            print("Validation notes:", "; ".join(issues))
        return best


def random_search(environment, evaluations=5000, seed=99):
    """Random-search baseline using the same fitness evaluator."""
    rng = np.random.default_rng(seed)
    best = None
    best_so_far = []
    values = []
    for _ in range(evaluations):
        candidate = Chromosome.random(environment, rng)
        repair_chromosome(candidate, environment)
        evaluate(candidate, environment, repair=False)
        values.append(candidate.fitness)
        if best is None or candidate.fitness > best.fitness:
            best = candidate.copy()
        best_so_far.append(best.fitness)
    return best, np.asarray(best_so_far), np.asarray(values)

# UAV Disaster Surveillance Strategy Engine Using a Genetic Algorithm

**Team members:** [Member 1], [Member 2], [Member 3]  
**Course:** Artificial Intelligence  
**Date:** [Submission date]

## 1. Introduction and problem formulation
This project models a disaster-surveillance area as a 50 x 50 grid. Three UAVs travel from fixed starting positions to inspect priority targets. The environment includes no-fly cells and graded hazard zones. The objective is to produce routes and sensor schedules that balance mission coverage and detection against energy expenditure, flight time, and hazard exposure.

## 2. Chromosome design
A chromosome contains one plan per UAV. Each plan has a fixed-length sequence of (x, y) waypoints, normalized speed values, and binary sensor activation bits. This hybrid encoding was chosen because coordinates and speed are numeric variables, while sensor activation is binary. Segment crossover exchanges corresponding waypoint, speed, and sensor intervals. Mutation perturbs coordinates, adjusts speeds, or flips sensor bits.

## 3. Fitness function and constraint handling
The evaluator uses:

F = 0.35C + 0.25D - 0.15E - 0.10T - 0.15H - P

C is weighted coverage, D is weighted sensor-enabled target detection, E is normalized energy expenditure, T is normalized flight time, H is normalized hazard exposure, and P penalizes range and no-fly violations. Positive weights prioritize mission success; negative weights discourage costly or dangerous routes. Coordinates are repaired to the grid and away from blocked cells. Route segments are checked for no-fly intersections. The current repair method is heuristic, so final feasibility checks are reported separately.

## 4. Experimental setup
The default run uses a population of 50, 100 generations, crossover probability 0.8, mutation rate 0.15, tournament size 3, and elitism of 2. The random-search baseline uses the same environment and fitness evaluator. Report the actual candidate-evaluation budgets, random seeds, Python version, and runtime from your run.

## 5. Results and discussion
Insert the generated `fitness_curve.png`, `uav_paths.png`, `coverage_heatmap.png`, and `comparison.png`. Report the best GA fitness, random-search best fitness, coverage, energy, time, hazard exposure, and feasibility findings from `results/summary.json`.

Do not claim a GA advantage unless supported by the results. Discuss whether the best fitness improved over generations, which target cells were covered, and whether the final route respected all operational constraints.

## 6. Hyperparameter sensitivity analysis
Use `results/sensitivity_results.csv` and `results/sensitivity.png`. Compare population sizes 20, 50, and 100; mutation rates 0.05, 0.15, and 0.30; and tournament sizes 2, 3, and 5. For each, discuss best fitness and runtime. Explain the trade-off: larger populations use more evaluations per generation; low mutation can reduce exploration; high mutation can disrupt good routes; larger tournaments increase selection pressure and may reduce diversity.

Fill in measured values:
- Best population-size setting: [value and evidence]
- Best mutation-rate setting: [value and evidence]
- Best tournament-size setting: [value and evidence]

## 7. Conclusion and limitations
Summarize the best configuration supported by the experiment and compare it with random search. This is a stochastic heuristic and does not prove a globally optimal solution. The model uses simplified energy, time, hazard, and sensor assumptions and is intended for coursework rather than real UAV deployment.

## References
[List any course notes, textbooks, or external references actually consulted.]

# UAV Disaster Surveillance Strategy Engine Using a Genetic Algorithm

**Course:** Artificial Intelligence  
**Team members:** [Enter all three names]  
**Submission date:** [Enter date]

## 1. Introduction and problem formulation
This project implements an educational strategy engine for a fleet of three simulated unmanned aerial vehicles (UAVs) surveying a 50 × 50 grid. The environment contains priority targets, no-fly zones, and graded hazard regions. The task is multi-objective: improve priority-target coverage and detection while reducing energy expenditure, travel time, and hazard exposure.

The solution uses a Genetic Algorithm (GA), a population-based stochastic optimization method. Each candidate solution encodes all three drone plans. The algorithm repeatedly selects higher-scoring plans, recombines them, mutates some genes, repairs routes, and preserves elite candidates. It is a heuristic and does not guarantee a globally optimal route.

## 2. Chromosome design
Each chromosome contains one plan per UAV. A plan contains (1) eight waypoint coordinate pairs, (2) eight normalized speed values in [0.2, 1.0], and (3) eight binary sensor activation values. Coordinates are spatial genes, speed values are real-valued genes, and sensor states are binary genes. This hybrid representation directly models trajectory, speed, and sensor schedule decisions.

Crossover swaps a segment of waypoint coordinates together with the corresponding speed and sensor genes. Mutation can perturb a waypoint by a small random offset, adjust speed, or flip a sensor bit. The chromosome is repaired after genetic operations to clip coordinates to the grid and avoid blocked route segments.

## 3. Fitness function and constraints
The implemented weighted fitness function is:

**F = 0.35C + 0.25D − 0.15E − 0.10T − 0.15H − P**

Here, C is weighted target coverage, D is sensor-enabled target detection, E is normalized energy, T is normalized flight time, H is normalized hazard exposure, and P is a penalty for route-range or no-fly violations. All cost metrics are normalized or clipped against fixed reference budgets. Coverage and detection receive the largest positive weights because completing the mission is the primary objective; energy, time, and hazard exposure reduce the score.

The model uses an illustrative energy estimate based on segment distance, speed, and sensor activation. Time is approximated from distance and speed. Hazard exposure accumulates hazard severity along sampled route cells. These assumptions are suitable for a coursework simulation, not real flight control. Route repair is heuristic; the implementation separately validates the best route against no-fly, range, and battery constraints.

## 4. Experimental setup
The main experiment used a population of 30 chromosomes, 50 generations, crossover probability 0.8, mutation rate 0.15, tournament size 3, and elitism of 2. A fixed random seed was used for reproducibility. The GA reported 2536 candidate evaluations. Random search used the same environment and fitness evaluator with 2536 candidate evaluations.

The main run took approximately 12.8 seconds in the execution environment used for testing. Hardware, Python version, and package versions can affect runtime. The sensitivity experiment used short 5-generation runs to keep execution practical; these results are preliminary and should not be treated as statistically conclusive.

## 5. Results and discussion
The main GA run reached a best fitness of **0.4293**, compared with **0.3559** for random search under the same candidate-evaluation count. In this run, GA's best fitness was higher than the random-search baseline. The best chromosome achieved weighted coverage/detection of **0.956**, estimated total energy **422.84**, estimated total time **358.57**, and hazard exposure **15.75**. The route validator reported no range, battery, or no-fly violations for the best candidate.

The fitness convergence plot shows the best score improving during evolution and later flattening, while average fitness varies as mutation and crossover introduce new candidates. The route map shows each UAV path over the hazard map, with targets and no-fly cells. The coverage heatmap illustrates the cells observed by sensor-active route segments. The result is one good candidate for the modeled environment; another random seed may produce a different solution.

### Main result table
| Metric | Result |
|---|---:|
| GA best fitness | 0.4293 |
| Random-search best fitness | 0.3559 |
| Weighted coverage / detection | 0.956 |
| Estimated energy | 422.84 |
| Estimated flight time | 358.57 |
| Hazard exposure | 15.75 |
| GA candidate evaluations | 2536 |
| Random-search evaluations | 2536 |

## 6. Hyperparameter sensitivity analysis
One-factor-at-a-time experiments varied population size, mutation rate, and tournament size. Each short run used five generations and a fixed seed per configuration. Results therefore indicate sensitivity in this particular run, not universal parameter rankings.

- **Population size:** the best score among tested settings occurred at pop=15 (fitness 0.3560). A larger population samples more candidates per generation but increases computation; a very small population can lose diversity.
- **Mutation rate:** the best tested setting was mut=0.15 (fitness 0.3766). Low mutation may limit exploration; high mutation can disrupt good waypoint sequences. The best value depends on the map and random seed.
- **Selection pressure:** the best tested setting was tour=5 (fitness 0.3556). Larger tournament size favors top candidates more strongly and can accelerate convergence, but may reduce diversity.

For a stronger study, repeat each configuration across several seeds and compare mean fitness, standard deviation, feasibility rate, and runtime. Since these experiments are short, avoid overgeneralizing from the highest single score.

## 7. Conclusion and limitations
The project demonstrates a modular, object-oriented GA for a simplified multi-UAV surveillance task. The hybrid chromosome encodes waypoint, speed, and sensor decisions; the weighted fitness balances mission performance and costs; and repair/validation routines address basic constraints. In the tested run, the GA exceeded the random-search score under the same evaluation budget.

Limitations include simplified energy and flight-time models, a static environment, heuristic route repair, and a small fixed waypoint count. Real UAV use would require validated dynamics, continuous collision checking, battery calibration, dynamic replanning, and formal safety assurance. The simulation should be interpreted as an AI coursework prototype rather than a deployment-ready planner.

## Figures
Insert or retain these generated outputs from the `results/` directory:
- `fitness_curve.png` — best and average fitness by generation
- `uav_paths.png` — evolved UAV paths
- `coverage_heatmap.png` — sensor coverage
- `comparison.png` — GA versus random search
- `sensitivity.png` — hyperparameter comparison

## References
Add any course slides, textbook chapters, or external references actually consulted.

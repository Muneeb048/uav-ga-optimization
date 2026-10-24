# UAV Disaster Surveillance Optimization Using a Genetic Algorithm

## Overview
This educational simulation uses a Genetic Algorithm (GA) to optimize routes for three UAVs on a 50 x 50 grid. It models high-priority targets, hazard zones, no-fly zones, normalized speed settings, and binary sensor activation schedules.

The project demonstrates:
- Object-oriented chromosome and UAV-plan representation.
- Population initialization, tournament selection, segment crossover, mutation, elitism, and termination.
- A weighted fitness function balancing target coverage/detection against energy, flight time, and hazard exposure.
- Route repair and constraint validation.
- Fitness convergence, evolved path, coverage heatmap, GA-vs-random-search plots, and CSV result files.
- Optional hyperparameter sensitivity experiments.

This is a simplified educational model, not a flight-certified planner. It does not model real aerodynamics, communication failures, dynamic obstacles, or real sensor physics.

## How to run (one command)
Install Python 3 from python.org and tick **Add python.exe to PATH**. Then open this folder and run:

```bat
main
```

That is the same as double-clicking `main.bat`. The first run installs NumPy and Matplotlib automatically, then:

1. Evolves UAV routes in the terminal.
2. Opens a **live simulation window** of three UAVs flying the best routes.
3. After you close that window, opens the result plots (fitness, paths, coverage, comparison).

You do **not** need a virtual environment, `pip`, or extra PowerShell commands.

Equivalent command:

```bat
python main.py
```

Outputs are saved to `results/`:
- `fitness_curve.png`: best and average fitness per generation
- `uav_paths.png`: final routes with hazards, targets, and no-fly cells
- `coverage_heatmap.png`: sensor-pass heatmap
- `comparison.png`: GA and random-search best-so-far curves
- `fitness_history.csv`: fitness history
- `summary.json`: metrics and baseline summary

## Run sensitivity analysis
This experiment tests population sizes 10/15/20, mutation rates 0.05/0.15/0.30, and tournament sizes 2/3/5.
```powershell
$env:RUN_SENSITIVITY="1"
python main.py
```
On Command Prompt:
```bat
set RUN_SENSITIVITY=1
python main.py
```
Sensitivity results are saved to `results/sensitivity_results.csv` and `results/sensitivity.png`. This performs additional runs and may take longer.

## Fitness function
The evaluator uses:
`F = 0.35*C + 0.25*D - 0.15*E - 0.10*T - 0.15*H - P`

where C is weighted coverage, D is detection rate, E is normalized energy, T is normalized flight time, H is normalized hazard exposure, and P is a constraint penalty. Metrics are normalized or clipped to reference budgets to make weights interpretable.

## Chromosome
Each chromosome contains a plan for each UAV. Each plan includes:
1. Real/integer-valued waypoint coordinates (x, y)
2. Normalized speed values in [0.2, 1.0]
3. Binary sensor activation values

## Constraints and limitations
The code repairs out-of-bounds/blocked waypoints and checks line segments for no-fly-zone intersections. Range and battery limits are evaluated separately. Since route repair is heuristic, inspect the printed feasibility report and discuss any violations honestly in the report. For safety-critical use, replace the simple repair routine with a robust grid path planner and validate every segment after all operators.

## Reproducibility
Random seeds are set in `main.py` and in the environment/GA constructors. Changing seeds can change evolved routes and metrics.

## Suggested report experiments
Run the default experiment, then sensitivity analysis. Use actual values from `results/summary.json`, `results/sensitivity_results.csv`, and the generated plots. Do not claim the GA beats random search unless the results support that conclusion.

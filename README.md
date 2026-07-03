# Large-scale workflow placement in serverless computing using integer nonlinear programming
Under revision for IEEE Access
## How to use/Evaluate the results

a) Install the necessary dependencies:

```pip install -r requirements.txt```

b) Simple example:
Run the placement problem

```placement.py --num_run=0```

Calculate the costs through simulation by running

```selection_simulation.py --num_run=0```

c) Scaling experiments:

For the first comparison against the heuristic run:

```scaling_experiments.py```

In the command line run

```run_time_case_study_heuristic.bat```

To visualize or show the data run

```visualization_time_heuristic.py num_run=123```

Change "heuristic" in paramters_1.json to false

For the second comparison against the centralized approach run:

```scaling_experiments.py```

In the command line run

```run_time_case_study.bat```

To visualize or show the data run

```visualization_time.py num_run=126```

d) Sensitivity to weights

To generate the different parameter files run

```generate_different_weights.py```

In the command line run

```run_case_study.bat```

To generate the images of the Paerto fronts run

```visualization.py --num_run=2```

e) Sensitivity to uncertain parameters (demand, running time, latency, RAM)

The placement is optimized against the assumed parameter values. To test how
robust a computed placement is when the actual values differ, first run the
placement for a run as usual:

```placement.py --num_run=N```

Then configure the `sensitivity_config` block in `settings/parameters_N.json`.
Each uncertain parameter (`request_rate`, `time`, `speedup`, `latency`, `ram`)
can independently be set to `"mode": "noise"` (relative noise of `noise_pct`
around the baseline) or `"mode": "scale"` (a fixed `factor`), or `"off"`. Set
`num_scenarios` and a `seed` for reproducibility.

Generate the scenario input files and evaluate the fixed placement against each:

```generate_sensitivity_scenarios.py --num_run=N```

```run_sensitivity.py --num_run=N```

The scenario inputs are written to `sensitivity/run_N/scenario_<i>/` (only the
perturbed cloud/workflow requirement files; everything else is inherited from
the baseline run). The placement solution files (`placement/run_N/opt_*.json`)
are never modified. Latency is perturbed via the `L` matrix that `placement.py`
writes into the cloud requirement files after solving, so the placement
decision stays fixed. Aggregated results are written to
`results/run_N/sensitivity/summary.json` and `summary.csv`.

To attribute the effect of each individual uncertainty, run a one-factor-at-a-
time scale sweep (each parameter is varied alone across a set of scale factors):

```run_scale_sweep.py --num_run=N```

This generates its own scenarios under `sensitivity/run_N/scale_sweep/` and
writes `results/run_N/sensitivity/scale_sweep_summary.json` (optional
`--factors=0.5,1.0,2.0` and `--params=latency,time`).

The sweep evaluates the *fixed nominal* placement under wrong parameters. To
also see how it compares against re-planning, add `--adapt`:

```run_scale_sweep.py --num_run=N --adapt```

For every sweep point this additionally re-solves the placement (and the
centralized/heuristic comparison, per the run's settings) with the parameters
adapted to the perturbed values, inside a self-contained sandbox
`sensitivity/run_N/scale_sweep/<param>/factor_<tag>/adapted/` (mirroring the
usual `settings/cloud/workflows/placement/results` layout, so the baseline run
is never touched). The adapted costs are added to the summary as
`adapted_overall_cost` / `adapted_overall_cost_comparison`. Note: this runs one
full placement solve per sweep point (with a centralized comparison that is a
full-architecture MIQP each time) and stores one full run folder per point, so
use it with a small `--factors`/`--params` selection on large runs. The
adapted run at factor 1.0 reproduces the baseline exactly (sanity check).

To plot the study run

```visualization_sensitivity.py --num_run=N```

which saves a tornado chart, per-parameter response curves (with the adapted
placement overlaid as dashed lines when `--adapt` data exists), an
adaptation-offset chart (absolute and relative regret of keeping the nominal
placement instead of re-planning), a distribution overview of the combined
study, and a regression-based estimate of the individual effects to
`results/run_N/`. Each figure covers both the decomposed and the comparison
solution, and the titles state whether the comparison is the centralized
optimization or the heuristic (read from `settings/parameters_N.json`).

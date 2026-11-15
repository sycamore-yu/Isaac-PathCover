# SUPER trajectory oracle

The reference is SUPER commit `2ad3419c127a617c6d7df6925e81a14175a9c096`.
`regenerate.py` extracts the original Exp/Backup objective bodies, optimizer
variable structs and yaw methods. It links the upstream MINCO, banded system,
L-BFGS, polynomial trajectory and interpolation sources. The only replacement
headers remove ROS/color logging dependencies. `driver.cpp` supplies fixed input
and serializes the original results. No C++ code is used by the planner runtime.

From the repository root:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/reference/super_trajectory/regenerate.py \
  --source /home/tong/tongworkspace/reference_repos/SUPER
OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/reference/super_trajectory/regenerate_hotstart.py \
  --source /home/tong/tongworkspace/reference_repos/SUPER
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q \
  tests/test_super_trajectory.py tests/test_super_trajectory_hotstart.py \
  tests/test_trajectory_reference.py
```

The build needs `g++` and Eigen headers. The generated executable and extracted
sources live in ignored `.reference/super_trajectory`. `--no-build` reuses that
executable; it must be rebuilt after changes to the upstream sources or driver.
The fixture records source SHA-256 hashes and the exact textual input accepted
by the C++ driver, including polytope vertices and initial decision variables.
`deserialize()` makes regression checks independent of a different LP optimum
or vertex enumeration order on a later machine.

The six cases cover both Exp position parameterizations, snap energy on/off,
both Backup time parameterizations, active jerk/body-rate/thrust penalties,
analytic time and boundary-state gradients, complete L-BFGS convergence and
3rd-, 5th- and 7th-degree yaw interpolation (six cases per degree).
Fixed objective and derivative checks have strict numerical
tolerances. Full solves also compare the stopping status, cost and 101 trajectory
samples; their tolerance accounts for different floating-point iterates in a
nonconvex objective without changing the source stopping criteria.

`regenerate_hotstart.py` additionally compiles the original public explicit
`init_times/init_points` overloads, geometry setup and acceptance checks. Four
input groups contain five calls: two Exp parameterizations, nonuniform Backup,
and two consecutive uniform Backup calls after an ordinary heuristic call. The
original C++ object carries `total_time` between calls. Python receives the same
recorded inherited value explicitly, so each comparison uses the same problem;
this does not compare two independently evolving sequences. The wrapper records
objective inputs and outputs without changing arithmetic in the original body.

The hot-start fixture also cross-evaluates both final points in the original C++
and Python objective. Nine specifically named endpoints fail the strict `3e-9`
gradient matching limit and are marked `xfail(strict=True)`. They are numerical
matching limitations, not passing checks or evidence of a changed formula.
The first uniform hot start differs by `0.676678 m` after complete optimization.
This remains `0.676678 m` when both Backup trajectories are joined to the same
Exp prefix on the absolute clock. The fixture preserves costs, switch times,
durations, endpoints and local/absolute trajectory comparisons; no wider
trajectory tolerance hides that result.

The focused test run on 2026-11-15 reported **62 passed, 9 xfailed**. The xfailed
cases are excluded from the pass count. The worst final-point precision probe
is reproducible with:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/reference/super_trajectory/precision_diagnostic.py
```

That offline probe removes Numba decorators from a copy of the same expressions
and compares float64, NumPy `longdouble` and five-point finite differences. It
does not change runtime arithmetic. Its compact output is retained in
`tests/fixtures/super_trajectory_precision_diagnostic.json`.

All offline entry points default to the separate `.runtime/numba_hotstart`
cache. `tests/conftest.py` selects `.runtime/numba_tests` before Numba imports.
Temporary diagnostic callbacks must not write into the production cache.

To replay a recorded application problem:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python tests/reference/super_trajectory/benchmark.py \
  results/super_risp_development_20261115/profile \
  --cpp-oracle .reference/super_trajectory/oracle
```

The replay directory contains `exp_input.json` and `backup_input.json` from the
real planner. The report includes all inputs, environment versions, module
timing quantiles, original/Python iterations, evaluations, final trajectories,
fixed objective/gradient errors and final trajectory differences. Startup JIT
work is measured separately. See `docs/super_trajectory.md` for the evidence and
the source's diagnostic and hot-start state dependencies.

# Isaac-PathCover

Native Python SUPER with PathCover/RISP corridors on Isaac Lab 3.0 / Newton.

- Preserve SUPER commit `2ad3419c127a617c6d7df6925e81a14175a9c096`; the only algorithm substitution is CIRI geometry with PathCover/RISP commit `8723a05d1993bf0199a453b47dc6ee167d493c23`.
- The single runtime path is ROG-Map, A*, SUPER CorridorGenerator with RISP, MINCO S4NU ExpTrajOpt/BackupTrajOpt/YawTrajOpt, and the source PlanFromRest/ReplanOnce/FSM.
- Project runtime source is Python. NumPy, SciPy, Numba CPU JIT, Torch and installed Isaac Lab/Newton dependencies are allowed. No private native extension or binding.
- `.reference/` contains ignored original sources and independent numerical oracle executables used only for validation. Runtime must not import or execute them.
- Keep the source Isaac-L2F checkout read-only. Do not use `feat/acmpc-native-pathcover-v1` or import CORTO code.
- Scenario assets are immutable exports of the existing Medium (100 obstacles) and Dense reconstructed (180 obstacles) voxel solids. Preserve their provenance and exact start/goal.
- Test numerical contracts against saved official reference fixtures. Record hardware, thread count, solver status, residuals and measured runtimes.
- Use `.venv/bin/python`. Outputs and profiles belong in ignored `results/` or `.runtime/`; compact validation summaries and fixtures belong in `tests/` and `docs/`.
- No alternate flight backend, hidden solver replacement or ideal position executor.
- Planning runs in a CPU process while Newton advances. Preserve source timestamps, prefixes, backup switching and original stopping/commit rules; add no independent acceptance gates.
- Numerical source fixtures, medium/dense flights, measured planning latency and speed/backup timelines determine acceptance. Preserve raw outcomes and provenance.

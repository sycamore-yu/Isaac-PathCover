# Isaac-PathCover

Independent Python implementation of PathCover on Isaac Lab 3.0 / Newton.

- Preserve upstream `kunalnk123690/PathCover` commit `8723a05d1993bf0199a453b47dc6ee167d493c23` algorithm semantics.
- The official active quadrotor trajectory path is 3D GCOPTER, MINCO S3/S4, positive-time and vertex parameterization, and Lewis–Overton L-BFGS.
- Project runtime source is Python. NumPy, SciPy, Torch and installed Isaac Lab/Newton dependencies are allowed. No private native extension or binding.
- `.reference/` contains ignored original sources and independent numerical oracle executables used only for validation. Runtime must not import or execute them.
- Keep the source Isaac-L2F checkout read-only. Do not use `feat/acmpc-native-pathcover-v1` or import CORTO code.
- Scenario assets are immutable exports of the existing Medium (100 obstacles) and Dense reconstructed (180 obstacles) voxel solids. Preserve their provenance and exact start/goal.
- Test numerical contracts against saved official reference fixtures. Record hardware, thread count, solver status, residuals and measured runtimes.
- Use `.venv/bin/python`. Outputs and profiles belong in ignored `results/` or `.runtime/`; compact validation summaries and fixtures belong in `tests/` and `docs/`.
- No alternate flight backend, hidden solver replacement or ideal position executor.

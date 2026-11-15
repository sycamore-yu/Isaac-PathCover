# Isaac-PathCover

Native Python SUPER with PathCover/RISP corridors on Isaac Lab 3.0 / Newton.

- Preserve the SUPER planning pipeline from commit `2ad3419c127a617c6d7df6925e81a14175a9c096`; replace only CIRI geometry with PathCover/RISP commit `8723a05d1993bf0199a453b47dc6ee167d493c23`. Execution controllers may differ.
- Both controllers share one planning path: ROG-Map, A*, SUPER CorridorGenerator with RISP, MINCO S4NU ExpTrajOpt/BackupTrajOpt/YawTrajOpt, and source PlanFromRest/ReplanOnce/FSM.
- Project runtime source is Python. NumPy, SciPy, Numba CPU JIT, Torch and installed Isaac Lab/Newton dependencies are allowed. No private native extension or binding.
- `.reference/` contains ignored original sources and independent numerical oracle executables used only for validation. Runtime must not import or execute them.
- Keep the source Isaac-L2F checkout read-only. Do not use `feat/acmpc-native-pathcover-v1` or import CORTO code.
- Scenario assets are immutable exports of the existing Medium (100 obstacles) and Dense reconstructed (180 obstacles) voxel solids. Preserve their provenance and exact start/goal.
- For controller changes, verify direct Ideal execution and geometric collision checks, NMPC hover and rotor limits, then compare Medium seed2 and Dense seed4. Record hardware, thread count and measured solve times.
- Use `.venv/bin/python`. Outputs and profiles belong in ignored `results/` or `.runtime/`; compact validation summaries and fixtures belong in `tests/` and `docs/`.
- The execution modes are `--controller ideal|nmpc`: SUPER perfect-drone state assignment or the copied Python 10-state TurboMPC. NMPC sends collective thrust acceleration and body rates to the rate controller, RotorActuator and Newton. Motor lag remains in the plant. Both consume the same published CmdTraj; no geometric tracker or additional controller mode.
- Planning runs in a CPU process in both modes. Ideal advances the prescribed trajectory state and geometric collision checks directly; NMPC advances the existing Newton physics and rotor model. Preserve source timestamps, prefixes, backup switching and original stopping/commit rules; add no independent acceptance gates.
- Preserve planning parameters, FSM and timing rules. Retain historical results. Offline C++ hot-start matching is outside the controller task.

"""Build original C++ only for offline ROG-Map fixture regeneration."""
from pathlib import Path
import json
import hashlib
import shlex
import subprocess
import sys

root = Path(__file__).resolve().parents[3]
source = Path(sys.argv[1]) if len(sys.argv) > 1 else root.parents[2] / "reference_repos/SUPER"
build = root / ".runtime/super_mapping_oracle"
build.mkdir(parents=True, exist_ok=True)
(build / "log").mkdir(exist_ok=True)
reference = Path(__file__).resolve().parent
flags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "--libs", "pcl_common", "pcl_io", "yaml-cpp"], text=True))
sources = [source / "rog_map/src/rog_map" / f"{name}.cpp" for name in ("rog_map", "prob_map", "inf_map", "counter_map", "sliding_map", "esdf_map")]
sources.append(source / "rog_map/include/rog_map/rog_map_core/raycaster.cpp")
compile_flags = ["g++", "-O2", "-std=c++17", "-DORIGIN_AT_CORNER",
                 f'-DROOT_DIR="{build}/"', f"-I{reference / 'include'}", f"-I{source / 'rog_map/include'}",
                 "-I/usr/include/eigen3", *[flag for flag in flags if flag.startswith(("-I", "-D"))]]
objects = []
for path in [*sources, reference / "oracle.cpp"]:
    obj = build / (path.stem + ".o")
    if not obj.exists() or obj.stat().st_mtime < path.stat().st_mtime:
        subprocess.run([*compile_flags, "-c", str(path), "-o", str(obj)], check=True)
    objects.append(obj)
subprocess.run(["g++", *map(str, objects), *flags, "-o", str(build / "oracle")], check=True)
result = {"super_commit": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(), "cases": {}}
result["source_sha256"] = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
for mode in ("ray", "hits", "features"):
    output = subprocess.check_output([str(build / "oracle"), str(reference / "config.yaml"), mode], text=True)
    (build / f"{mode}.txt").write_text(output)
    phases = 4 if mode == "features" else 3
    values = {"queries": [], "boxes": [[] for _ in range(phases)], "rays": [[], [], [], []], "lines": []}
    if mode == "features":
        values.update(frontier=[], frontier_boxes=[[] for _ in range(phases)], esdf=[], esdf_bounds=[])
    for line in output.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "QUERY":
            values["queries"].append([int(fields[1]), *map(float, fields[2:])])
        elif fields[0] == "BOX":
            values["boxes"][int(fields[1])].append(list(map(float, fields[2:])))
        elif fields[0] == "RAY":
            values["rays"][int(fields[1])].append(list(map(float, fields[2:])))
        elif fields[0] == "LINE":
            values["lines"].append(list(map(int, fields[1:])))
        elif fields[0] == "FRONTIER":
            values["frontier"].append([int(fields[1]), *map(float, fields[2:])])
        elif fields[0] == "FRONTIER_BOX":
            values["frontier_boxes"][int(fields[1])].append(list(map(float, fields[2:])))
        elif fields[0] == "ESDF":
            values["esdf"].append([int(fields[1]), *map(float, fields[2:])])
        elif fields[0] == "ESDF_BOUNDS":
            values["esdf_bounds"].append([int(fields[1]), *map(float, fields[2:])])
    result["cases"][mode] = values
target = root / "tests/fixtures/super_mapping_reference.json"
target.write_text(json.dumps(result, indent=2) + "\n")
print(target)

"""Compile SUPER's original astar.cpp and ROG-Map for fixed search fixtures."""
from pathlib import Path
import hashlib
import json
import re
import shlex
import subprocess
import sys

root=Path(__file__).resolve().parents[3]
source=root.parents[2]/'reference_repos/SUPER'
reference=Path(__file__).resolve().parent
build=root/'.runtime/super_search_oracle'
build.mkdir(parents=True,exist_ok=True)
(build/'log').mkdir(exist_ok=True)
mapping_reference=reference.parent/'super_mapping'
flags=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','pcl_common','pcl_io','yaml-cpp'],text=True))
sources=[source/'rog_map/src/rog_map'/f'{n}.cpp' for n in ('rog_map','prob_map','inf_map','counter_map','sliding_map','esdf_map')]
sources.extend([source/'rog_map/include/rog_map/rog_map_core/raycaster.cpp',source/'super_planner/src/super_core/astar.cpp'])
compile_sources=list(map(str,sources))
if '--map-objects' in sys.argv:
    object_directory=Path(sys.argv[sys.argv.index('--map-objects')+1]).resolve()
    compile_sources=[str(object_directory/f'{p.stem}.o') for p in sources[:-1]]+[str(sources[-1])]
command=['g++','-O2','-std=c++17','-DORIGIN_AT_CORNER','-DCLASS_COLOR_MSG_UTILS',f'-DROOT_DIR="{build}/"',
         f'-I{reference/"include"}',f'-I{mapping_reference/"include"}',f'-I{source/"super_planner/include"}',
         f'-I{source/"rog_map/include"}','-I/usr/include/eigen3',str(reference/'oracle.cpp'),*compile_sources,*flags,'-o',str(build/'oracle')]
if '--no-build' not in sys.argv:
    subprocess.run(command,check=True)
result={'source_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest()for p in sources},'cases':{},'nearest':{},'lines':{}}
for diagonal in (True,False):
    config=build/f'config_{int(diagonal)}.yaml'
    config.write_text((mapping_reference/'config.yaml').read_text()+f'\nastar:\n  map_voxel_num: [81,81,61]\n  allow_diag: {str(diagonal).lower()}\n  heu_type: 2\n  visual_process: false\n  debug_visualization_en: false\n')
    output=subprocess.check_output([str(build/'oracle'),str(config)],text=True)
    (build/f'output_{int(diagonal)}.txt').write_text(output)
    for line in output.splitlines():
        parts=re.sub(r'\x1b\[[0-9;]*m', '', line).split()
        if parts and parts[0]=='CASE':
            name=f'diagonal_{int(diagonal)}_{parts[1]}'
            result['cases'][name]={'allow_diag':diagonal,'scenario':parts[1],'status':parts[2],'path':[]}
        elif parts and parts[0]=='POINT':
            result['cases'][f'diagonal_{int(diagonal)}_{parts[1]}']['path'].append(list(map(float,parts[2:])))
        elif parts and parts[0]=='NEAREST':
            result['nearest'][parts[1]]={'found':bool(int(parts[2])), 'input':list(map(float,parts[3:6])),
                                        'output':list(map(float,parts[6:9]))}
        elif parts and parts[0]=='LINE':
            result['lines'][parts[1]]={'neighbor_step':int(parts[2]),'max_distance':float(parts[3]),
                                      'free':bool(int(parts[4])),'start':list(map(float,parts[5:8])),
                                      'end':list(map(float,parts[8:11])),'last':list(map(float,parts[11:14]))}
target=root/'tests/fixtures/super_search_reference.json'
target.write_text(json.dumps(result,indent=2)+'\n');print(target)

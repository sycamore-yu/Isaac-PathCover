"""Replay captured SUPER optimizer inputs; optionally compare the offline C++ oracle."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter

os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(__file__).resolve().parents[3]/'.runtime/numba_hotstart'))

import numba
import numpy as np
import scipy

from regenerate import serialize
from isaac_pathcover.trajectory.minco import PiecewisePolynomial
from isaac_pathcover.trajectory.super_opt import ExpTrajOpt, BackupTrajOpt, warmup


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('replay_dir',type=Path)
    parser.add_argument('--trials',type=int,default=10)
    parser.add_argument('--cpp-oracle',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    started=perf_counter();warmup();initialization=perf_counter()-started
    exp=json.loads((args.replay_dir/'exp_input.json').read_text())
    backup=json.loads((args.replay_dir/'backup_input.json').read_text())
    exp_args=(exp['head'],exp['tail'],exp['guide_path'],exp['guide_times'],exp['corridors'])
    backup_args=(PiecewisePolynomial(backup['durations'],backup['coefficients']),backup['t0'],backup['te'],
        backup['heu_ts'],backup['heu_end'],backup['heu_duration'],backup['corridor'])
    report={'environment':{'python':platform.python_version(),'numpy':np.__version__,
        'scipy':scipy.__version__,'numba':numba.__version__},'numerical_initialization_s':initialization,'cases':{}}
    for name,cls,inputs in [('exp',ExpTrajOpt,exp_args),('backup',BackupTrajOpt,backup_args)]:
        prepared=cls().prepare(*inputs);initial=prepared.initial_x.copy()
        f,g=prepared.objective(initial)
        trials=[]
        for _ in range(args.trials):
            obj=cls();started=perf_counter();result=obj.optimize(*inputs);elapsed=perf_counter()-started
            trials.append({'elapsed_s':elapsed,'success':result.success,'status':result.status_code,
                'cost':result.cost,'iterations':result.iterations,'evaluations':result.evaluations,
                'switch_time':result.switch_time,'penalty_log':result.penalty_log.tolist(),'timings':result.timings})
        elapsed=np.array([t['elapsed_s'] for t in trials])
        data={'input':serialize(prepared,initial,True),'trials':trials,
            'p50_s':float(np.median(elapsed)),'p95_s':float(np.percentile(elapsed,95)),
            'optimized_durations':result.trajectory.durations.tolist(),
            'optimized_coefficients':result.trajectory.coefficients.tolist()}
        if args.cpp_oracle:
            started=perf_counter()
            reference=json.loads(subprocess.run([args.cpp_oracle],input=data['input'],text=True,capture_output=True,check=True).stdout)
            data['cpp_process_s']=perf_counter()-started;data['reference']=reference
            data['cost_scaled_error']=float(abs(f-reference['cost'])/(1+abs(reference['cost'])))
            data['gradient_scaled_error']=float(np.max(np.abs(g-reference['gradient'])/(1+np.abs(reference['gradient']))))
            expected=PiecewisePolynomial(reference['optimized_durations'],np.array(reference['optimized_coefficients']).reshape(-1,8,3))
            sample_t=np.linspace(0.,min(expected.duration,result.trajectory.duration),101)
            data['optimized_cost_scaled_error']=float(abs(result.cost-reference['optimized_cost'])/(1+abs(reference['optimized_cost'])))
            data['position_max_error_m']=float(np.max(np.linalg.norm(result.trajectory.evaluate(sample_t)-expected.evaluate(sample_t),axis=1)))
            data['velocity_max_error_m_s']=float(np.max(np.linalg.norm(result.trajectory.evaluate(sample_t,1)-expected.evaluate(sample_t,1),axis=1)))
            data['duration_error_s']=float(abs(result.trajectory.duration-expected.duration))
        report['cases'][name]=data
        print(name,'P50',data['p50_s'],'P95',data['p95_s'],'Python iterations/evaluations',
            trials[-1]['iterations'],trials[-1]['evaluations'],flush=True)
        if args.cpp_oracle:
            print(name,'C++ iterations/evaluations',reference['optimizer_iterations'],reference['optimizer_evaluations'],
                'fixed errors',data['cost_scaled_error'],data['gradient_scaled_error'],flush=True)
    output=args.output or args.replay_dir/'trajectory_numba_cpp.json'
    output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

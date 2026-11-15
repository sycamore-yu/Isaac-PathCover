"""Fixed original-C++ explicit hot-start overload and inherited-time checks."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from unittest.mock import patch

import numpy as np

from regenerate import ROOT, build, box, matrix, vector, serialize as serialize_problem
from isaac_pathcover.trajectory import super_opt
from isaac_pathcover.trajectory.super_opt import OptimizerConfig, ExpTrajOpt, BackupTrajOpt
from isaac_pathcover.trajectory.minco import MINCO, PiecewisePolynomial


def cases():
    head=np.zeros((3,4));head[:,0]=[-.5,0.,1.];head[:,1]=[.4,-.1,.1]
    head[:,2]=[.05,.08,-.03];head[:,3]=[.01,-.02,.03]
    tail=np.zeros((3,4));tail[:,0]=[4.,.5,1.2]
    corridors=[box([-1,-1,0],[2,1,2]),box([0,-1,0],[5,1,2])]
    for typ in (1,2):
        yield {'name':f'exp_hot_type{typ}','kind':'exp','config':asdict(OptimizerConfig(pos_constraint_type=typ)),
               'head':head.tolist(),'tail':tail.tolist(),'corridors':[h.tolist() for h in corridors],
               'init_times':[1.3,2.1],'init_points':[[1.2,.1,1.1]],'repeat':1}
    exp=MINCO(head,tail,2,4).set_parameters([[2.,.2,1.2]],[1.4,1.7]).trajectory()
    for uniform in (False,True):
        yield {'name':f'backup_hot_uniform{int(uniform)}','kind':'backup',
               'config':asdict(OptimizerConfig.backup(uniform_time_en=uniform)),
               'head':head.tolist(),'tail':tail.tolist(),
               'corridors':[box([-1,-1,0],[5,2,2.5]).tolist()],
               'init_times':[.82,1.05],'init_points':[[1.7,.35,1.2],[2.7,.6,1.2]],
               'exp_durations':exp.durations.tolist(),'exp_coefficients':exp.coefficients.tolist(),
               't0':.3,'te':1.4,'heu_ts':.7,'prior_duration':1.7,'repeat':2 if uniform else 1}


def serialize(raw):
    c=OptimizerConfig(**raw['config']);backup=raw['kind']=='backup'
    values=[int(backup),2,c.pos_constraint_type,int(c.block_energy_cost),int(c.uniform_time_en),
            c.smooth_eps,c.integral_reso,c.penna_t,c.penna_ts,raw.get('t0',0.),raw.get('te',1.),
            raw.get('heu_ts',0.),c.opt_accuracy,raw.get('prior_duration',0.),raw['repeat']]
    data=' '.join(map(str,values))+'\n'
    data+=vector(c.physical_parameters)
    data+=vector([c.max_vel,c.max_acc,c.max_jerk,c.max_omg,c.min_acc_thr*c.mass,c.max_acc_thr*c.mass])
    data+=vector([c.penna_pos,c.penna_vel,c.penna_acc,c.penna_jerk,c.penna_attract,c.penna_omg,c.penna_thr])
    data+=matrix(raw['head'])+matrix(raw['tail'])+vector(raw['init_times'])+matrix(np.asarray(raw['init_points']).T)
    data+=str(len(raw['corridors']))+'\n'+''.join(matrix(h) for h in raw['corridors'])
    data+=str(len(raw.get('exp_durations',[])))+'\n'
    for t,c in zip(raw.get('exp_durations',[]),raw.get('exp_coefficients',[])):
        data+=repr(t)+'\n'+matrix(np.asarray(c)[::-1].T)
    return data


def make_prepared(raw,reference):
    """Freeze C++ vertex order while exercising the Python public preparation."""
    config=OptimizerConfig(**raw['config']);backup=raw['kind']=='backup'
    if backup:
        exp=PiecewisePolynomial(raw['exp_durations'],raw['exp_coefficients'])
        duration=reference['inherited_total_time'] if config.uniform_time_en else sum(raw['init_times'])
        obj=BackupTrajOpt(config)
        args=(exp,raw['t0'],raw['te'],raw['heu_ts'],raw['init_points'][-1],duration,raw['corridors'][0])
    else:
        obj=ExpTrajOpt(config)
        args=(raw['head'],raw['tail'],[],[],[np.asarray(h).copy() for h in raw['corridors']])
    keywords={'init_times':raw['init_times'],'init_points':raw['init_points']}
    frozen=[np.asarray(poly) for poly in reference['vertices']]
    with patch('isaac_pathcover.trajectory.super_opt._vertex_coordinates',side_effect=frozen):
        obj.prepare(*args,**keywords)
    return obj,args,keywords,frozen


def scaled_error(actual,expected):
    expected=np.asarray(expected)
    return float(np.max(np.abs(np.asarray(actual)-expected)/(1+np.abs(expected)),initial=0.))


def compare(raw,reference,objective_oracle):
    obj,args,keywords,frozen=make_prepared(raw,reference)
    cost,gradient=obj.objective(obj.initial_x)
    metrics={
        'initial_x_scaled_error':scaled_error(obj.initial_x,reference['initial_x']),
        'initial_cost_scaled_error':scaled_error(cost,reference['initial_cost']),
        'initial_gradient_scaled_error':scaled_error(gradient,reference['initial_gradient']),
        'initial_coefficients_scaled_error':scaled_error(obj.minco.coefficients.reshape(-1,3),reference['initial_coefficients']),
        'initial_durations_scaled_error':scaled_error(obj.minco.durations,reference['initial_durations']),
        'stored_init_times_scaled_error':scaled_error(obj.init_times,reference['stored_init_times']),
        'stored_init_points_scaled_error':scaled_error(obj.init_points,np.asarray(reference['stored_init_points']).T)}
    if raw['kind']=='exp':
        metrics['attractors_scaled_error']=scaled_error(obj.attractors,np.asarray(reference['attractors']).T)
        metrics['attractor_radii_scaled_error']=scaled_error(obj.radii,reference['attractor_radii'])
    f,g=obj.objective(np.asarray(reference['initial_x']))
    metrics['common_initial_cost_scaled_error']=scaled_error(f,reference['initial_cost'])
    metrics['common_initial_gradient_scaled_error']=scaled_error(g,reference['initial_gradient'])
    cpp_repeat=json.loads(subprocess.run([objective_oracle],
        input=serialize_problem(obj,np.asarray(reference['initial_x']),True),
        text=True,capture_output=True,check=True).stdout)
    metrics['cpp_public_overload_final_x_repeat_scaled_error']=scaled_error(cpp_repeat['optimized_x'],reference['last_x'])
    metrics['cpp_optimizer_status']=cpp_repeat['optimizer_status']
    metrics['cpp_optimizer_iterations']=cpp_repeat['optimizer_iterations']
    metrics['cpp_optimizer_evaluations']=cpp_repeat['optimizer_evaluations']
    captured=[]
    source_solve=super_opt._solve
    def record_solve(problem,initial=None):
        solved=source_solve(problem,initial);captured.append(solved);return solved
    with patch('isaac_pathcover.trajectory.super_opt._vertex_coordinates',side_effect=frozen), \
         patch('isaac_pathcover.trajectory.super_opt._solve',side_effect=record_solve):
        result=obj.optimize(*args,**keywords)
    metrics.update({'success':result.success,'cpp_success':reference['success'],
                    'optimizer_status':result.status,'optimizer_status_code':captured[-1].status,
                    'iterations':result.iterations,'evaluations':result.evaluations})
    if result.trajectory is not None and reference['trajectory']['durations']:
        expected=PiecewisePolynomial(reference['trajectory']['durations'],reference['trajectory']['coefficients'])
        times=np.linspace(0.,min(result.trajectory.duration,expected.duration),101)
        metrics.update({'optimized_cost_scaled_error':scaled_error(result.cost,reference['last_cost']),
                        'position_max_error_m':float(np.max(np.linalg.norm(result.trajectory.evaluate(times)-expected.evaluate(times),axis=1))),
                        'velocity_max_error_m_s':float(np.max(np.linalg.norm(result.trajectory.evaluate(times,1)-expected.evaluate(times,1),axis=1))),
                        'duration_error_s':abs(result.trajectory.duration-expected.duration),
                        'switch_time_error_s':abs(result.switch_time-reference['out_ts']) if result.switch_time is not None else 0.})
        if result.switch_time is not None:
            cpp_ts=reference['out_ts'];py_ts=result.switch_time
            exp=PiecewisePolynomial(raw['exp_durations'],raw['exp_coefficients'])
            absolute_times=np.linspace(0.,min(cpp_ts+expected.duration,py_ts+result.trajectory.duration),1001)
            def command(trajectory,ts):
                values=exp.evaluate(absolute_times)
                active=absolute_times>=ts
                values[active]=trajectory.evaluate(absolute_times[active]-ts)
                return values
            metrics.update({'cpp_switch_time_s':cpp_ts,'python_switch_time_s':py_ts,
                            'cpp_duration_s':expected.duration,'python_duration_s':result.trajectory.duration,
                            'cpp_tail_position':expected.evaluate(expected.duration).tolist(),
                            'python_tail_position':result.trajectory.evaluate(result.trajectory.duration).tolist(),
                            'tail_position_error_m':float(np.linalg.norm(expected.evaluate(expected.duration)-result.trajectory.evaluate(result.trajectory.duration))),
                            'absolute_command_position_max_error_m':float(np.max(np.linalg.norm(command(expected,cpp_ts)-command(result.trajectory,py_ts),axis=1))),
                            'absolute_command_sample_count':1001})
    cross=[]
    for name,x in [('cpp_final',np.asarray(reference['last_x'])),('python_final',captured[-1].x)]:
        cpp=json.loads(subprocess.run([objective_oracle],input=serialize_problem(obj,x,False),
            text=True,capture_output=True,check=True).stdout)
        f,g=obj.objective(x)
        cross.append({'name':name,'x':x.tolist(),'reference':cpp,
                      'cost_scaled_error':scaled_error(f,cpp['cost']),
                      'gradient_scaled_error':scaled_error(g,cpp['gradient']),
                      'coefficients_scaled_error':scaled_error(obj.minco.coefficients.reshape(-1,3),cpp['coefficients'])})
    metrics['cross_evaluation']=cross
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('/home/tong/tongworkspace/reference_repos/SUPER'))
    parser.add_argument('--no-build',action='store_true');args=parser.parse_args()
    build_dir=ROOT/'.reference/super_trajectory_hotstart'
    oracle=build_dir/'oracle' if args.no_build else build(args.source,build_dir,hotstart=True)
    evidence=[]
    for raw in cases():
        text=serialize(raw)
        cpp=json.loads(subprocess.run([oracle],input=text,text=True,capture_output=True,check=True).stdout)
        metrics=[compare(raw,stage,ROOT/'.reference/super_trajectory/oracle') for stage in cpp['stages']]
        evidence.append({'name':raw['name'],'input':raw,'oracle_input':text,'reference':cpp,'python_comparison':metrics})
        print(raw['name'],json.dumps([{k:v for k,v in row.items() if k!='cross_evaluation'} for row in metrics]),flush=True)
    sources=['super_planner/src/traj_opt/exp_traj_optimizer_s4.cpp',
             'super_planner/src/traj_opt/backup_traj_optimizer_s4.cpp',
             'super_planner/src/utils/minco.cpp','super_planner/src/utils/lbfgs.cpp',
             'super_planner/src/utils/geometry_utils.cpp','super_planner/src/utils/polytope.cpp',
             'super_planner/src/utils/optimization_utils.cpp']
    report={'source_commit':'2ad3419c127a617c6d7df6925e81a14175a9c096',
            'source_sha256':{path:hashlib.sha256((args.source/path).read_bytes()).hexdigest() for path in sources},
            'initialization_scope':'Original public explicit overloads, including original geometry setup and acceptance; Python vertex order frozen from C++ to isolate optimization coordinates.',
            'uniform_backup_state':'Each stage inherits the recorded original C++ total_time from the preceding heuristic or explicit optimization; Python receives that scalar as heu_duration.',
            'cases':evidence}
    (ROOT/'tests/fixtures/super_trajectory_hotstart.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()

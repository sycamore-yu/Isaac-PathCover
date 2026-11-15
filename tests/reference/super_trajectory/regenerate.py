"""Build original SUPER numerical functions, evaluate fixtures, and compare Python.

The extraction removes ROS/logging dependencies; the objective, gradient, MINCO,
flatness, L-BFGS, interpolation and polynomial implementations are upstream C++.
This script is an offline oracle and is never imported by the planner runtime.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.runtime/numba_hotstart'))
sys.path.insert(0,str(ROOT/'src'))
from isaac_pathcover.trajectory.super_opt import OptimizerConfig, ExpTrajOpt, BackupTrajOpt, YawTrajOpt, _pack_problem
from isaac_pathcover.trajectory.minco import MINCO, PiecewisePolynomial


def function(source, name):
    start = source.index(name)
    start = source.rfind('\n',0,start)+1
    # Return types sometimes occupy the preceding line.
    if source[start:].startswith(name):
        start = source.rfind('\n',0,start-1)+1
    brace = source.index('{',start)
    depth = 0
    token = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\{|\}',re.S)
    for match in token.finditer(source,brace):
        if match.group()=='{': depth += 1
        if match.group()=='}':
            depth -= 1
            if depth==0:
                return source[start:match.end()]+'\n'
    raise ValueError(name)


def build(source, build_dir, *, hotstart=False):
    inc = source/'super_planner/include'
    src = source/'super_planner/src'
    build_dir.mkdir(parents=True,exist_ok=True)
    for name in ['utils/header/color_msg_utils.hpp','super_utils/color_msg_utils.hpp']:
        target=build_dir/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('#pragma once\n')
    prologue='''#include <iostream>
#include <iomanip>
#include <cfloat>
#include <vector>
#include <sstream>
#include <traj_opt/minco.h>
#include <utils/geometry/quadrotor_flatness.hpp>
#include <utils/optimization/optimization_utils.h>
#include <utils/optimization/polynomial_interpolation.h>
using namespace std;
using namespace super_utils;
using namespace geometry_utils;
using namespace traj_opt;
using namespace math_utils;
using namespace optimization_utils;
using namespace color_text;
using Vec8f = Eigen::Matrix<double,8,1>;
using Mat83f = Eigen::Matrix<double,8,3>;
#define POS_IDX 1
#define VEL_IDX 2
#define ACC_IDX 3
#define JER_IDX 4
#define ATT_IDX 5
#define OMG_IDX 6
#define THR_IDX 7
'''
    if hotstart:
        prologue += '''#include <data_structure/base/polytope.h>
#include <utils/header/scope_timer.hpp>
struct OracleConfig {
 int pos_constraint_type=1,piece_num=2;
 bool print_optimizer_log=false,save_log_en=false;
 double max_vel=8.,max_acc=15.,max_omg=2.5,penna_margin=.05,opt_accuracy=5e-6;
 double penna_pos=5e9,penna_vel=5e8,penna_acc=5e8,penna_omg=-1.,penna_thr=-1.;
};
struct OracleLog {void warn(const std::string&){} double getSimTime(){return 0.;}};
OracleLog oracle_log;
bool capture_enabled=false;
VecDf captured_x,captured_gradient,captured_times,last_x;
MatD3f captured_coefficients;
double captured_cost=0.,last_cost=0.;
'''
    extracted=[]
    for cls,stem in [('ExpTrajOpt','exp_traj_optimizer_s4'),('BackupTrajOpt','backup_traj_optimizer_s4')]:
        header=(inc/'traj_opt'/f'{stem}.h').read_text()
        body=(src/'traj_opt'/f'{stem}.cpp').read_text()
        begin=header.index('struct OptimizationVariables')
        end=header.index('} opt_vars',begin)+len('} opt_vars')
        struct=header[begin:end]+';'
        functions=[function(body,cls+'::'+name) for name in ['constraintsFunctional','costFunctional']]
        static_count=2
        extra_members=''
        if hotstart:
            # Only the cost entry is wrapped for recording; its body remains original.
            functions[1]=functions[1].replace(cls+'::costFunctional',cls+'::sourceCostFunctional',1)
            wrapper='''double CLASS::costFunctional(void *ptr,const VecDf &x,VecDf &g) {
 const double value=sourceCostFunctional(ptr,x,g);
 if(capture_enabled) {
  auto& o=*static_cast<OptimizationVariables*>(ptr);
  if(captured_x.size()==0) {
   captured_x=x;captured_gradient=g;captured_cost=value;
   captured_coefficients=o.minco.getCoeffs();captured_times=o.times;
  }
  last_x=x;last_cost=value;
 }
 return value;
}'''.replace('CLASS',cls)
            functions.append(wrapper);static_count=3
            names=['processCorridor','setupProblemAndCheck','optimize(Trajectory &traj, const double &relCostTol)']
            if cls=='ExpTrajOpt':
                prologue+=function(body,'truncateToSixDecimals')
                names += ['processCorridorWithGuideTraj','defaultInitialization']
                signature='optimize(const StatePVAJ &headPVAJ, const StatePVAJ &tailPVAJ,\n                          PolytopeVec &sfcs,'
                functions += [function(body,cls+'::'+name) for name in names]
                functions.append(function(body,cls+'::'+signature))
            else:
                names += ['checkTrajMagnitudeBound']
                functions += [function(body,cls+'::'+name) for name in names]
                first=body.index(cls+'::optimize(const Trajectory &exp_traj,')
                second=body.index(cls+'::optimize(const Trajectory &exp_traj,',first+1)
                functions += [function(body[first-5:],cls+'::optimize(const Trajectory &exp_traj,'),
                              function(body[second-5:],cls+'::optimize(const Trajectory &exp_traj,')]
            extra_members='OracleConfig cfg_;std::ostringstream failed_traj_log,penalty_log;OracleLog* ros_ptr_=&oracle_log;'
        declarations=[]
        for index,f in enumerate(functions):
            signature=f[:f.index('{')].replace(cls+'::','').strip()
            declarations.append(('static ' if index<static_count else '')+signature+';')
        extracted.append('class '+cls+' { public: '+extra_members+struct+'\n'+'\n'.join(declarations)+'\n};\n'+'\n'.join(functions))
    # Compile upstream interpolation for independent yaw-coefficient checks.
    yaw=(src/'traj_opt/yaw_traj_opt.cpp').read_text()
    yaw_funcs=[function(yaw,'YawTrajOpt::'+name) for name in ['getYawTimeAllocation','getYawWaypointAllocation','optimize']]
    geom=(src/'utils/geometry_utils.cpp').read_text()
    normalization=function(geom,'normalizeNextYaw')
    normalization=normalization.replace('geometry_utils::normalizeNextYaw','normalizeNextYaw')
    if not hotstart:
        prologue+='namespace geometry_utils { '+normalization+' }\n'
    yaw_decl='''class YawTrajOpt { public: bool free_goal_{false}; double yaw_dot_max_{3.};
void getYawTimeAllocation(const double&,VecDf&) const;
static void getYawWaypointAllocation(const Vec4f&,Vec4f&,VecDf&,VecDf&,const Trajectory&);
bool optimize(const Vec4f&,const Vec4f&,const Trajectory&,Trajectory&,const int&,const bool&,const bool&);
};\n'''
    driver=(Path(__file__).parent/'driver.cpp').read_text()
    if hotstart:
        driver=driver.split('int last_iteration=0;')[0]+(Path(__file__).parent/'hotstart_driver.cpp').read_text()
    merged=prologue+'\n'.join(extracted)+yaw_decl+'\n'.join(yaw_funcs)+'\n'+driver
    (build_dir/'oracle.cpp').write_text(merged)
    cmd=['g++','-O2','-DNDEBUG','-std=c++17',f'-I{build_dir}',f'-I{inc}',f'-I{source}/rog_map/include','-I/usr/include/eigen3',str(build_dir/'oracle.cpp')]
    cmd += [str(src/'utils'/f'{s}.cpp') for s in ['minco','banded_system','trajectory','piece','root_finder','optimization_utils','lbfgs']]
    if hotstart:
        cmd += ['-include','cstdint']
        cmd += [str(src/'utils'/f'{s}.cpp') for s in ['geometry_utils','polytope','ellipsoid','sdlp','quickhull']]
    cmd += ['-o',str(build_dir/'oracle')]
    subprocess.run(cmd,check=True)
    return build_dir/'oracle'


def matrix(a):
    a=np.asarray(a);return f'{a.shape[0]} {a.shape[1]} '+' '.join(map(repr,a.ravel().tolist()))+'\n'


def vector(a):
    a=np.asarray(a).ravel();return str(a.size)+' '+' '.join(map(repr,a.tolist()))+'\n'


def serialize(obj,x,optimize=False):
    cfg=obj.config;backup=isinstance(obj,BackupTrajOpt)
    n=cfg.piece_num if backup else obj.piece_count
    data=f'{int(backup)} {n} {cfg.pos_constraint_type} {int(cfg.block_energy_cost)} {int(cfg.uniform_time_en)}\n'
    data+=' '.join(map(str,[cfg.smooth_eps,cfg.integral_reso,cfg.penna_t,cfg.penna_ts,
        obj.min_ts if backup else 0,obj.max_ts if backup else 1,obj.ts if backup else 0]))+'\n'
    data+=vector(cfg.physical_parameters)
    data+=vector([cfg.max_vel,cfg.max_acc,cfg.max_jerk,cfg.max_omg,cfg.min_acc_thr*cfg.mass,cfg.max_acc_thr*cfg.mass])
    data+=vector([cfg.penna_pos,cfg.penna_vel,cfg.penna_acc,cfg.penna_jerk,cfg.penna_attract,cfg.penna_omg,cfg.penna_thr])
    data+=matrix(obj.head)+matrix(obj.tail)+vector(x)
    hpolys=[obj.hpoly] if backup else obj.hpolys
    data+=str(len(hpolys))+'\n'+''.join(matrix(p) for p in hpolys)
    data+=str(len(obj.vpolys))+'\n'+''.join(matrix(p.T) for p in obj.vpolys)
    data+=matrix(np.empty((3,0)) if backup else obj.attractors.T)
    data+=vector(np.empty(0) if backup else obj.radii)
    if backup:
        data+=str(len(obj.exp_traj.durations))+'\n'
        for t,c in zip(obj.exp_traj.durations,obj.exp_traj.coefficients):
            data+=repr(float(t))+'\n'+matrix(c[::-1].T)
    else:
        data+='0\n'
    data+=f'{int(optimize)} {cfg.opt_accuracy}\n'
    return data


def deserialize(raw):
    """Read the exact saved C++ input rather than regenerate an initialization."""
    values=iter(raw.split())
    def number():return float(next(values))
    def integer():return int(next(values))
    def vec():return np.array([number() for _ in range(integer())])
    def mat():
        rows,cols=integer(),integer()
        return np.array([number() for _ in range(rows*cols)]).reshape(rows,cols)
    mode,n,typ,block,uniform=[integer() for _ in range(5)]
    eps,res,rho,weight_ts,t0,te,ts=[number() for _ in range(7)]
    phys,bounds,penalties=vec(),vec(),vec();head,tail=mat(),mat();x=vec()
    hp=[mat() for _ in range(integer())];vp=[mat().T.copy() for _ in range(integer())]
    attr,radii=mat().T.copy(),vec();et=[];ec=[]
    for _ in range(integer()):et.append(number());ec.append(mat()[:,::-1].T.copy())
    optimize,accuracy=integer(),number()
    cfg=OptimizerConfig(pos_constraint_type=typ,block_energy_cost=bool(block),uniform_time_en=bool(uniform),
        piece_num=n,integral_reso=int(res),smooth_eps=eps,penna_t=rho,penna_ts=weight_ts,
        max_vel=bounds[0],max_acc=bounds[1],max_jerk=bounds[2],max_omg=bounds[3],
        min_acc_thr=bounds[4]/phys[0],max_acc_thr=bounds[5]/phys[0],
        penna_pos=penalties[0],penna_vel=penalties[1],penna_acc=penalties[2],penna_jerk=penalties[3],
        penna_attract=penalties[4],penna_omg=penalties[5],penna_thr=penalties[6],
        mass=phys[0],grav=phys[1],dh=phys[2],dv=phys[3],cp=phys[4],v_eps=phys[5],opt_accuracy=accuracy)
    obj=BackupTrajOpt(cfg) if mode else ExpTrajOpt(cfg)
    obj.head,obj.tail,obj.vpolys,obj.initial_x=head,tail,vp,x
    obj.minco=MINCO(head,tail,n,4)
    if mode:
        obj.hpoly=hp[0];obj.min_ts=t0;obj.max_ts=te;obj.ts=ts
        obj.exp_traj=PiecewisePolynomial(et,ec);obj.temporal_dim=1 if uniform else n
    else:
        obj.hpolys=hp;obj.attractors=attr;obj.radii=radii;obj.piece_count=n
    obj.problem=_pack_problem(obj,bool(mode))
    return obj,x


def box(low,high):
    return np.column_stack([np.r_[-np.eye(3),np.eye(3)],np.r_[low,-np.asarray(high)]])


def cases():
    h=np.zeros((3,4));h[:,0]=[0.,0.,1.];h[:,1]=[.4,-.1,.1]
    h[:,2]=[.05,.08,-.03];h[:,3]=[.01,-.02,.03]
    t=np.zeros((3,4));t[:,0]=[4.,.7,1.2]
    guide=np.array([[0,0,1],[1,.1,1.1],[2,.5,1.2],[3,.7,1.2],[4,.7,1.2]])
    guide_t=np.array([0.,.6,1.2,1.9,2.6])
    polys=[box([-1,-1,0],[2.6,1.5,2.5]),box([1.5,-.8,.1],[5,2,2.5])]
    for typ in (1,2):
        for energy in (False,True):
            cfg=OptimizerConfig(pos_constraint_type=typ,block_energy_cost=energy,
                max_vel=2.,max_acc=3.,max_jerk=6.,max_omg=1.,min_acc_thr=8.,max_acc_thr=11.,
                penna_t=40.,penna_pos=1000.,penna_vel=150.,penna_acc=120.,penna_jerk=20.,
                penna_attract=70.,penna_omg=100.,penna_thr=40.)
            obj=ExpTrajOpt(cfg).prepare(h,t,guide,guide_t,polys)
            x=obj.initial_x.copy();x[:2]+=[.21,-.12]
            if typ==1: x[2:]+=[.1,.04,-.03]
            else: x[2:]*=1.06
            yield f'exp_type{typ}_energy{int(not energy)}',obj,x
    exp=MINCO(h,t,2,4).set_parameters([[2.,.2,1.2]],[1.4,1.7]).trajectory()
    for uniform in (False,True):
        cfg=OptimizerConfig.backup(uniform_time_en=uniform,block_energy_cost=False,
            max_vel=2.,max_acc=4.,max_jerk=6.,max_omg=1.,min_acc_thr=8.,max_acc_thr=12.,
            penna_t=40.,penna_ts=200.,penna_pos=1000.,penna_vel=150.,penna_acc=120.,
            penna_jerk=20.,penna_omg=100.,penna_thr=40.)
        obj=BackupTrajOpt(cfg).prepare(exp,.3,1.4,.7,[2.8,.7,1.3],1.7,box([-1,-1,0],[5,2,2.5]))
        x=obj.initial_x.copy();x[-1]+=.13
        yield f'backup_uniform{int(uniform)}',obj,x


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,default=Path('/home/tong/tongworkspace/reference_repos/SUPER'))
    parser.add_argument('--no-build',action='store_true');args=parser.parse_args()
    build_dir=ROOT/'.reference/super_trajectory'
    oracle=build_dir/'oracle' if args.no_build else build(args.source,build_dir)
    evidence=[]
    for name,obj,x in cases():
        raw=serialize(obj,x,optimize=True)
        cpp=json.loads(subprocess.run([oracle],input=raw,text=True,capture_output=True,check=True).stdout)
        f,g=obj.objective(x)
        solved=obj.solve_prepared(x);obj.objective(solved.x)
        actual=obj.minco.trajectory()
        expected=PiecewisePolynomial(cpp['optimized_durations'],np.array(cpp['optimized_coefficients']).reshape(-1,8,3))
        sample_t=np.linspace(0.,min(actual.duration,expected.duration),101)
        item={'name':name,'input':raw,'reference':cpp,'python_cost':f,
              'cost_scaled_error':abs(f-cpp['cost'])/max(1.,abs(cpp['cost'])),
              'gradient_scaled_error':float(np.max(np.abs(g-np.array(cpp['gradient']))/(1+np.abs(cpp['gradient'])))),
              'python_optimization':{'status':solved.status,'cost':solved.cost,'iterations':solved.iterations,
                  'evaluations':solved.evaluations,'cost_scaled_error':abs(solved.cost-cpp['optimized_cost'])/(1+abs(cpp['optimized_cost'])),
                  'position_max_error_m':float(np.max(np.linalg.norm(actual.evaluate(sample_t)-expected.evaluate(sample_t),axis=1))),
                  'velocity_max_error_m_s':float(np.max(np.linalg.norm(actual.evaluate(sample_t,1)-expected.evaluate(sample_t,1),axis=1))),
                  'duration_error_s':float(abs(actual.duration-expected.duration))}}
        evidence.append(item)
        print(name,item['cost_scaled_error'],item['gradient_scaled_error'],flush=True)
    target=ROOT/'tests/fixtures/super_trajectory_reference.json'
    sources=['super_planner/src/traj_opt/exp_traj_optimizer_s4.cpp',
        'super_planner/src/traj_opt/backup_traj_optimizer_s4.cpp','super_planner/src/traj_opt/yaw_traj_opt.cpp',
        'super_planner/src/utils/minco.cpp','super_planner/src/utils/lbfgs.cpp',
        'super_planner/src/utils/optimization_utils.cpp']
    target.write_text(json.dumps({'source_commit':'2ad3419c127a617c6d7df6925e81a14175a9c096',
        'source_sha256':{name:hashlib.sha256((args.source/name).read_bytes()).hexdigest() for name in sources},
        'cases':evidence},indent=2)+'\n')


if __name__=='__main__':main()

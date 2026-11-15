"""SUPER 2ad3419 ExpTrajOpt, BackupTrajOpt and YawTrajOpt.

Source: super_planner/src/traj_opt/*_s4.cpp and yaw_traj_opt.cpp.
Copyright 2025 Yunfan Ren, MaRS Lab (LGPL-3.0-or-later).
MINCO and coordinate transforms: Copyright 2021 Zhepei Wang (MIT).
The numerical kernels are Python compiled with the standard Numba CPU JIT.
"""
from dataclasses import dataclass, field, replace
from time import perf_counter
import numpy as np
from ..planning.geometry import GeometryError, chebyshev_center, vertices
from .lbfgs import LBFGSParameters, minimize
from .minco import MINCO, PiecewisePolynomial
from .numerics import objective as numerical_objective, objective_full, point_objective


@dataclass(frozen=True)
class OptimizerConfig:
    """Values from SUPER config/static_dense.yaml, without ROS parameters."""
    pos_constraint_type: int = 1
    uniform_time_en: bool = False
    piece_num: int = 2
    block_energy_cost: bool = False
    opt_accuracy: float = 5e-6
    integral_reso: int = 15
    smooth_eps: float = .01
    max_vel: float = 8.
    max_acc: float = 15.
    max_jerk: float = 120.
    max_omg: float = 2.5
    min_acc_thr: float = 6.
    max_acc_thr: float = 17.
    penna_margin: float = .05
    penna_t: float = 1e7
    penna_ts: float = 1e7
    penna_pos: float = 5e9
    penna_vel: float = 5e8
    penna_acc: float = 5e8
    penna_jerk: float = -5e8
    penna_attract: float = 5e8
    penna_omg: float = -1.
    penna_thr: float = -1.
    mass: float = 1.64
    grav: float = 9.81
    dh: float = .35
    dv: float = .35
    cp: float = .001
    v_eps: float = .0001

    @classmethod
    def backup(cls,**kwargs):
        return replace(cls(pos_constraint_type=2,uniform_time_en=True,
            block_energy_cost=True,opt_accuracy=1e-5,integral_reso=12,
            penna_t=100.,penna_pos=1e10,penna_vel=1e8,penna_acc=1e8,
            penna_jerk=-1e6,penna_attract=-1.),**kwargs)

    @property
    def physical_parameters(self):return self.mass,self.grav,self.dh,self.dv,self.cp,self.v_eps


@dataclass
class OptimizationResult:
    trajectory: PiecewisePolynomial | None
    success: bool
    status: str
    status_code: int
    cost: float
    iterations: int = 0
    evaluations: int = 0
    elapsed_s: float = 0.
    penalty_log: np.ndarray = field(default_factory=lambda:np.zeros(8))
    timings: dict = field(default_factory=dict)
    init_times: np.ndarray = field(default_factory=lambda:np.empty(0))
    init_points: np.ndarray = field(default_factory=lambda:np.empty((0,3)))
    switch_time: float | None = None


def inverse_t(times):
    times=np.asarray(times,dtype=float);out=np.empty_like(times);mask=times>1.
    out[mask]=np.sqrt(2*times[mask]-1)-1
    out[~mask]=1-np.sqrt(2/times[~mask]-1)
    return out


def _inverse_points(points,vpolys):
    params=LBFGSParameters(past=0,delta=1e-5,g_epsilon=np.finfo(np.float32).eps,max_iterations=128)
    values=[]
    for point,poly in zip(points,vpolys):
        result=minimize(point_objective,np.full(len(poly),np.sqrt(1/len(poly))),params,
                        (np.ascontiguousarray(point),np.ascontiguousarray(poly)))
        values.append(result.x)
    return np.concatenate(values) if values else np.empty(0)


def _planes(poly):
    if isinstance(poly,tuple):
        a,b=poly;return np.column_stack([a,-np.asarray(b)])
    return np.asarray(getattr(poly,"planes",poly),dtype=float)


def _vertex_coordinates(planes,interior=None):
    v=vertices(planes[:,:3],-planes[:,3],interior)
    return np.vstack([v[0],v[1:]-v[0]])


def _pack_problem(obj,backup):
    c=obj.config;n=c.piece_num if backup else obj.piece_count
    cfg=np.array([backup,c.pos_constraint_type,c.block_energy_cost,c.uniform_time_en,
        c.integral_reso,c.smooth_eps,c.penna_t,c.penna_ts,c.max_vel,c.max_acc,c.max_jerk,c.max_omg,
        c.min_acc_thr*c.mass,c.max_acc_thr*c.mass,c.penna_pos,c.penna_vel,c.penna_acc,c.penna_jerk,
        c.penna_attract,c.penna_omg,c.penna_thr,*c.physical_parameters,c.opt_accuracy,n],dtype=float)
    polys=[obj.hpoly]*n if backup else obj.hpolys
    counts=np.array([len(p) for p in polys],dtype=np.int64)
    h=np.zeros((n,int(counts.max()),4))
    for i,p in enumerate(polys):h[i,:len(p)]=p
    vcounts=np.array([len(p) for p in obj.vpolys],dtype=np.int64)
    v=np.zeros((len(vcounts),int(vcounts.max(initial=1)),3))
    for i,p in enumerate(obj.vpolys):v[i,:len(p)]=p
    attr=np.empty((0,3)) if backup else obj.attractors
    radii=np.empty(0) if backup else obj.radii
    et=obj.exp_traj.durations if backup else np.empty(0)
    ec=obj.exp_traj.coefficients if backup else np.empty((0,8,3))
    bounds=np.array([obj.min_ts,obj.max_ts,obj.ts]) if backup else np.array([0.,1.,0.])
    return (cfg,np.ascontiguousarray(obj.head),np.ascontiguousarray(obj.tail),h,counts,v,vcounts,
            np.ascontiguousarray(attr),np.ascontiguousarray(radii),np.ascontiguousarray(et),
            np.ascontiguousarray(ec),bounds)


def _solve(obj,initial=None):
    return minimize(numerical_objective,obj.initial_x if initial is None else initial,
        LBFGSParameters(mem_size=256,past=3,min_step=1e-32,g_epsilon=0.,delta=obj.config.opt_accuracy),obj.problem)


def _evaluate(obj,x,backup):
    cost,gradient,coeffs,times,points,ts,log=objective_full(np.asarray(x),obj.problem)
    if backup:
        obj.ts=ts;obj.head=obj.exp_traj.state(ts);obj.tail=np.zeros((3,4));obj.tail[:,0]=points[-1]
        obj.minco.set_conditions(obj.head,obj.tail)
    obj.minco.set_parameters(points[:-1] if backup else points,times)
    # Retain the kernel's source-order coefficients for numerical inspection.
    obj.minco.coefficients=coeffs
    obj.penalty_log=log
    return cost,gradient


class ExpTrajOpt:
    def __init__(self,config=None):self.config=config or OptimizerConfig()

    def prepare(self,head,tail,guide_path,guide_t,corridors,init_times=None,init_points=None):
        from ..planning.corridor import simplify_sfc
        self.head,self.tail=np.asarray(head,dtype=float),np.asarray(tail,dtype=float)
        if init_times is not None:
            guide_path=np.vstack([self.head[:,0],np.asarray(init_points),self.tail[:,0]])
            guide_t=np.r_[0.,np.cumsum(init_times)]
        guide_path,guide_t=np.asarray(guide_path),np.asarray(guide_t)
        if len(guide_path)!=len(guide_t):raise ValueError("guide path and time stamps have different lengths")
        simplified=simplify_sfc(corridors,self.head[:,0],self.tail[:,0])
        if isinstance(corridors,list):corridors[:]=simplified
        self.hpolys=[_planes(p).copy() for p in simplified]
        if not self.hpolys:raise GeometryError("empty exploration corridor")
        for h in self.hpolys:h/=np.linalg.norm(h[:,:3],axis=1)[:,None]
        self.piece_count=len(self.hpolys);attractors=[];radii=[];vpolys=[]
        for a,b in zip(self.hpolys[:-1],self.hpolys[1:]):
            h=np.vstack([a,b]);center,radius=chebyshev_center(h[:,:3],-h[:,3])
            if radius<0 or not np.isfinite(radius):raise GeometryError("invalid exploration corridor intersection")
            attractors.append(center);radii.append(radius/2);vpolys.append(_vertex_coordinates(h,center))
        attractors=np.asarray(attractors).reshape(-1,3);radii=np.asarray(radii)
        if len(attractors):
            ids=np.linalg.norm(guide_path[:,None]-attractors[None],axis=-1).argmin(axis=0)
            stamps=np.r_[0.,guide_t[ids],guide_t[-1]]
        else:stamps=np.array([0.,guide_t[-1]])
        self.init_times=np.maximum(.01,np.diff(stamps))*.8;self.init_points=attractors.copy()
        if self.init_times.min()<1e-3:raise ValueError("initial trajectory piece duration below 1e-3")
        if init_times is not None:
            self.init_times=np.asarray(init_times,dtype=float).copy();self.init_points=np.asarray(init_points,dtype=float).copy()
        self.vpolys=vpolys
        self.attractors=np.trunc(attractors*1e6)/1e6;self.radii=np.trunc(radii*1e6)/1e6
        xi=self.init_points.ravel() if self.config.pos_constraint_type==1 else _inverse_points(self.init_points,vpolys)
        self.initial_x=np.r_[inverse_t(self.init_times),xi]
        self.minco=MINCO(self.head,self.tail,self.piece_count,4);self.problem=_pack_problem(self,False)
        return self

    def objective(self,x):return _evaluate(self,x,False)

    def solve_prepared(self,initial=None):return _solve(self,initial)

    def optimize(self,head,tail,guide_path,guide_t,corridors,init_times=None,init_points=None):
        started=perf_counter()
        try:self.prepare(head,tail,guide_path,guide_t,corridors,init_times,init_points)
        except (GeometryError,ValueError) as exc:
            return OptimizationResult(None,False,str(exc),-1,float('inf'),elapsed_s=perf_counter()-started)
        prepared=perf_counter();result=_solve(self)
        _,_,coeffs,times,_,_,log=objective_full(result.x,self.problem);self.penalty_log=log
        c=self.config
        violation=((c.penna_pos>0 and log[1]>.2) or (c.penna_acc>0 and log[3]>c.max_acc*c.penna_margin)
            or (c.penna_omg>0 and log[6]>c.max_omg*c.penna_margin) or (c.penna_thr>0 and log[7]>c.max_acc*c.penna_margin))
        success=result.success and not violation
        trajectory=PiecewisePolynomial(times,coeffs) if success else None;ended=perf_counter()
        return OptimizationResult(trajectory,success,"SUPER penalty violation" if violation else result.message,
            -1 if violation else result.status,result.cost if success else float('inf'),result.iterations,
            result.evaluations,ended-started,log,{"initialization_s":prepared-started,"optimization_s":ended-prepared},
            self.init_times.copy(),self.init_points.copy())


class BackupTrajOpt:
    def __init__(self,config=None):self.config=config or OptimizerConfig.backup()

    def prepare(self,exp_traj,t0,te,heu_ts,heu_end,heu_duration,corridor,init_times=None,init_points=None):
        c=self.config;self.exp_traj,self.min_ts,self.max_ts=exp_traj,float(t0),float(te);self.ts=float(heu_ts)
        self.hpoly=_planes(corridor).copy();self.hpoly/=np.linalg.norm(self.hpoly[:,:3],axis=1)[:,None]
        self.vpolys=[_vertex_coordinates(self.hpoly)]*c.piece_num
        self.head=exp_traj.state(heu_ts);self.tail=np.zeros((3,4));self.tail[:,0]=heu_end
        self.init_points=self.head[:,0]+np.arange(1,c.piece_num+1)[:,None]*(self.tail[:,0]-self.head[:,0])/c.piece_num
        self.init_times=np.full(c.piece_num,heu_duration/c.piece_num)
        if init_times is not None:
            self.init_times=np.asarray(init_times,dtype=float).copy();self.init_points=np.asarray(init_points,dtype=float).copy()
        self.temporal_dim=1 if c.uniform_time_en else c.piece_num
        tau=inverse_t(np.array([heu_duration]) if c.uniform_time_en else self.init_times)
        xi=self.init_points.ravel() if c.pos_constraint_type==1 else _inverse_points(self.init_points,self.vpolys)
        tau_s=np.arcsin((heu_ts-(t0+te)/2)/((te-t0)/2))
        self.initial_x=np.r_[tau,xi,tau_s];self.minco=MINCO(self.head,self.tail,c.piece_num,4)
        self.problem=_pack_problem(self,True)
        return self

    def objective(self,x):return _evaluate(self,x,True)

    def solve_prepared(self,initial=None):return _solve(self,initial)

    def optimize(self,exp_traj,t0,te,heu_ts,heu_end,heu_duration,corridor,init_times=None,init_points=None):
        started=perf_counter()
        try:self.prepare(exp_traj,t0,te,heu_ts,heu_end,heu_duration,corridor,init_times,init_points)
        except (GeometryError,ValueError) as exc:
            return OptimizationResult(None,False,str(exc),-1,float('inf'),elapsed_s=perf_counter()-started,switch_time=heu_ts)
        prepared=perf_counter();result=_solve(self)
        _,_,coeffs,times,_,ts,log=objective_full(result.x,self.problem);self.ts=ts;self.penalty_log=log
        c=self.config
        violation=((c.penna_pos>0 and log[1]>.2) or (c.penna_vel>0 and log[2]>c.max_vel*c.penna_margin)
            or (c.penna_acc>0 and log[3]>c.max_acc*c.penna_margin) or (c.penna_omg>0 and log[6]>c.max_omg*c.penna_margin)
            or (c.penna_thr>0 and log[7]>c.max_acc*c.penna_margin))
        success=result.success and not violation;trajectory=PiecewisePolynomial(times,coeffs) if success else None
        if success:
            success=not ((c.penna_pos>0 and log[1]>c.penna_pos*.05)
                or (c.penna_vel>0 and trajectory.max_derivative_norm(1)>1.2*c.max_vel)
                or (c.penna_acc>0 and trajectory.max_derivative_norm(2)>1.2*c.max_acc))
            if not success:violation=True;trajectory=None
        ended=perf_counter()
        return OptimizationResult(trajectory,success,"SUPER penalty violation" if violation else result.message,
            -1 if violation else result.status,result.cost if success else float('inf'),result.iterations,
            result.evaluations,ended-started,log,{"initialization_s":prepared-started,"optimization_s":ended-prepared},
            self.init_times.copy(),self.init_points.copy(),float(ts))


class YawTrajOpt:
    def __init__(self,yaw_dot_max=3.):self.yaw_dot_max=float(yaw_dot_max)

    def time_allocation(self,duration):
        dt=np.pi/self.yaw_dot_max
        if duration<2*dt:return np.array([duration])
        count=int(np.ceil((duration-2*dt)/dt))
        times=np.r_[dt,np.full(count,(duration-2*dt)/count) if count else [],dt]
        if len(times)==3 and times[1]<times[0]/3:times=np.full(2,duration/2)
        return times

    @staticmethod
    def _normalize(last,current):
        if np.isnan(current):return last
        while abs(last-current)>np.pi:
            current+=2*np.pi if last>current else -2*np.pi
        return current

    def optimize(self,init,goal,position,order=3,free_start=False,free_goal=True):
        init,goal=np.array(init,dtype=float),np.array(goal,dtype=float);duration=position.duration
        if free_start:
            ti=min(.5,duration);direction=position.evaluate(ti)-position.evaluate(0.)
            while np.linalg.norm(direction)<.5 and ti<duration:
                ti+=.1;direction=position.evaluate(ti)-position.evaluate(0.)
            init[0]=np.arctan2(direction[1],direction[0])
        if free_goal:
            ti=max(0.,duration-.5);direction=position.evaluate(duration)-position.evaluate(ti)
            while np.linalg.norm(direction)<.5 and ti>0:
                ti-=.1;direction=position.evaluate(duration)-position.evaluate(ti)
            goal[0]=np.arctan2(direction[1],direction[0])
        times=self.time_allocation(duration);waypoints=[];last=init[0]
        for t in np.cumsum(times)[:-1]:
            if t+.5>=duration:direction=position.evaluate(duration)-position.evaluate(max(0.,duration-.5))
            else:direction=position.evaluate(t+.5)-position.evaluate(t)
            current=self._normalize(last,np.arctan2(direction[1],direction[0])) if np.linalg.norm(direction)>.1 else last
            waypoints.append([current]);last=current
        goal[0]=self._normalize(last,goal[0]);s=(order+1)//2
        if order not in (3,5,7):raise ValueError("yaw polynomial order must be 3, 5 or 7")
        yaw=MINCO(init[None,:s],goal[None,:s],len(times),s).set_parameters(waypoints,times).trajectory()
        coeffs=np.zeros((len(times),8,3));coeffs[:,:order+1,0]=yaw.coefficients[:,:,0]
        return PiecewisePolynomial(times,coeffs)


def warmup():
    """Compile/load both numerical signatures before the worker announces ready."""
    head=np.zeros((3,4));head[:,0]=[0.,0.,1.];tail=head.copy();tail[0,0]=1.
    poly=np.column_stack([np.r_[-np.eye(3),np.eye(3)],[-2.,-2.,0.,-3.,-2.,-2.]])
    exp=ExpTrajOpt();exp.prepare(head,tail,[[0.,0.,1.],[1.,0.,1.]],[0.,1.],[poly])
    exp.solve_prepared();trajectory=MINCO(head,tail,1,4).set_parameters([],np.array([2.])).trajectory()
    backup=BackupTrajOpt();backup.prepare(trajectory,.1,1.5,.5,[.8,0.,1.],1.,poly)
    backup.solve_prepared()

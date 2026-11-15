"""SUPER's cautious L-BFGS and Lewis–Overton line search in Python.

The original scalar loops are CPU JIT compiled by Numba. Source: SUPER
2ad3419 utils/optimization/lbfgs.h and src/utils/lbfgs.cpp; original algorithm
Copyright (c) 2021 Zhepei Wang, MIT license. No altered convergence criteria.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit


@dataclass(frozen=True)
class LBFGSParameters:
    mem_size: int = 8
    g_epsilon: float = 1e-5
    past: int = 3
    delta: float = 1e-6
    max_iterations: int = 0
    max_linesearch: int = 64
    min_step: float = 1e-20
    max_step: float = 1e20
    f_dec_coeff: float = 1e-4
    s_curv_coeff: float = .9
    cautious_factor: float = 1e-6
    machine_prec: float = 1e-16


@dataclass
class LBFGSResult:
    x: np.ndarray
    cost: float
    gradient: np.ndarray
    status: int
    iterations: int
    evaluations: int

    @property
    def success(self):return self.status>=0

    @property
    def message(self):
        return {0:"gradient convergence",1:"relative cost convergence",
            -1012:"nonfinite objective",-1011:"minimum line-search step",
            -1010:"maximum line-search step",-1009:"line-search limit",
            -1008:"iteration limit",-1007:"line-search interval too small",
            -1006:"nonpositive line-search step",-1005:"non-descent direction"}.get(self.status,str(self.status))


@njit(cache=True,fastmath=False,error_model="numpy",inline="always")
def _dot(a,b):
    value=0.
    for i in range(len(a)):value+=a[i]*b[i]
    return value


# Dispatcher-valued arguments cannot be reused safely from Numba's disk cache.
@njit(cache=False,fastmath=False,error_model="numpy")
def _line_search(fun,args,xp,gp,finit,direction,step,max_linesearch,min_step,max_step,f_dec,s_curv,machine_prec):
    if not step>0.:
        return xp,finit,gp,step,-1006,0
    dginit=_dot(gp,direction)
    if dginit>0.:
        return xp,finit,gp,step,-1005,0
    dgtest=f_dec*dginit;dstest=s_curv*dginit
    mu,nu=0.,max_step;bracketed=False;touched=False
    for count in range(1,max_linesearch+1):
        x=xp+step*direction;f,g=fun(x,args)
        if not np.isfinite(f):return x,f,g,step,-1012,count
        if f>finit+step*dgtest:
            nu=step;bracketed=True
        elif _dot(g,direction)<dstest:mu=step
        else:return x,f,g,step,count,count
        if count==max_linesearch:return x,f,g,step,-1009,count
        if bracketed and nu-mu<machine_prec*nu:return x,f,g,step,-1007,count
        step=.5*(mu+nu) if bracketed else 2.*step
        if step<min_step:return x,f,g,step,-1011,count
        if step>max_step:
            if touched:return x,f,g,step,-1010,count
            touched=True;step=max_step
    return xp,finit,gp,step,-1009,max_linesearch


@njit(cache=False,fastmath=False,error_model="numpy")
def _minimize(fun,initial,args,mem_size,g_epsilon,past,delta,max_iterations,max_linesearch,
              min_step,max_step,f_dec,s_curv,cautious_factor,machine_prec):
    x=initial.copy();f,g=fun(x,args);calls=1
    if np.max(np.abs(g))/max(1.,np.max(np.abs(x)))<=g_epsilon:return x,f,g,0,0,calls
    direction=-g;step=1./np.sqrt(_dot(direction,direction))
    pf=np.empty(max(1,past));pf[0]=f
    s_mem=np.zeros((mem_size,len(x)));y_mem=np.zeros_like(s_mem)
    ys_mem=np.zeros(mem_size);alpha=np.zeros(mem_size)
    end,bound,k=0,0,1
    while True:
        xp,gp,fp=x,g,f
        step=step if step<max_step else .5*max_step
        x,f,g,step,status,count=_line_search(fun,args,xp,gp,fp,direction,step,
            max_linesearch,min_step,max_step,f_dec,s_curv,machine_prec)
        calls+=count
        if status<0:return xp,f,gp,status,k,calls
        if np.max(np.abs(g))/max(1.,np.max(np.abs(x)))<g_epsilon:return x,f,g,0,k,calls
        if past:
            if k>=past and abs(pf[k%past]-f)/max(1.,abs(f))<delta:return x,f,g,1,k,calls
            pf[k%past]=f
        if max_iterations and k>=max_iterations:return x,f,g,-1008,k,calls
        k+=1
        for i in range(len(x)):s_mem[end,i]=x[i]-xp[i];y_mem[end,i]=g[i]-gp[i]
        ys=_dot(y_mem[end],s_mem[end]);yy=_dot(y_mem[end],y_mem[end]);ys_mem[end]=ys
        direction=-g
        cautious=_dot(s_mem[end],s_mem[end])*np.sqrt(_dot(gp,gp))*cautious_factor
        if ys>cautious:
            bound=min(mem_size,bound+1);end=(end+1)%mem_size;j=end
            for _ in range(bound):
                j=(j+mem_size-1)%mem_size;alpha[j]=_dot(s_mem[j],direction)/ys_mem[j]
                for i in range(len(x)):direction[i]-=alpha[j]*y_mem[j,i]
            for i in range(len(x)):direction[i]*=ys/yy
            for _ in range(bound):
                beta=_dot(y_mem[j],direction)/ys_mem[j]
                for i in range(len(x)):direction[i]+=(alpha[j]-beta)*s_mem[j,i]
                j=(j+1)%mem_size
        step=1.


def minimize(fun,initial,parameters=None,arguments=()):
    """Optimize a compiled Python objective ``fun(x, arguments) -> (f,g)``."""
    p=parameters or LBFGSParameters()
    values=_minimize(fun,np.asarray(initial,dtype=np.float64),arguments,p.mem_size,p.g_epsilon,
        p.past,p.delta,p.max_iterations,p.max_linesearch,p.min_step,p.max_step,
        p.f_dec_coeff,p.s_curv_coeff,p.cautious_factor,p.machine_prec)
    return LBFGSResult(*values)

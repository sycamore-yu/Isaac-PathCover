"""Python-defined GPU updates for structured ADMM, paired with nvmath/cuDSS.

One GPU program owns a QP. Completed QPs return before loading their matrices.
Triton is supplied by PyTorch; no C++ binding, CUDA source or build step is used.
The update equations follow the retained MIT-licensed TurboMPC reference.
"""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gamma_kernel(X,Z,Y,Y0,YD,RHO,Q,A0,AM,AP,G,C0,CD,RHS,STATUS,
                  H:tl.constexpr,N:tl.constexpr,NX:tl.constexpr,N0:tl.constexpr,
                  M:tl.constexpr,SIGMA:tl.constexpr,RF:tl.constexpr,BLOCK:tl.constexpr):
    b = tl.program_id(0)
    if tl.load(STATUS+b) == -1:
        i = tl.arange(0,BLOCK)
        t,j = i//N,i%N
        valid = i < H*N
        rho = tl.load(RHO+b)
        rf = rho*RF
        value = SIGMA*tl.load(X+b*H*N+i,valid,other=0)-tl.load(Q+b*H*N+i,valid,other=0)
        eq = tl.full((BLOCK,),0,tl.float64)
        for k in tl.static_range(N0):
            a = tl.load(A0+b*N0*N+k*N+j,valid & (t==0),other=0)
            y = rf*tl.load(C0+b*N0+k)-tl.load(Y0+b*N0+k)
            eq = tl.fma(a,y,eq)
        for k in tl.static_range(NX):
            row = (t-1)*NX+k
            a = tl.load(AP+b*(H-1)*NX*N+row*N+j,valid & (t>0),other=0)
            y = rf*tl.load(CD+b*(H-1)*NX+row,valid & (t>0),other=0)
            y -= tl.load(YD+b*(H-1)*NX+row,valid & (t>0),other=0)
            eq = tl.fma(a,y,eq)
        for k in tl.static_range(NX):
            row = t*NX+k
            a = tl.load(AM+b*(H-1)*NX*N+row*N+j,valid & (t<H-1),other=0)
            y = rf*tl.load(CD+b*(H-1)*NX+row,valid & (t<H-1),other=0)
            y -= tl.load(YD+b*(H-1)*NX+row,valid & (t<H-1),other=0)
            eq = tl.fma(a,y,eq)
        value += eq
        inequality = tl.full((BLOCK,),0,tl.float64)
        for k in tl.static_range(M):
            row = t*M+k
            a = tl.load(G+b*H*M*N+row*N+j,valid,other=0)
            y = rho*tl.load(Z+b*H*M+row,valid,other=0)-tl.load(Y+b*H*M+row,valid,other=0)
            inequality = tl.fma(a,y,inequality)
        tl.store(RHS+b*H*N+i,value+inequality,valid)


@triton.jit
def _positive_root(offset,kappa):
    root = tl.sqrt(offset*offset+4*kappa)
    return tl.where(offset>=0,.5*(offset+root),2*kappa/(root-offset))


@triton.jit
def _barrier(value,lower,upper,kappa):
    lf,uf = tl.abs(lower)<float('inf'),tl.abs(upper)<float('inf')
    lo,hi = tl.where(lf,lower,0.),tl.where(uf,upper,1.)
    mid = lo+.5*(hi-lo)
    inside_lo,inside_hi = libdevice.nextafter(lo,hi),libdevice.nextafter(hi,lo)
    estimate = tl.where(value<=mid,tl.minimum(lo+_positive_root(value-lo,kappa),mid),
                        tl.maximum(hi-_positive_root(hi-value,kappa),mid))
    estimate = tl.minimum(tl.maximum(estimate,inside_lo),inside_hi)
    for _ in tl.static_range(7):
        dl,du = estimate-lo,hi-estimate
        residual = estimate-value+kappa*(1/du-1/dl)
        derivative = 1+kappa*(1/(du*du)+1/(dl*dl))
        step = -residual/derivative
        fraction = tl.where(step<0,.99*dl/(-step),tl.where(step>0,.99*du/step,1.))
        estimate = tl.minimum(tl.maximum(estimate+tl.minimum(fraction,1.)*step,inside_lo),inside_hi)
    infinity = tl.full((),float('inf'),tl.float64)
    left = tl.maximum(lo+_positive_root(value-lo,kappa),libdevice.nextafter(lo,infinity))
    right = tl.minimum(hi-_positive_root(hi-value,kappa),libdevice.nextafter(hi,-infinity))
    return tl.where(lf & uf,estimate,tl.where(lf,left,tl.where(uf,right,value)))


@triton.jit
def _post_kernel(X,Z,Y,Y0,YD,XI,RHO,SOL,A0,AM,AP,G,C0,CD,LO,HI,STATUS,
                 H:tl.constexpr,N:tl.constexpr,NX:tl.constexpr,N0:tl.constexpr,
                 M:tl.constexpr,ALPHA:tl.constexpr,RF:tl.constexpr,
                 SLACK:tl.constexpr,WEIGHT:tl.constexpr,BARRIER:tl.constexpr,
                 KAPPA:tl.constexpr,BLOCK:tl.constexpr):
    b = tl.program_id(0)
    if tl.load(STATUS+b) == -1:
        i = tl.arange(0,BLOCK)
        rho = tl.load(RHO+b)
        # Initial and between-stage equalities use the unrelaxed linear solve.
        c0 = tl.full((BLOCK,),0,tl.float64)
        cd = tl.full((BLOCK,),0,tl.float64)
        t = i//NX
        for j in tl.static_range(N):
            a = tl.load(A0+b*N0*N+i*N+j,i<N0,other=0)
            v = tl.load(SOL+b*H*N+j)
            c0 = tl.fma(a,v,c0)
            am = tl.load(AM+b*(H-1)*NX*N+i*N+j,i<(H-1)*NX,other=0)
            ap = tl.load(AP+b*(H-1)*NX*N+i*N+j,i<(H-1)*NX,other=0)
            vm = tl.load(SOL+b*H*N+t*N+j,i<(H-1)*NX,other=0)
            vp = tl.load(SOL+b*H*N+(t+1)*N+j,i<(H-1)*NX,other=0)
            cd += am*vm+ap*vp
        y0 = tl.load(Y0+b*N0+i,i<N0,other=0)
        yd = tl.load(YD+b*(H-1)*NX+i,i<(H-1)*NX,other=0)
        tl.store(Y0+b*N0+i,y0+rho*RF*ALPHA*(c0-tl.load(C0+b*N0+i,i<N0,other=0)),i<N0)
        tl.store(YD+b*(H-1)*NX+i,yd+rho*RF*ALPHA*(cd-tl.load(CD+b*(H-1)*NX+i,i<(H-1)*NX,other=0)),i<(H-1)*NX)
        if M > 0:
            gx = tl.full((BLOCK,),0,tl.float64)
            t = i//M
            for j in tl.static_range(N):
                a = tl.load(G+b*H*M*N+i*N+j,i<H*M,other=0)
                v = tl.load(SOL+b*H*N+t*N+j,i<H*M,other=0)
                gx = tl.fma(a,v,gx)
            z = tl.load(Z+b*H*M+i,i<H*M,other=0)
            y = tl.load(Y+b*H*M+i,i<H*M,other=0)
            lower = tl.load(LO+b*H*M+i,i<H*M,other=-float('inf'))
            upper = tl.load(HI+b*H*M+i,i<H*M,other=float('inf'))
            relaxed = ALPHA*gx+(1-ALPHA)*z
            tilted = relaxed+y/rho
            if BARRIER:
                scale = KAPPA/rho
                if SLACK:
                    scale = KAPPA*(1/rho+1/WEIGHT)
                projected = _barrier(tilted,lower,upper,scale)
            else:
                projected = tl.where(tilted<lower,lower,tilted)
                projected = tl.where(projected>upper,upper,projected)
            if SLACK:
                xi = rho/(WEIGHT+rho)*(projected-tilted)
                if BARRIER:
                    newz = projected-xi
                else:
                    fraction = WEIGHT/(WEIGHT+rho)
                    newz = (1-fraction)*tilted+fraction*projected
            else:
                xi,newz = tl.full((BLOCK,),0,tl.float64),projected
            tl.store(Y+b*H*M+i,y+rho*(relaxed-newz),i<H*M)
            tl.store(Z+b*H*M+i,newz,i<H*M)
            tl.store(XI+b*H*M+i,xi,i<H*M)
        old = tl.load(X+b*H*N+i,i<H*N,other=0)
        solved = tl.load(SOL+b*H*N+i,i<H*N,other=0)
        tl.store(X+b*H*N+i,ALPHA*solved+(1-ALPHA)*old,i<H*N)


def gamma(qp,state,rhs,status,*,sigma,rho_factor):
    x,z,y,y0,yd,_,rho = state
    b,h,n = qp.q.shape
    nx,n0,m = qp.A_minus.shape[-2],qp.A0.shape[-2],qp.G.shape[-2]
    _gamma_kernel[(b,)](x,z,y,y0,yd,rho,qp.q,qp.A0,qp.A_minus,qp.A_plus,qp.G,
                        qp.c0,qp.c_dyn,rhs,status,h,n,nx,n0,m,sigma,rho_factor,
                        triton.next_power_of_2(h*n),num_warps=8)


def post_solve(qp,state,solved,status,*,alpha,rho_factor,use_slack,slack_weight,barrier,kappa):
    x,z,y,y0,yd,xi,rho = state
    b,h,n = qp.q.shape
    nx,n0,m = qp.A_minus.shape[-2],qp.A0.shape[-2],qp.G.shape[-2]
    block = triton.next_power_of_2(max(h*n,(h-1)*nx,n0,h*m))
    _post_kernel[(b,)](x,z,y,y0,yd,xi,rho,solved,qp.A0,qp.A_minus,qp.A_plus,qp.G,
                       qp.c0,qp.c_dyn,qp.lower,qp.upper,status,h,n,nx,n0,m,
                       alpha,rho_factor,use_slack,slack_weight,barrier,kappa,block,num_warps=8)


@triton.jit
def _snapshot(X, Y, Y0, YD, XI, PX, PY, PY0, PYD, PXI, H: tl.constexpr, N: tl.constexpr, NX: tl.constexpr, N0: tl.constexpr, M: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    tl.store(PX + b * H * N + i, tl.load(X + b * H * N + i, i < H * N, other=0), i < H * N)
    tl.store(PY + b * H * M + i, tl.load(Y + b * H * M + i, i < H * M, other=0), i < H * M)
    tl.store(PY0 + b * N0 + i, tl.load(Y0 + b * N0 + i, i < N0, other=0), i < N0)
    tl.store(PYD + b * (H - 1) * NX + i, tl.load(YD + b * (H - 1) * NX + i, i < (H - 1) * NX, other=0), i < (H - 1) * NX)
    tl.store(PXI + b * H * M + i, tl.load(XI + b * H * M + i, i < H * M, other=0), i < H * M)

@triton.jit
def _triangular(LU, PIV, RHS, SOL, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    value = tl.load(RHS + b * SIZE + i, i < SIZE, other=0)
    for k in range(SIZE):
        pivot = tl.load(PIV + b * SIZE + k) - 1
        a = tl.sum(tl.where(i == k, value, 0.0), axis=0)
        p = tl.sum(tl.where(i == pivot, value, 0.0), axis=0)
        value = tl.where(i == k, p, tl.where(i == pivot, a, value))
    for k in range(SIZE):
        a = tl.sum(tl.where(i == k, value, 0.0), axis=0)
        column = tl.load(LU + b * SIZE * SIZE + k * SIZE + i, i < SIZE, other=0)
        value = tl.where(i > k, value - column * a, value)
    for offset in range(SIZE):
        k = SIZE - 1 - offset
        diagonal = tl.load(LU + b * SIZE * SIZE + k * SIZE + k)
        a = tl.sum(tl.where(i == k, value, 0.0), axis=0) / diagonal
        column = tl.load(LU + b * SIZE * SIZE + k * SIZE + i, i < SIZE, other=0)
        value = tl.where(i == k, a, tl.where(i < k, value - column * a, value))
    tl.store(SOL + b * SIZE + i, value, i < SIZE)

@triton.jit
def _segment_kernel(X, Z, Y, Y0, YD, XI, RHO, Q, A0, AM, AP, G, C0, CD, LO, HI, RHS, SOL, LU, PIV, STATUS, PX, PY, PY0, PYD, PXI, iterations, SAVE: tl.constexpr, H: tl.constexpr, N: tl.constexpr, NX: tl.constexpr, N0: tl.constexpr, M: tl.constexpr, SIGMA: tl.constexpr, RF: tl.constexpr, ALPHA: tl.constexpr, SLACK: tl.constexpr, WEIGHT: tl.constexpr, BARRIER: tl.constexpr, KAPPA: tl.constexpr, BX: tl.constexpr, BP: tl.constexpr):
    active = tl.load(STATUS + tl.program_id(0)) == -1
    if SAVE:
        if not active:
            _snapshot(X, Y, Y0, YD, XI, PX, PY, PY0, PYD, PXI, H, N, NX, N0, M, BP)
    if active:
        for iteration in range(iterations):
            if SAVE:
                if iteration + 1 == iterations:
                    _snapshot(X, Y, Y0, YD, XI, PX, PY, PY0, PYD, PXI, H, N, NX, N0, M, BP)
            _gamma_kernel(X, Z, Y, Y0, YD, RHO, Q, A0, AM, AP, G, C0, CD, RHS, STATUS, H, N, NX, N0, M, SIGMA, RF, BX)
            tl.debug_barrier()
            _triangular(LU, PIV, RHS, SOL, H * N, BX)
            tl.debug_barrier()
            _post_kernel(X, Z, Y, Y0, YD, XI, RHO, SOL, A0, AM, AP, G, C0, CD, LO, HI, STATUS, H, N, NX, N0, M, ALPHA, RF, SLACK, WEIGHT, BARRIER, KAPPA, BP)
            tl.debug_barrier()

def run_segment(workspace, length, save_previous):
    qp, cfg = (workspace.qp, workspace.config)
    b, h, n = qp.q.shape
    nx, n0, m = (qp.A_minus.shape[-2], qp.A0.shape[-2], qp.G.shape[-2])
    _segment_kernel[b,](*workspace.state, qp.q, qp.A0, qp.A_minus, qp.A_plus, qp.G, qp.c0, qp.c_dyn, qp.lower, qp.upper, workspace.linear.rhs, workspace.linear.solution, workspace.linear.lu, workspace.linear.pivots, workspace.status, *workspace.previous, length, save_previous, h, n, nx, n0, m, cfg.sigma, cfg.rho_f_factor, cfg.alpha, cfg.use_slack, cfg.slack_weight, cfg.constraint_handling == 1, cfg.log_barrier_kappa, triton.next_power_of_2(h * n), triton.next_power_of_2(max(h * n, n0, (h - 1) * nx, h * m)), num_warps=4)

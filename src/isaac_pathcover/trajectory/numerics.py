"""CPU numerical kernels for SUPER's Python trajectory optimizers.

The kernels retain the upstream banded LU, quadrature, analytic derivatives
and floating-point rules. Numba compiles this Python source; fastmath is off.
SUPER 2ad3419, Copyright 2025 Yunfan Ren (LGPL-3.0-or-later).
MINCO and flatness mathematics: Copyright 2021 Zhepei Wang (MIT).
"""
import numpy as np
from numba import njit


FACTORS = np.array([[1.,1,1,1,1,1,1,1],
 [0.,1,2,3,4,5,6,7],[0.,0,2,6,12,20,30,42],
 [0.,0,0,6,24,60,120,210],[0.,0,0,0,24,120,360,840],
 [0.,0,0,0,0,120,720,2520],[0.,0,0,0,0,0,720,5040],
 [0.,0,0,0,0,0,0,5040]])
END_DERIVATIVES = np.array([4,5,6,0,0,1,2,3],dtype=np.int64)


@njit(cache=True,fastmath=False,error_model="numpy",inline="always")
def dot3(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]


@njit(cache=True,fastmath=False,error_model="numpy",inline="always")
def smooth(value,mu):
    if value<0.:
        return 0.,0.
    if value>mu:
        return value-.5*mu,1.
    z=value/mu
    return (mu-.5*value)*z*z*z,z*z*(-.5*z+3.*(mu-.5*value)/mu)


@njit(cache=True,fastmath=False,error_model="numpy",inline="always")
def positive_time(tau):
    if tau>0.:
        return (.5*tau+1.)*tau+1.
    return 1./((.5*tau-1.)*tau+1.)


@njit(cache=True,fastmath=False,error_model="numpy",inline="always")
def time_derivative(tau):
    if tau>0.:
        return tau+1.
    den=(.5*tau-1.)*tau+1.
    return (1.-tau)/(den*den)


@njit(cache=True,fastmath=False,error_model="numpy")
def trajectory_state(t,durations,coefficients):
    index=0
    while index<len(durations)-1 and t>durations[index]:
        t-=durations[index];index+=1
    powers=np.empty(8);powers[0]=1.
    for k in range(1,8):powers[k]=powers[k-1]*t
    state=np.zeros((5,3))
    for d in range(5):
        for k in range(d,8):
            for axis in range(3):
                state[d,axis]+=FACTORS[d,k]*powers[k-d]*coefficients[index,k,axis]
    return state


@njit(cache=True,fastmath=False,error_model="numpy")
def band_solve(a,b,transpose=False):
    n=len(b)
    if transpose:
        for j in range(n):
            for axis in range(3):b[j,axis]/=a[j,j]
            for i in range(j+1,min(j+8,n-1)+1):
                if a[j,i]!=0.:
                    for axis in range(3):b[i,axis]-=a[j,i]*b[j,axis]
        for j in range(n-1,-1,-1):
            for i in range(max(0,j-8),j):
                if a[j,i]!=0.:
                    for axis in range(3):b[i,axis]-=a[j,i]*b[j,axis]
    else:
        for j in range(n):
            for i in range(j+1,min(j+8,n-1)+1):
                if a[i,j]!=0.:
                    for axis in range(3):b[i,axis]-=a[i,j]*b[j,axis]
        for j in range(n-1,-1,-1):
            for axis in range(3):b[j,axis]/=a[j,j]
            for i in range(max(0,j-8),j):
                if a[i,j]!=0.:
                    for axis in range(3):b[i,axis]-=a[i,j]*b[j,axis]
    return b


@njit(cache=True,fastmath=False,error_model="numpy")
def minco_s4(head,tail,points,times):
    n=len(times);width=8*n
    a=np.zeros((width,width));b=np.zeros((width,3))
    for d in range(4):
        a[d,d]=FACTORS[d,d]
        for axis in range(3):b[d,axis]=head[axis,d];b[width-4+d,axis]=tail[axis,d]
    for i in range(n):
        powers=np.empty(8);powers[0]=1.
        for k in range(1,8):powers[k]=powers[k-1]*times[i]
        count=8 if i<n-1 else 4
        for r in range(count):
            d=END_DERIVATIVES[r] if i<n-1 else r
            row=8*i+4+r
            for k in range(d,8):a[row,8*i+k]=FACTORS[d,k]*powers[k-d]
            if i<n-1 and r!=3:a[row,8*(i+1)+d]=-FACTORS[d,d]
        if i<n-1:
            for axis in range(3):b[8*i+7,axis]=points[i,axis]
    for k in range(width-1):
        for i in range(k+1,min(k+8,width-1)+1):
            if a[i,k]!=0.:a[i,k]/=a[k,k]
        for j in range(k+1,min(k+8,width-1)+1):
            value=a[k,j]
            if value!=0.:
                for i in range(k+1,min(k+8,width-1)+1):
                    if a[i,k]!=0.:a[i,j]-=a[i,k]*value
    return band_solve(a,b).reshape(n,8,3),a


@njit(cache=True,fastmath=False,error_model="numpy")
def energy_partials(c,times):
    cost=0.;gc=np.zeros_like(c);gt=np.zeros(len(times))
    for i in range(len(times)):
        t1=times[i];t2=t1*t1;t3=t2*t1;t4=t2*t2;t5=t4*t1;t6=t4*t2;t7=t4*t3
        b4,b5,b6,b7=c[i,4],c[i,5],c[i,6],c[i,7]
        cost+=(576.*dot3(b4,b4)*t1+2880.*dot3(b4,b5)*t2+4800.*dot3(b5,b5)*t3
            +5760.*dot3(b4,b6)*t3+21600.*dot3(b5,b6)*t4+10080.*dot3(b4,b7)*t4
            +25920.*dot3(b6,b6)*t5+40320.*dot3(b5,b7)*t5+100800.*dot3(b6,b7)*t6
            +100800.*dot3(b7,b7)*t7)
        for axis in range(3):
            gc[i,7,axis]=10080.*b4[axis]*t4+40320.*b5[axis]*t5+100800.*b6[axis]*t6+201600.*b7[axis]*t7
            gc[i,6,axis]=5760.*b4[axis]*t3+21600.*b5[axis]*t4+51840.*b6[axis]*t5+100800.*b7[axis]*t6
            gc[i,5,axis]=2880.*b4[axis]*t2+9600.*b5[axis]*t3+21600.*b6[axis]*t4+40320.*b7[axis]*t5
            gc[i,4,axis]=1152.*b4[axis]*t1+2880.*b5[axis]*t2+5760.*b6[axis]*t3+10080.*b7[axis]*t4
        gt[i]=(576.*dot3(b4,b4)+5760.*dot3(b4,b5)*t1+14400.*dot3(b5,b5)*t2
            +17280.*dot3(b4,b6)*t2+86400.*dot3(b5,b6)*t3+40320.*dot3(b4,b7)*t3
            +129600.*dot3(b6,b6)*t4+201600.*dot3(b5,b7)*t4+604800.*dot3(b6,b7)*t5
            +705600.*dot3(b7,b7)*t6)
    return cost,gc,gt


@njit(cache=True,fastmath=False,error_model="numpy")
def flatness_jacobian(v,a,j,yaw,yaw_rate,physical_parameters):
    """Thrust, quaternion, body rates and analytic Jacobians in v/a/j/yaw/yaw-rate."""
    mass,grav,dh,dv,cp,veps=physical_parameters
    eye=np.eye(11);iv=eye[:3];ia=eye[3:6];ij=eye[6:9]
    speed=np.sqrt(dot3(v,v)+veps);ds=np.zeros(11);ds[:3]=v/speed
    wt=1.+cp*speed;dwt=cp*ds
    w=wt*v;wj=wt*iv+np.outer(v,dwt)
    u=a+(dh/mass)*w;u[2]+=grav;uj=ia+(dh/mass)*wj
    norm=np.sqrt(dot3(u,u));z=u/norm;normj=z @ uj
    zj=(uj-np.outer(z,normj))/norm
    va=dot3(v,a);vaj=np.zeros(11);vaj[:3]=a;vaj[3:6]=v
    wt_dot=cp*va/speed;wt_dot_j=cp*(vaj/speed-va*ds/(speed*speed))
    dw=wt*a+wt_dot*v
    dwj=wt*ia+np.outer(a,dwt)+wt_dot*iv+np.outer(v,wt_dot_j)
    du=j+(dh/mass)*dw;duj=ij+(dh/mass)*dwj
    proj=dot3(z,du);projj=z @ duj+du @ zj
    dz=(du-z*proj)/norm
    dzj=(duj-zj*proj-np.outer(z,projj)-np.outer(dz,normj))/norm
    force=mass*a+dv*w;force[2]+=mass*grav;forcej=mass*ia+dv*wj
    thrust=dot3(z,force);thrustj=z @ forcej+force @ zj
    den=1.+z[2];ratio=dz[2]/den;ratioj=dzj[2]/den-dz[2]*zj[2]/(den*den)
    tilt_den=np.sqrt(2.*den);tilt_den_j=zj[2]/tilt_den
    tilt0=.5*tilt_den;tilt1=-z[1]/tilt_den;tilt2=z[0]/tilt_den
    tilt0j=.5*tilt_den_j
    tilt1j=-zj[1]/tilt_den+z[1]*tilt_den_j/(tilt_den*tilt_den)
    tilt2j=zj[0]/tilt_den-z[0]*tilt_den_j/(tilt_den*tilt_den)
    ch,sh=np.cos(.5*yaw),np.sin(.5*yaw)
    quat=np.array([tilt0*ch,tilt1*ch+tilt2*sh,tilt2*ch-tilt1*sh,tilt0*sh])
    quatj=np.empty((4,11))
    quatj[0]=tilt0j*ch;quatj[1]=tilt1j*ch+tilt2j*sh
    quatj[2]=tilt2j*ch-tilt1j*sh;quatj[3]=tilt0j*sh
    quatj[:,9]=.5*np.array([-tilt0*sh,-tilt1*sh+tilt2*ch,-tilt2*sh-tilt1*ch,tilt0*ch])
    h=dz[:2]-z[:2]*ratio;hj=dzj[:2]-zj[:2]*ratio-np.outer(z[:2],ratioj)
    cy,sy=np.cos(yaw),np.sin(yaw)
    omega=np.empty(3);omega[0]=h[0]*sy-h[1]*cy;omega[1]=h[0]*cy+h[1]*sy
    omj=np.empty((3,11));omj[0]=hj[0]*sy-hj[1]*cy;omj[1]=hj[0]*cy+hj[1]*sy
    omj[0,9]=omega[1];omj[1,9]=-omega[0]
    numer=z[1]*dz[0]-z[0]*dz[1]
    numerj=zj[1]*dz[0]+z[1]*dzj[0]-zj[0]*dz[1]-z[0]*dzj[1]
    omega[2]=numer/den+yaw_rate;omj[2]=numerj/den-numer*zj[2]/(den*den);omj[2,10]=1.
    return thrust,quat,omega,thrustj,quatj,omj


@njit(cache=True,fastmath=False,error_model="numpy")
def flat_penalty(v,a,j,cfg):
    thrust,quat,omega,thrustj,quatj,omj=flatness_jacobian(v,a,j,0.,0.,cfg[21:27])
    mean=.5*(cfg[12]+cfg[13]);radius=.5*abs(cfg[13]-cfg[12])
    vo=dot3(omega,omega)-cfg[11]*cfg[11];vt=(thrust-mean)**2-radius*radius
    fo,dfo=smooth(vo,cfg[5]);ft,dft=smooth(vt,cfg[5])
    gradient=2.*cfg[19]*dfo*(omega @ omj)+2.*cfg[20]*dft*(thrust-mean)*thrustj
    return cfg[19]*fo+cfg[20]*ft,gradient[:9],vo,vt


@njit(cache=True,fastmath=False,error_model="numpy")
def point_forward(xi,vpolys,vcounts):
    points=np.empty((len(vcounts),3));offset=0
    for i in range(len(vcounts)):
        count=vcounts[i];norm2=0.
        for k in range(count):norm2+=xi[offset+k]*xi[offset+k]
        for axis in range(3):
            points[i,axis]=vpolys[i,0,axis]
            for k in range(count-1):points[i,axis]+=vpolys[i,k+1,axis]*xi[offset+k]*xi[offset+k]/norm2
        offset+=count
    return points


@njit(cache=True,fastmath=False,error_model="numpy")
def point_backward(xi,vpolys,vcounts,gp):
    gradient=np.zeros_like(xi);cost=0.;offset=0
    for i in range(len(vcounts)):
        count=vcounts[i];norm2=0.
        for k in range(count):norm2+=xi[offset+k]*xi[offset+k]
        invnorm=1./np.sqrt(norm2);inner=0.
        for k in range(count-1):
            gradient[offset+k]=2.*dot3(vpolys[i,k+1],gp[i])*xi[offset+k]*invnorm
            inner+=gradient[offset+k]*xi[offset+k]*invnorm
        for k in range(count):gradient[offset+k]=(gradient[offset+k]-xi[offset+k]*invnorm*inner)*invnorm
        violation=norm2-1.
        if violation>0.:
            cost+=violation**3
            for k in range(count):gradient[offset+k]+=6.*violation*violation*xi[offset+k]
        offset+=count
    return cost,gradient


@njit(cache=True,fastmath=False,error_model="numpy")
def objective_full(x,args):
    cfg,head0,tail0,hpolys,hcounts,vpolys,vcounts,attractors,radii,exp_t,exp_c,ts_bounds=args
    backup=cfg[0]>0.;pos_type=int(cfg[1]);n=int(cfg[28]);uniform=backup and cfg[3]>0.
    dimt=1 if uniform else n
    tau=x[:dimt];xi=x[dimt:-1] if backup else x[dimt:]
    times=np.empty(n)
    if uniform:times[:]=positive_time(tau[0])/n
    else:
        for i in range(n):times[i]=positive_time(tau[i])
    points=xi.copy().reshape((-1,3)) if pos_type==1 else point_forward(xi,vpolys,vcounts)
    head=head0.copy();tail=tail0.copy();ts=0.;expstate=np.zeros((5,3))
    if backup:
        ts=ts_bounds[2]
        if cfg[7]>0.:ts=.5*(ts_bounds[1]-ts_bounds[0])*np.sin(x[-1])+.5*(ts_bounds[1]+ts_bounds[0])
        expstate=trajectory_state(ts,exp_t,exp_c)
        for d in range(4):
            for axis in range(3):head[axis,d]=expstate[d,axis];tail[axis,d]=0.
        tail[:,0]=points[-1]
    coeffs,lu=minco_s4(head,tail,points,times)
    cost=0.;gc=np.zeros_like(coeffs);gt=np.zeros(n);log=np.zeros(8)
    if cfg[2]==0.:
        cost,gc,gt=energy_partials(coeffs,times);log[0]=cost
    res=int(cfg[4]);frac=1./res
    powers=np.empty(8);bases=np.zeros((5,8));state=np.zeros((5,3));grad=np.zeros((4,3))
    for i in range(n):
        step=times[i]*frac
        for j in range(res+1):
            s=j*step;powers[0]=1.
            for k in range(1,8):powers[k]=powers[k-1]*s
            state.fill(0.)
            for d in range(5):
                for k in range(d,8):
                    bases[d,k]=FACTORS[d,k]*powers[k-d]
                    for axis in range(3):state[d,axis]+=bases[d,k]*coeffs[i,k,axis]
            grad.fill(0.);density=0.
            if cfg[14]>0.:
                for k in range(hcounts[i]):
                    violation=dot3(hpolys[i,k,:3],state[0])+hpolys[i,k,3]
                    if not backup:log[1]=max(log[1],violation)
                    f,df=smooth(violation,cfg[5]);density+=cfg[14]*f
                    for axis in range(3):grad[0,axis]+=cfg[14]*df*hpolys[i,k,axis]
            if not backup and cfg[18]>0.:
                is_waypoint=j==0 and i!=0
                is_end=j==res and i!=n-1
                if is_waypoint or is_end:
                    index=i if is_end else i-1
                    delta=state[0]-attractors[index]
                    violation=dot3(delta,delta)-radii[index]*radii[index]
                    log[5]=max(log[5],violation)
                    f,df=smooth(violation,cfg[5]);density+=cfg[18]*f
                    for axis in range(3):grad[0,axis]+=cfg[18]*df*2.*delta[axis]
            for d in range(1,4):
                bound=cfg[7+d];weight=cfg[14+d]
                violation=dot3(state[d],state[d])-bound*bound
                if weight>0.:
                    f,df=smooth(violation,cfg[5]);density+=weight*f
                    for axis in range(3):grad[d,axis]+=weight*df*2.*state[d,axis]
                if weight>0. or backup:log[1+d]=max(log[1+d],violation)
            if cfg[19]>0. and cfg[20]>0.:
                value,flatgrad,vo,vt=flat_penalty(state[1],state[2],state[3],cfg)
                density+=value;log[6]=max(log[6],vo);log[7]=max(log[7],vt)
                for d in range(1,4):
                    for axis in range(3):grad[d,axis]+=flatgrad[3*(d-1)+axis]
            node=.5 if j==0 or j==res else 1.
            for k in range(8):
                for axis in range(3):
                    gc[i,k,axis]+=(bases[0,k]*grad[0,axis]+bases[1,k]*grad[1,axis]+
                        bases[2,k]*grad[2,axis]+bases[3,k]*grad[3,axis])*node*step
            chain=dot3(grad[0],state[1])+dot3(grad[1],state[2])+dot3(grad[2],state[3])+dot3(grad[3],state[4])
            gt[i]+=chain*j*frac*node*step+node*frac*density
            cost+=node*step*density
    adj=band_solve(lu,gc.copy().reshape((8*n,3)),True)
    gp=np.empty_like(points)
    for i in range(n-1):gp[i]=adj[8*i+7]
    if backup:gp[-1]=adj[-4]
    end_state=np.zeros((8,3))
    for i in range(n):
        powers[0]=1.
        for k in range(1,8):powers[k]=powers[k-1]*times[i]
        end_state.fill(0.)
        for d in range(1,8):
            for k in range(d,8):
                for axis in range(3):end_state[d,axis]+=FACTORS[d,k]*powers[k-d]*coeffs[i,k,axis]
        count=8 if i<n-1 else 4
        for r in range(count):
            d=END_DERIVATIVES[r] if i<n-1 else r
            gt[i]-=dot3(adj[8*i+4+r],end_state[d+1])
        cost+=cfg[6]*times[i];gt[i]+=cfg[6]
    gradient=np.empty_like(x)
    if uniform:
        gradient[0]=gt.sum()/n*time_derivative(tau[0])
    else:
        for i in range(n):gradient[i]=gt[i]*time_derivative(tau[i])
    if pos_type==1:gradient[dimt:dimt+xi.size]=gp.ravel()
    else:
        extra,gxi=point_backward(xi,vpolys,vcounts,gp)
        cost+=extra;gradient[dimt:dimt+xi.size]=gxi
    if backup:
        gs=0.
        if cfg[7]>0.:
            cost+=cfg[7]*(ts_bounds[1]-ts);gs=-cfg[7]
            for d in range(4):gs+=dot3(adj[d],expstate[d+1])
            gs*=.5*(ts_bounds[1]-ts_bounds[0])*np.cos(x[-1])
        gradient[-1]=gs
    return cost,gradient,coeffs,times,points,ts,log


@njit(cache=True,fastmath=False,error_model="numpy")
def objective(x,args):
    cost,gradient,coeffs,times,points,ts,log=objective_full(x,args)
    return cost,gradient


@njit(cache=True,fastmath=False,error_model="numpy")
def point_objective(x,args):
    point,poly=args
    counts=np.array([len(poly)],dtype=np.int64);polys=poly.reshape((1,len(poly),3))
    delta=point_forward(x,polys,counts)[0]-point
    extra,g=point_backward(x,polys,counts,(2.*delta).reshape((1,3)))
    return dot3(delta,delta)+extra,g

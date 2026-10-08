"""Exact local velocity-duration transcription of constant-acceleration motion.

Fixed observation positions split the position recurrence into legs. Interior
node velocities and interval durations determine accelerations locally. The
original full objective, finite bounds and physical constraints are retained;
the controller optimizer independently audits the reconstructed full decision.
"""
import numpy as np
from scipy.optimize import Bounds, minimize


class VelocityMapping:
    def __init__(self,request,x0):
        self.request=request;self.n=len(request.initial_decision.durations);self.nodes=self.n+1
        self.nv=3*(self.n-1);self.size=self.nv+self.n
        self.fixed={i:np.array([p.x,p.y,p.z]) for i,p in request.fixed_node_positions}
        if 0 not in self.fixed or self.n not in self.fixed:raise ValueError('fixed launch and recovery positions required')
        self._cache=None

    def initial(self,x0):return np.r_[x0[3*self.nodes+3:6*self.nodes-3],x0[-self.n:]]

    def expand(self,z):
        if self._cache is not None and np.array_equal(z,self._cache[0]):return self._cache[1:]
        n,nodes,size=self.n,self.nodes,self.size
        vel=np.zeros((nodes,3));vel[1:-1]=z[:self.nv].reshape(n-1,3);duration=z[self.nv:]
        dv=np.zeros((nodes,3,size));dv[1:-1,:,:self.nv]=np.eye(self.nv).reshape(n-1,3,self.nv)
        pos=np.zeros_like(vel);dp=np.zeros_like(dv);pos[0]=self.fixed[0]
        acc=np.zeros((n,3));da=np.zeros((n,3,size))
        for i,t in enumerate(duration):
            acc[i]=(vel[i+1]-vel[i])/t
            da[i]=(dv[i+1]-dv[i])/t;da[i,:,self.nv+i]-=acc[i]/t
            if i+1 in self.fixed:pos[i+1]=self.fixed[i+1]
            else:
                pos[i+1]=pos[i]+.5*(vel[i]+vel[i+1])*t
                dp[i+1]=dp[i]+.5*(dv[i]+dv[i+1])*t
                dp[i+1,:,self.nv+i]+=.5*(vel[i]+vel[i+1])
        dt=np.zeros((n,size));dt[:,self.nv:]=np.eye(n)
        x=np.r_[pos.ravel(),vel.ravel(),acc.ravel(),duration]
        jac=np.vstack((dp.reshape(3*nodes,size),dv.reshape(3*nodes,size),da.reshape(3*n,size),dt))
        self._cache=(z.copy(),x,jac);return x,jac

    def equality_rows(self):return np.array([6*(i-1)+c for i in sorted(self.fixed) if i>0 for c in range(3)])


def minimize_reduced_dynamics(request,fun,x0,*,jac,bounds,constraints,**kw):
    m=VelocityMapping(request,x0);rows=m.equality_rows();expand=m.expand;cs=[]
    for c in constraints:
        if c['type']=='eq':
            cs.append({'type':'eq','fun':lambda z,c=c:c['fun'](expand(z)[0])[rows],
                'jac':lambda z,c=c:c['jac'](expand(z)[0])[rows]@expand(z)[1]})
        else:
            cs.append({'type':'ineq','fun':lambda z,c=c:c['fun'](expand(z)[0]),
                'jac':lambda z,c=c:c['jac'](expand(z)[0])@expand(z)[1]})
    lower=np.flatnonzero(np.isfinite(bounds.lb));upper=np.flatnonzero(np.isfinite(bounds.ub))
    cs.append({'type':'ineq','fun':lambda z:np.r_[expand(z)[0][lower]-bounds.lb[lower],bounds.ub[upper]-expand(z)[0][upper]],
        'jac':lambda z:np.vstack((expand(z)[1][lower],-expand(z)[1][upper]))})
    v0,v1=3*m.nodes+3,6*m.nodes-3
    raw=minimize(lambda z:fun(expand(z)[0]),m.initial(x0),jac=lambda z:jac(expand(z)[0])@expand(z)[1],
        bounds=Bounds(np.r_[bounds.lb[v0:v1],bounds.lb[-m.n:]],np.r_[bounds.ub[v0:v1],bounds.ub[-m.n:]]),constraints=cs,**kw)
    raw.reduced_variable_count=len(raw.x);raw.full_variable_count=len(x0);raw.x=expand(raw.x)[0]
    return raw


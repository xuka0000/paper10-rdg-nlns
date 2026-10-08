"""Replay a recorded deterministic rejection of the exact same request.

Solver failures remain failures. Recompute their physical audit and preserve
the original evidence; do not spend another solver budget on an implicit retry.
"""
from dataclasses import replace
from experiments_3d.data_upgrade import read_json
from Mask.geometry import Vector3
from Mask.trajectory_optimization import (TrajectoryDecision,TrajectoryOptimizationResult,
    audit_trajectory_decision,trajectory_objective,trajectory_expected_energy)


class FrozenFailureCache:
    def __init__(self,cache):self.cache=cache;self.rejected_hits=0

    def solve(self,request):
        key=self.cache.key(request);path=self.cache.directory/'failures'/f'{key}.json'
        if not path.exists():return self.cache.solve(request)
        record=read_json(path)
        if record['request_sha256']!=key or record['source_identity']!=self.cache.source_identity:
            raise ValueError('recorded rejected request identity changed')
        audit_status=record['audit']['status']
        if audit_status not in {'solver_failed','audit_failed'} or not record['decision']:
            raise ValueError('unknown failure requires explicit diagnosis')
        d=record['decision'];vectors=lambda field:tuple(Vector3(**v) for v in d[field])
        decision=TrajectoryDecision(vectors('positions'),vectors('velocities'),vectors('accelerations'),tuple(d['durations']))
        audit=audit_trajectory_decision(request,decision,audit_status!='solver_failed',record['audit']['message'])
        cap=getattr(request.dynamic_limits,'maximum_control_acceleration',None)
        if cap is not None and any(a.norm()**2>cap**2+request.config.inequality_tolerance for a in decision.accelerations):
            audit=replace(audit,admitted=False,status='audit_failed',message='native controller acceleration envelope violated')
        if audit.admitted or audit.status!=audit_status:
            raise ValueError('replayed rejection audit no longer agrees with original evidence')
        self.rejected_hits+=1
        return TrajectoryOptimizationResult(False,record['status'],audit.message,'recorded-rejection/exact-request',decision,
            trajectory_objective(request,decision),trajectory_expected_energy(request,decision),audit,())

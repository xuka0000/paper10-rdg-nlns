"""Add native control limits and interior clearance without relaxing audits."""
from dataclasses import dataclass, replace
import numpy as np
from scipy.optimize import minimize
from pathlib import Path
from experiments_3d.reduced_dynamics import minimize_reduced_dynamics

from Agent.optimization.scipy_trajectory_agent import (ScipyTrajectoryAgent, _trajectory_equality_jacobian,
    _trajectory_inequality_jacobian, _trajectory_objective_gradient, _solver_inequality_row_mask,
    _position_margin_jacobian, _is_time_scalable_out_and_back, _time_scaled_decision,
    _time_scale_interval_groups, _time_scale_group_bounds, _time_scale_dynamic_margins)
from Mask.flight_relation import FlightDynamicLimits
from Mask.trajectory_optimization import (TrajectoryOptimizationResult, audit_trajectory_decision,
    trajectory_equality_residuals, trajectory_inequality_margins, trajectory_objective, trajectory_expected_energy,
    _position_clearance_margins, encode_decision_vector, decode_decision_vector, decision_bounds)


@dataclass(frozen=True, slots=True)
class ControllerLimits(FlightDynamicLimits):
    maximum_control_acceleration: float = 4.0

    def __post_init__(self):
        FlightDynamicLimits.__post_init__(self)
        if not 0 < self.maximum_control_acceleration <= 5.0:
            raise ValueError("control envelope must not exceed native 5 m/s2 limit")

    def directional_accelerations(self, direction):
        a,b = FlightDynamicLimits.directional_accelerations(self,direction)
        return min(a,self.maximum_control_acceleration),min(b,self.maximum_control_acceleration)


def additional_margins(request, decision):
    cap=request.dynamic_limits.maximum_control_acceleration
    values=[cap*cap-a.norm()**2 for a in decision.accelerations]
    for i,duration in enumerate(decision.durations):
        for fraction in np.arange(1,request.config.dense_samples_per_interval)/request.config.dense_samples_per_interval:
            t=duration*fraction
            position=decision.positions[i]+decision.velocities[i].scaled(t)+decision.accelerations[i].scaled(.5*t*t)
            values.extend(_position_clearance_margins(request,position))
    return np.asarray(values)


def additional_jacobian(request, decision):
    intervals=len(decision.durations)
    nodes=intervals+1
    vo,ao,to=3*nodes,6*nodes,6*nodes+3*intervals
    jac=np.zeros((len(additional_margins(request,decision)),to+intervals))
    for i,a in enumerate(decision.accelerations):
        jac[i,ao+3*i:ao+3*i+3]=-2*np.array([a.x,a.y,a.z])
    row=intervals
    for i,duration in enumerate(decision.durations):
        for fraction in np.arange(1,request.config.dense_samples_per_interval)/request.config.dense_samples_per_interval:
            t=duration*fraction
            velocity=decision.velocities[i];acceleration=decision.accelerations[i]
            p=decision.positions[i]+velocity.scaled(t)+acceleration.scaled(.5*t*t)
            spatial=_position_margin_jacobian(request,p)
            jac[row:row+6,3*i:3*i+3]=spatial
            jac[row:row+6,vo+3*i:vo+3*i+3]=spatial*t
            jac[row:row+6,ao+3*i:ao+3*i+3]=spatial*(.5*t*t)
            dp=velocity.scaled(fraction)+acceleration.scaled(duration*fraction*fraction)
            jac[row:row+6,to+i]=spatial@np.array([dp.x,dp.y,dp.z])
            row+=6
    return jac


class ControllerAwareOptimizer(ScipyTrajectoryAgent):
    cache_source_files = (Path(__file__).with_name('reduced_dynamics.py'),)

    def solve(self,request):
        if not isinstance(request.dynamic_limits,ControllerLimits):
            raise ValueError("explicit ControllerLimits required")
        if _is_time_scalable_out_and_back(request):
            groups=_time_scale_interval_groups(request)
            bounds=_time_scale_group_bounds(request,groups)
            decode=lambda x:_time_scaled_decision(request,x,groups)
            raw=minimize(lambda x:trajectory_objective(request,decode(x)),np.asarray([b[0] for b in bounds]),
                method="SLSQP",bounds=bounds,constraints=({"type":"ineq","fun":lambda x:np.concatenate(
                    (_time_scale_dynamic_margins(request,decode(x)),additional_margins(request,decode(x))))},),
                options={"maxiter":self._config.max_iterations,"ftol":self._config.function_tolerance,"disp":False})
        else:
            mask=_solver_inequality_row_mask(request)
            decode=lambda x:decode_decision_vector(request,x)
            raw=minimize_reduced_dynamics(request,lambda x:trajectory_objective(request,decode(x)),encode_decision_vector(request.initial_decision),
                jac=lambda x:_trajectory_objective_gradient(request,decode(x)),method="SLSQP",bounds=decision_bounds(request),
                constraints=({"type":"eq","fun":lambda x:trajectory_equality_residuals(request,decode(x)),
                    "jac":lambda x:_trajectory_equality_jacobian(request,decode(x))},
                    {"type":"ineq","fun":lambda x:np.concatenate((trajectory_inequality_margins(request,decode(x))[mask],additional_margins(request,decode(x)))),
                    "jac":lambda x:np.vstack((_trajectory_inequality_jacobian(request,decode(x))[mask],additional_jacobian(request,decode(x))))}),
                options={"maxiter":self._config.max_iterations,"ftol":self._config.function_tolerance,"disp":False})
        decision=decode(raw.x)
        audit=audit_trajectory_decision(request,decision,bool(raw.success),str(raw.message))
        control_margin=min(request.dynamic_limits.maximum_control_acceleration**2-a.norm()**2 for a in decision.accelerations)
        if control_margin < -request.config.inequality_tolerance:
            audit=replace(audit,admitted=False,status="audit_failed",message="native controller acceleration envelope violated")
        objective=trajectory_objective(request,decision)
        energy=trajectory_expected_energy(request,decision)
        primitives=self._build_primitives(request,decision,audit,objective) if audit.admitted else ()
        solver = "SLSQP/controller-envelope/velocity-duration" if hasattr(raw,'reduced_variable_count') else "SLSQP/controller-envelope/time-scaling"
        return TrajectoryOptimizationResult(audit.admitted,audit.status,audit.message,solver,decision,objective,energy,audit,primitives)

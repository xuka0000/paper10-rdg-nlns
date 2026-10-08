"""Fixed-order causal dispatch: propagate actual travel/sortie delays.

No assignment, route, pose or UAV sequence is reoptimized. Unconstrained MILP
vehicle slack is replaced by the common earliest-dispatch rule. Ground travel
uses the actual 3D road evaluator; UAV durations come from native telemetry.
"""
from collections import defaultdict,deque
import math


def earliest_dispatch(nodes,edges,lower):
    following=defaultdict(list)
    indegree={n:0 for n in nodes}
    for before,after,duration in edges:
        if duration<0 or math.isnan(duration):raise ValueError('invalid execution lag')
        following[before].append((after,duration));indegree[after]+=1
    queue=deque(sorted(n for n in nodes if indegree[n]==0))
    times={n:float(lower.get(n,0.)) for n in nodes}
    visited=0
    while queue:
        before=queue.popleft();visited+=1
        for after,lag in following[before]:
            times[after]=max(times[after],times[before]+lag)
            indegree[after]-=1
            if not indegree[after]:queue.append(after)
    if visited!=len(nodes):raise ValueError('fixed route/UAV precedence contains a causal cycle')
    return times


def dispatch(snapshot,result,physical_build,manifest,native_audit):
    values=dict(result.variable_values)
    native={f['flight_id']:f for f in native_audit['flights']}
    flights={f['flight_id']:f for f in manifest['flights']}
    nodes=set();edges=[];lower={};routes={};gantt=[]
    physical_roads={(o.vehicle_id,o.road_leg_id):o for o in physical_build.road_options}
    stop_node={s.object_id:s.road_node_id for s in snapshot.stop_sites}
    for vehicle in snapshot.vehicles:
        used=[r for r in snapshot.road_legs if values.get(f'road_use[{vehicle.object_id},{r.object_id}]',0.)>.5]
        outgoing={r.source_id:r for r in used}
        if len(outgoing)!=len(used):raise ValueError('branching fixed vehicle route')
        start=snapshot.vehicle_state(vehicle.object_id).current_road_node_id
        sequence=[start];legs=[];current=start
        while current in outgoing:
            road=outgoing.pop(current);legs.append(road);current=road.target_id
            if current in sequence:raise ValueError('cyclic vehicle route')
            sequence.append(current)
        if outgoing:raise ValueError('disconnected selected road component')
        routes[vehicle.object_id]=(sequence,legs)
        for node in sequence:
            a=f'A:{vehicle.object_id}:{node}';d=f'D:{vehicle.object_id}:{node}'
            nodes.update((a,d));edges.append((a,d,0.))
        for road in legs:
            actual=physical_roads[(vehicle.object_id,road.object_id)]
            lag=actual.travel_time if actual.accessible else math.inf
            edges.append((f'D:{vehicle.object_id}:{road.source_id}',f'A:{vehicle.object_id}:{road.target_id}',lag))
    by_uav=defaultdict(list)
    for cid,flight in flights.items():
        v=flight['vehicle_id'];launch=f'L:{cid}';recover=f'R:{cid}'
        nodes.update((launch,recover));lower[launch]=flight['planned_launch_s']
        source,target=stop_node[flight['launch_stop_id']],stop_node[flight['recovery_stop_id']]
        for required in (f'A:{v}:{source}',f'A:{v}:{target}'):
            if required not in nodes:raise ValueError('sortie stop not on fixed vehicle route')
        edges.extend([(f'A:{v}:{source}',launch,0.),(launch,f'D:{v}:{source}',0.),
            (f'A:{v}:{target}',recover,0.),(recover,f'D:{v}:{target}',0.)])
        actual=native[cid]
        successful=actual['recovery_approach_success'] and not actual['constraint_violations']
        edges.append((launch,recover,actual.get('flight_duration_s',actual['duration_s']) if successful else math.inf))
        by_uav[flight['uav_id']].append(cid)
    for uav,ids in by_uav.items():
        ordered=sorted(ids,key=lambda cid:(flights[cid]['planned_launch_s'],cid))
        turnaround=snapshot.uav_state(uav).turnaround_time
        for prior,next_id in zip(ordered,ordered[1:]):edges.append((f'R:{prior}',f'L:{next_id}',turnaround))
    times=earliest_dispatch(nodes,edges,lower)
    executed=[];completed=set();energy=0.;uav_wait=0.;vehicle_wait=0.
    horizon=physical_build.time_horizon
    for cid,flight in flights.items():
        launch,recover=times[f'L:{cid}'],times[f'R:{cid}']
        actual=native[cid]
        duration=actual.get('flight_duration_s',actual['duration_s'])
        launched=math.isfinite(launch) and launch<horizon
        rendezvous=launched and math.isfinite(recover) and recover<=horizon
        wait=max(0.,recover-launch-duration) if rendezvous else None
        if launched:
            energy+=actual['electrical_energy_j']
            if not actual['constraint_violations'] and launch+duration<=horizon:
                completed.update(actual['effective_task_ids'])
        if wait is not None:uav_wait+=wait
        executed.append({'flight_id':cid,'uav_id':flight['uav_id'],'vehicle_id':flight['vehicle_id'],
            'launched':launched,'launch_s':launch if launched else None,'native_duration_s':duration,
            'rendezvous_s':recover if rendezvous else None,'rendezvous_success':rendezvous,
            'uav_rendezvous_wait_s':wait,'planned_launch_s':flight['planned_launch_s'],
            'launch_delay_s':launch-flight['planned_launch_s'] if launched else None})
    for vehicle,(sequence,legs) in routes.items():
        for node in sequence:
            a,d=times[f'A:{vehicle}:{node}'],times[f'D:{vehicle}:{node}']
            if math.isfinite(a) and math.isfinite(d):vehicle_wait+=d-a
            gantt.append({'vehicle_id':vehicle,'node_id':node,'arrival_s':a if math.isfinite(a) else None,
                          'departure_s':d if math.isfinite(d) else None})
    end_times=[times[f'D:{v}:{sequence[-1]}'] for v,(sequence,_) in routes.items()]
    done=all(math.isfinite(t) and t<=horizon for t in end_times) and len(completed)==len(manifest['denominator_task_ids'])
    return {'dispatch_policy':'fixed assignments/routes/UAV order; launches no earlier than planned; earliest resource-feasible departure',
        'physical_boundary':'3D kinematic ground travel and native UAV approach to staging point; contact docking not simulated',
        'flights':executed,'vehicle_gantt':gantt,'task_denominator':len(manifest['denominator_task_ids']),
        'safely_completed_task_ids':sorted(completed),'effective_task_coverage':len(completed)/max(1,len(manifest['denominator_task_ids'])),
        'rendezvous_success_rate':sum(f['rendezvous_success'] for f in executed)/max(1,len(executed)),
        'mission_complete':done,'mission_completion_s':max(end_times) if done else None,
        'vehicle_wait_s':vehicle_wait,'uav_wait_s':uav_wait,'native_electrical_energy_j':energy,
        'native_wait_integrated':all((f['uav_rendezvous_wait_s'] or 0.)<=native[f['flight_id']].get('rendezvous_wait_s',0.)+.011 for f in executed)}

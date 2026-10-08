"""Headless native component execution with process and report verification."""
from pathlib import Path
from experiments_3d.data_upgrade import read_json,sha256,write_once
from experiments_3d.execution import audit_execution
from experiments_3d.supervision import run_bounded


def execute(manifest: Path,output: Path,*,editor: Path,project: Path,deadline_s=180.):
    output.mkdir(parents=True,exist_ok=False)
    if not editor.is_file() or not project.is_file():raise FileNotFoundError('native editor/project missing')
    telemetry=output/'telemetry.csv'
    report=output/'automation'
    command=[str(editor.resolve()),str(project.resolve()),'/Engine/Maps/Entry','-unattended','-nop4','-nosplash','-nullrhi','-nosound',
        '-stdout','-FullStdOutLogOutput','-ExecCmds=Automation RunTests IRLiveS140.Paper10.FixedPlanExecution',
        '-TestExit=Automation Test Queue Empty',f'-Paper10ExecutionInput={manifest.resolve()}',
        f'-Paper10ExecutionOutput={telemetry.resolve()}',f'-ReportExportPath={report.resolve()}']
    attempt=run_bounded(command,output/'process',deadline_s)
    if attempt['status']!='completed':raise RuntimeError('Native execution failed; inspect preserved process logs')
    index=read_json(report/'index.json')
    if index.get('succeeded')!=1 or index.get('failed',0):raise RuntimeError('Native automation did not pass exactly one test')
    audit=audit_execution(manifest,telemetry,output/'audit.json')
    write_once(output/'identity.json',{'manifest_sha256':sha256(manifest),'automation_sha256':sha256(report/'index.json'),
        'telemetry_sha256':sha256(telemetry),'editor':str(editor.resolve()),'project':str(project.resolve()),
        'scope':'headless native controller/battery integration; geometric telemetry audit; no contact docking claim'})
    return audit


def execute_fleet(snapshot,result,physical,manifest,output,*,editor,project):
    import copy
    import dataclasses
    from Env.scene_builder import SceneBuilder
    from experiments_3d.fleet_execution import dispatch
    physical_build=SceneBuilder.from_project_data(dataclasses.replace(physical,service_candidates={'candidates':[]}))
    audit=execute(manifest,output/'approach',editor=editor,project=project)
    m=read_json(manifest)
    fleet=dispatch(snapshot,result,physical_build,m,audit)
    if not fleet['native_wait_integrated']:
        waiting={f['flight_id']:f['uav_rendezvous_wait_s'] or 0. for f in fleet['flights']}
        extended=copy.deepcopy(m)
        for f in extended['flights']:f['hover_after_flight_s']=waiting[f['flight_id']]
        path=output/'rendezvous_manifest.json'
        write_once(path,extended)
        audit=execute(path,output/'rendezvous',editor=editor,project=project)
        fleet=dispatch(snapshot,result,physical_build,m,audit)
        if not fleet['native_wait_integrated']:raise RuntimeError('native rendezvous duration did not cover dispatch wait')
    write_once(output/'fleet.json',fleet)
    return {'sortie_audit':audit,'fleet':fleet}

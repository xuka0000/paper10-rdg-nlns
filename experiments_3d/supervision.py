"""Session-owned bounded subprocess execution with durable terminal evidence."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
import tempfile
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    descriptor,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(descriptor,'w',encoding='utf-8') as stream:
            stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        # Windows readers/antivirus may hold a transient non-delete-sharing
        # handle. Retry only metadata publication, within one second; never
        # repeat a numerical job or a billable operation.
        until=time.monotonic()+1.
        while True:
            try:os.replace(temporary,path);break
            except PermissionError:
                if time.monotonic()>=until:raise
                time.sleep(.025)
    finally:
        if temporary.exists():temporary.unlink()


def stop_process_tree(process):
    """Terminate only this owned job and its descendants, including build tools."""
    import psutil
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in reversed(children):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)
    _, alive = psutil.wait_procs(children, timeout=10)
    if alive:
        raise RuntimeError(f"Owned child processes still alive: {[p.pid for p in alive]}")


def run_bounded(command: list[str], directory: Path, deadline_seconds: float, cwd: Path | None = None) -> dict:
    if deadline_seconds <= 0 or not command:
        raise ValueError("positive deadline and explicit command required")
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = {"command": command, "cwd": str(cwd or Path.cwd()), "status": "starting",
              "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "deadline_seconds": deadline_seconds,
              "observer": "parent process polling every second", "notification": "current terminal and status.json",
              "continuation": "one attempt; no automatic retry or dependent jobs after failure",
              "unattended_supervision": False}
    atomic_json(directory / "status.json", record)
    with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(command, cwd=cwd, stdout=stdout, stderr=stderr,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception as error:
            record.update(status="launch_failed", error=f"{type(error).__name__}: {error}")
            atomic_json(directory / "status.json", record)
            raise
        record.update(pid=process.pid, status="running")
        last_notice = -10.0
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                record.update(elapsed_seconds=elapsed, observed_at=dt.datetime.now(dt.timezone.utc).isoformat())
                atomic_json(directory / "status.json", record)
                if elapsed >= deadline_seconds:
                    stop_process_tree(process)
                    record["status"] = "deadline_exceeded"
                    break
                if elapsed - last_notice >= 10:
                    print(json.dumps({"status": "running", "elapsed_seconds": round(elapsed, 1), "pid": process.pid,
                                      "status_source": str(directory / "status.json")}), flush=True)
                    last_notice = elapsed
                time.sleep(min(1.0, deadline_seconds - elapsed))
        except BaseException:
            if process.poll() is None:
                stop_process_tree(process)
            record.update(status="observer_interrupted", returncode=process.returncode)
            atomic_json(directory / "status.json", record)
            raise
        if record["status"] == "running":
            record["status"] = "completed" if process.returncode == 0 else "failed"
        record.update(returncode=process.returncode, elapsed_seconds=time.monotonic() - started,
                      finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(directory / "status.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if record["status"] != "completed":
        print((directory / "stderr.log").read_text(encoding="utf-8")[-5000:], flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--deadline-seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    result = run_bounded(command, args.directory, args.deadline_seconds)
    raise SystemExit(0 if result["status"] == "completed" else 1)

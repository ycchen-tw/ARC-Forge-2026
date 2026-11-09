"""Own one generation worker; restart failed owned serving jobs, resume saved shards."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import httpx

from generate import atomic_json, exclusive


def identity(pid):
    path = Path('/proc') / str(pid)
    if not path.exists():
        return None
    try:
        fields=(path / 'stat').read_text().split()
        return None if fields[2]=='Z' else fields[21]
    except FileNotFoundError:
        return None


def main(a):
    out=a.out.resolve()
    root=Path(__file__).resolve().parents[2]
    python=os.environ['PYTHON']
    env=dict(os.environ,MODEL=a.model,TAG=a.server_tag,PORT=str(a.port))
    server_pid=a.adopt_pid
    cmdline=Path(f'/proc/{server_pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if 'sglang.launch_server' not in cmdline or '--served-model-name qwen38fn-specdata ' not in cmdline or f'--port {a.port} ' not in cmdline:
        raise RuntimeError('refusing to adopt an unrelated process')
    server_start=identity(server_pid)
    def stop_owned_server():
        if server_start is not None and identity(server_pid)==server_start:
            os.kill(server_pid,signal.SIGINT)
        for _ in range(60):
            if identity(server_pid)!=server_start:
                return
            time.sleep(1)
        raise RuntimeError('owned server did not exit; refusing to overlap a restart')

    with exclusive(out,'supervise'):
        atomic_json(out/'supervisor.json',{'pid':os.getpid(),'server_pid':server_pid,'status':'running'})
        for attempt in range(7):
            deadline=time.monotonic()+900
            while time.monotonic()<deadline:
                if identity(server_pid)!=server_start:
                    with (out/'server-launch.log').open('a') as log:
                        process=subprocess.Popen(['bash','work/qwen38fn_specdata/serve.sh'],cwd=root,
                            env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    server_pid=process.pid
                    server_start=identity(server_pid)
                try:
                    r=httpx.get(f'http://127.0.0.1:{a.port}/health',timeout=10)
                    if r.status_code==200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(10)
            else:
                raise RuntimeError('server readiness deadline exceeded')
            command=[python,'work/qwen38fn_specdata/generate.py','generate',
                '--out',str(out),'--model',a.model,'--server',f'http://127.0.0.1:{a.port}',
                '--concurrency','300','--max-new-tokens',str(a.max_new_tokens),
                '--max-length',str(a.max_length)]
            with (out/'generate.log').open('a') as log:
                worker=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
                                        start_new_session=True)
            atomic_json(out/'supervisor.json',{'pid':os.getpid(),'server_pid':server_pid,
                'worker_pid':worker.pid,'attempt':attempt,'status':'generating'})
            last_progress=time.monotonic()
            previous=None
            while worker.poll() is None:
                path=out/'progress.json'
                if path.exists():
                    current=path.stat().st_mtime_ns
                    if current!=previous:
                        previous=current
                        last_progress=time.monotonic()
                if identity(server_pid)!=server_start or time.monotonic()-last_progress>900:
                    worker.terminate()
                    try:
                        worker.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        worker.kill()
                        worker.wait()
                    stop_owned_server()
                    break
                time.sleep(15)
            if worker.returncode==0:
                atomic_json(out/'supervisor.json',{'pid':os.getpid(),'status':'complete',
                    'generation':json.loads((out/'invocation_complete.json').read_text())})
                stop_owned_server()
                return
            # Keep evidence and retry a transient failure. Configuration/data failures
            # stop after six retries instead of silently marking an incomplete run done.
            print(f'generation failed: rc={worker.returncode}; resume attempt {attempt+1}',flush=True)
            time.sleep(15)
        atomic_json(out/'supervisor.json',{'pid':os.getpid(),'status':'failed','attempts':7})
        stop_owned_server()
        raise RuntimeError('generation retry limit reached; inspect generate.log')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--adopt-pid',type=int,required=True)
    p.add_argument('--server-tag',required=True)
    p.add_argument('--port',type=int,default=30471)
    p.add_argument('--max-new-tokens',type=int,default=4096)
    p.add_argument('--max-length',type=int,default=40960)
    main(p.parse_args())

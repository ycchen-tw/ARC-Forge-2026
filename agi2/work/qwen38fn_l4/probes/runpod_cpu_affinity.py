"""Read-only CPU topology and owned-rank affinity evidence; no CUDA operations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re


def cpu_set(text):
    result = set()
    for part in text.strip().split(','):
        if '-' in part:
            a, b = map(int, part.split('-')); result.update(range(a, b + 1))
        elif part:
            result.add(int(part))
    return result


def gpu_nodes(topology):
    lines = re.sub(r'\x1b\[[0-9;]*m', '', topology).splitlines()
    header = next(line.split('\t') for line in lines if 'NUMA Affinity' in line)
    column = [cell.strip() for cell in header].index('NUMA Affinity')
    rows = {int(line[3]): line.split('\t') for line in lines if re.match(r'^GPU[0-3]\t', line)}
    if set(rows) != set(range(4)):
        raise ValueError('GPU topology needs four exact ordinal rows')
    return [int(rows[index][column].strip()) for index in range(4)]


def collect(topology):
    root = Path('/sys/devices/system/cpu')
    online = sorted(cpu_set((root / 'online').read_text()))
    cores, siblings = {}, {}
    for cpu in online:
        loc = root / f'cpu{cpu}/topology'
        cores[str(cpu)] = [int((loc / 'physical_package_id').read_text()), int((loc / 'core_id').read_text())]
        siblings[str(cpu)] = sorted(cpu_set((loc / 'thread_siblings_list').read_text()))
    nodes = {str(node): sorted(cpu_set(Path(f'/sys/devices/system/node/node{node}/cpulist').read_text()))
             for node in (0, 1)}
    maximum = Path('/sys/fs/cgroup/cpu.max').read_text().split()
    return dict(online_cpus=online, allowed_cpus=sorted(os.sched_getaffinity(0)),
        physical_core_ids=cores, thread_siblings=siblings, node_cpus=nodes,
        gpu_numa_nodes=gpu_nodes(topology), cpu_max=maximum,
        quota_cpus=None if maximum[0] == 'max' else int(maximum[0]) / int(maximum[1]))


def validate(record):
    if record['online_cpus'] != list(range(128)) or len({tuple(v) for v in record['physical_core_ids'].values()}) != 64:
        raise ValueError('CPU-affinity helper requires verified64 physical cores/128 logical CPUs')
    if any(record['thread_siblings'][str(cpu)] != [cpu % 64, cpu % 64 + 64] for cpu in range(128)):
        raise ValueError('CPU-affinity helper assumes HT sibling offset64')
    expected_nodes = {'0': sorted(set(range(32)) | set(range(64, 96))),
                      '1': sorted(set(range(32, 64)) | set(range(96, 128)))}
    if record['node_cpus'] != expected_nodes or record['gpu_numa_nodes'] != [0, 0, 1, 1]:
        raise ValueError('CPU/GPU NUMA layout differs from reviewed two-node mapping')
    rank_cpus = {str(rank): list(range(rank * 16, (rank + 1) * 16))
                 + list(range(rank * 16 + 64, (rank + 1) * 16 + 64)) for rank in range(4)}
    if any(not set(cpus) <= set(record['allowed_cpus']) for cpus in rank_cpus.values()):
        raise ValueError('cpuset/current affinity excludes recommended rank CPUs')
    return dict(record, recommended_rank_cpus=rank_cpus, status='cpu_topology_verified',
                limits=['CPU quota is shared; affinity does not increase it.',
                        'No NUMA memory-placement effect has been measured.'])


def status(pid):
    return dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)


def descendant(pid, owner):
    while pid > 1:
        if pid == owner:
            return True
        pid = int(status(pid)['PPid'])
    return False


def readback(log, owner_pid, owner_pgid, expected):
    matches = re.findall(r'Process (\d+) gpu_id (\d+) is running on CPUs: (\[[^\n]*?\])', log)
    if len(matches) != 4 or {int(gpu) for _, gpu, _ in matches} != set(range(4)):
        raise RuntimeError('missing/ambiguous four-rank affinity log receipts')
    rows = []
    for pid_text, gpu, logged in matches:
        pid = int(pid_text)
        if not descendant(pid, owner_pid) or os.getpgid(pid) != owner_pgid:
            raise RuntimeError('affinity PID is not owned by this model server')
        info = status(pid); actual = sorted(cpu_set(info['Cpus_allowed_list']))
        wanted = expected[gpu]
        if actual != wanted or json.loads(logged) != wanted:
            raise RuntimeError('actual rank CPU affinity differs from topology preflight')
        threads = {path.name: sorted(cpu_set(status_thread(path)['Cpus_allowed_list']))
                   for path in Path(f'/proc/{pid}/task').iterdir()}
        rows.append(dict(pid=pid, gpu_id=int(gpu), main_thread_cpus=actual,
            mems_allowed_list=info['Mems_allowed_list'].strip(), thread_cpus=threads,
            all_threads_match=all(cpus == wanted for cpus in threads.values())))
    return dict(status='main_thread_affinity_verified', ranks=rows,
                memory_placement='UNVERIFIED; mems_allowed_list is a cpuset, not a memory policy')


def status_thread(path):
    return dict(line.split(':', 1) for line in (path / 'status').read_text().splitlines() if ':' in line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-probe', type=Path, required=True, help='existing same-host GPU topology receipt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve existing CPU topology evidence')
    record = validate(collect(json.loads(args.gpu_probe.read_text())['topology']))
    args.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'status': record['status'], 'quota_cpus': record['quota_cpus']}))


if __name__ == '__main__':
    main()

"""Parallel upload of a new model-instance version: reuses node2 runs/qwen38fn_eval/20261010-gean320-int3-kaggle/
upload_parallel.py (overrides upload_files, the per-file loop replaced by a thread pool) and calls
model_instance_version_create at the end."""
import sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

INSTANCE, FOLDER, THREADS, NOTES = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]

class ParallelApi(KaggleApi):
    def upload_files(self, request, resources, folder, blob_type, upload_context,
                     quiet=False, dir_mode="skip", ignore_patterns=None):
        if folder != FOLDER or resources or dir_mode != "skip":
            raise ValueError("only the folder audited for this upload is supported")
        entries = list(Path(folder).iterdir())
        if any(not p.is_file() for p in entries):
            raise ValueError("the folder has entries that are not files")
        names = [p.name for p in entries]
        size = {n: Path(folder, n).stat().st_size for n in names}
        names.sort(key=lambda n: -size[n])
        total = sum(size.values())
        print(f"PLAN {len(names)} files {total/1e9:.2f} GB, {THREADS} threads", flush=True)
        started = time.monotonic()

        def up(name):
            api = KaggleApi(); api.authenticate()
            r = api._upload_file_or_folder(folder, name, blob_type, upload_context, "skip", True)
            if r is None or not getattr(r, "token", None):
                raise RuntimeError("upload not completed: " + name)
            return name, r

        done = sent = 0
        with ThreadPoolExecutor(THREADS) as pool:
            for fut in as_completed([pool.submit(up, n) for n in names]):
                name, r = fut.result()
                request.files.append(self._new_file(r))
                done += 1; sent += size[name]
                el = time.monotonic() - started
                print(f"DONE {done}/{len(names)} {name} total {sent/1e9:.1f}/{total/1e9:.1f} GB "
                      f"{el:.0f}s aggregate {sent/1e6/el:.0f} MB/s", flush=True)
        if done != len(names) or len(request.files) != len(names):
            raise RuntimeError("upload list incomplete")

api = ParallelApi(); api.authenticate()
resp = api.model_instance_version_create(INSTANCE, FOLDER, NOTES, quiet=True, dir_mode="skip")
print("RESPONSE", resp, flush=True)

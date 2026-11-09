"""Upload the packed gEAN320 INT3 deployment to a private HF model repo, then verify file list, sizes and privacy."""
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

model, repo = Path(sys.argv[1]), sys.argv[2]
manifest = json.loads((model / "deploy_manifest.json").read_text())
local = {p.name: p.stat().st_size for p in model.iterdir() if p.is_file()}
assert set(manifest["files"]) | {"deploy_manifest.json"} == set(local), set(local) ^ set(manifest["files"])
api = HfApi()
api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
info = api.model_info(repo)
assert info.private, f"{repo} is not private; refusing to upload"
api.upload_large_folder(repo_id=repo, repo_type="model", folder_path=str(model), num_workers=8,
                        ignore_patterns=[".cache/**"])
info = api.model_info(repo, files_metadata=True)
remote = {s.rfilename: s.size for s in info.siblings}
missing = {k for k in local if k not in remote}
mismatch = {k: (local[k], remote[k]) for k in local if k in remote and remote[k] != local[k]}
extra = set(remote) - set(local) - {".gitattributes"}
print(json.dumps({"repo": repo, "private": info.private, "sha": info.sha, "files": len(remote),
                  "bytes": sum(v or 0 for v in remote.values()), "missing": sorted(missing),
                  "size_mismatch": mismatch, "extra": sorted(extra)}, indent=1))
assert info.private and not missing and not mismatch and not extra
print("UPLOAD_VERIFIED")

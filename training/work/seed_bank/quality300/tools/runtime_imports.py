"""Static runtime closure for the task format's flat, bundled Python modules.

Only declared imports reachable from the supplied entrypoints are included.
Unrelated verification/rendering/authoring scripts and provenance snapshots are
not runtime dependencies. Dynamic imports and external data are not discovered.
"""
import ast
import hashlib
from pathlib import Path


def local_import_closure(package, entrypoints=('transform.py', 'generate_input.py')):
    package = Path(package)
    files = {path.stem: path for path in package.glob('*.py')}
    pending = [package / filename for filename in entrypoints]
    reached = set()
    while pending:
        source = pending.pop()
        if source.name in reached:
            continue
        reached.add(source.name)
        tree = ast.parse(source.read_bytes(), filename=str(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split('.')[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [(node.module or '').split('.')[0]]
            else:
                continue
            pending.extend(files[name] for name in names if name in files)
    return sorted(reached)


def runtime_code_sha256(package):
    package = Path(package)
    return {filename: hashlib.sha256((package / filename).read_bytes()).hexdigest()
            for filename in local_import_closure(package)}

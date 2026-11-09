"""Stateful Python executor for ARC tool-use runs.

Runs as a subprocess, one per solve attempt. Speaks newline-delimited JSON:

    parent -> child   {"init": {"train_pairs": [...], "test_input": [[...]], "extra": {...}}}
    child  -> parent  {"ready": true}
    parent -> child   {"code": "print(1+1)"}
    child  -> parent  {"output": "2\\n", "timed_out": false}
    parent -> child   EOF                         (child exits)

The protocol runs over a dedicated inherited fd (number in ARC_SANDBOX_OUT_FD),
not stdout, so a C extension writing to fd 1 cannot corrupt it. stdout goes to
/dev/null; stderr goes to a file when the parent supplies stderr_path, because a
child that dies is otherwise undiagnosable.

WHAT THIS IS NOT: a security sandbox. The B200 nodes offer this user no OS-level isolation
(no bwrap, user namespaces blocked by AppArmor, no docker socket, no sudo, and
systemd --user silently ignores its sandboxing properties). What is enforced here:

  * RLIMIT_FSIZE (0, 0)  - kernel-enforced, blocks every write to a regular file,
                           and cannot be raised back because the hard limit is 0.
  * RLIMIT_DATA          - heap ceiling.
  * SIGALRM              - per-call wall-clock timeout that leaves session state
                           intact (raises KeyboardInterrupt inside the call).
  * attribute removal    - os.remove/rmtree/system/exec*, and subprocess, socket
                           and ctypes made un-importable.
  * scratch cwd          - the child runs in a throwaway temp dir, removed on exit.
                           RLIMIT_FSIZE blocks writing content but NOT the O_CREAT
                           itself, so `np.save(...)` still leaves a 0-byte file; the
                           child inherits the launcher's cwd, so without this those
                           land in whatever directory the run was started from. A
                           47k-rollout collection put `cov0.npy` in the repo working
                           tree before this was added.

Deletion is stopped by the attribute-removal bullet only, which is a Python-level
measure that deliberately-hostile code can defeat. It is sized for non-adversarial
code that a model writes while solving a puzzle, not for untrusted input.
"""

import ast
import io
import json
import os
import resource
import signal
import sys
import tempfile
import traceback

TIMEOUT = float(os.environ.get("ARC_SANDBOX_TIMEOUT", "20"))
MAX_HEAP_MB = int(os.environ.get("ARC_SANDBOX_MAX_HEAP_MB", "8192"))
MAX_OUTPUT_CHARS = int(os.environ.get("ARC_SANDBOX_MAX_OUTPUT_CHARS", "4000"))

# os names to drop. Deleted unconditionally: if one is missing the platform is not
# what this file was written against, and that should surface at startup rather
# than leave a hole open silently.
_OS_BLOCKED = [
    "remove", "unlink", "rmdir", "removedirs", "rename", "renames", "replace",
    "truncate", "ftruncate", "system", "popen", "chmod", "chown", "link",
    "symlink", "fork", "forkpty", "kill", "killpg", "abort",
    "execv", "execve", "execl", "execle", "execlp", "execvp", "execvpe",
    "spawnv", "spawnve", "spawnl", "spawnle", "spawnlp", "spawnvp",
]
_SHUTIL_BLOCKED = [
    "rmtree", "move", "copy", "copy2", "copyfile", "copytree", "chown",
    "make_archive", "unpack_archive",
]
# shutil is deliberately absent: its dangerous names are already removed above, and
# making it un-importable only risks breaking a numpy code path for no extra cover.
_UNIMPORTABLE = ["subprocess", "socket", "ctypes", "multiprocessing"]


def harden():
    """Apply the limits. numpy is imported first: it must load before the heap
    ceiling and the module surgery, not after."""
    resource.setrlimit(resource.RLIMIT_DATA, (MAX_HEAP_MB * 1024 * 1024,) * 2)

    import numpy  # noqa: F401  (imported here so it is warm before hardening)

    import shutil

    for name in _OS_BLOCKED:
        delattr(os, name)
    for name in _SHUTIL_BLOCKED:
        delattr(shutil, name)
    for name in _UNIMPORTABLE:
        sys.modules[name] = None

    # Last, and irreversible: hard limit 0 means the process cannot raise it back.
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))


class Session:
    """Jupyter-ish: last expression is echoed, namespace persists across calls."""

    def __init__(self, timeout: float, max_output_chars: int):
        self._timeout = timeout
        self._max_output_chars = max_output_chars
        self.ns: dict = {"__builtins__": __builtins__, "__name__": "__main__"}

    def load_task(self, train_pairs, test_input, extra):
        """`test_input=None` leaves the name undefined: a joint multi-test solve
        exposes `test_inputs` via `extra` instead, and a stale singular name next
        to it would be an invitation to solve the wrong grid. `extra` is a dict of
        additional names (JSON values) and must not shadow the three built-ins."""
        import numpy as np

        self.ns["np"] = np
        self.ns["train_pairs"] = train_pairs
        if test_input is not None:
            self.ns["test_input"] = test_input
        clash = {"np", "train_pairs", "test_input"} & set(extra)
        if clash:
            raise ValueError(f"extra namespace shadows built-in sandbox names: {sorted(clash)}")
        self.ns.update(extra)

    def execute(self, code: str):
        stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        timed_out = False

        def on_alarm(signum, frame):
            raise KeyboardInterrupt(f"execution exceeded {self._timeout:g}s")

        old_handler = signal.signal(signal.SIGALRM, on_alarm)
        signal.setitimer(signal.ITIMER_REAL, self._timeout)
        try:
            sys.stdout, sys.stderr = stdout_buf, stderr_buf
            self._run(code)
        except KeyboardInterrupt as e:
            timed_out = True
            stderr_buf.write(f"[interrupted] {e}\n")
        except SystemExit as e:
            stderr_buf.write(f"SystemExit: {e.code}\n")
        except SyntaxError as e:
            # No user frame exists yet, so the generic formatter would strip this
            # down to a bare "SyntaxError:". format_exception_only keeps the
            # offending line and the caret.
            stderr_buf.write("".join(traceback.format_exception_only(type(e), e)))
        except BaseException:
            stderr_buf.write(self._user_traceback(traceback.format_exc()))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)
            sys.stdout, sys.stderr = old_stdout, old_stderr

        out, err = stdout_buf.getvalue(), stderr_buf.getvalue()
        if err:
            text = f"{out.rstrip()}\n{err}" if out.strip() else err
        elif not out.strip():
            text = "[no output — use print() to see results]"
        else:
            text = out
        return self._truncate(text), timed_out

    def _run(self, code: str):
        """Exec, echoing the value of a trailing expression."""
        tree = ast.parse(code)
        if not tree.body:
            return
        if isinstance(tree.body[-1], ast.Expr):
            if len(tree.body) > 1:
                head = ast.Module(body=tree.body[:-1], type_ignores=[])
                exec(compile(head, "<cell>", "exec"), self.ns)
            tail = ast.Expression(body=tree.body[-1].value)
            value = eval(compile(tail, "<cell>", "eval"), self.ns)
            if value is not None:
                print(repr(value))
        else:
            exec(compile(tree, "<cell>", "exec"), self.ns)

    @staticmethod
    def _user_traceback(tb: str) -> str:
        """Keep only frames from the model's own code, plus the exception line."""
        lines = tb.strip().split("\n")
        kept, i = [], 0
        while i < len(lines):
            if lines[i].strip().startswith('File "'):
                is_user = "<cell>" in lines[i]
                if is_user:
                    kept.append(lines[i])
                i += 1
                while i < len(lines):
                    nxt = lines[i]
                    if nxt.strip().startswith('File "') or (nxt.strip() and not nxt.startswith(" ")):
                        break
                    if is_user:
                        kept.append(nxt)
                    i += 1
            else:
                kept.append(lines[i])
                i += 1
        return "\n".join(kept) + "\n"

    def _truncate(self, text: str) -> str:
        limit = self._max_output_chars
        if limit <= 0 or len(text) <= limit:
            return text
        head = limit // 3
        note = f"\n\n... [truncated: {len(text)} chars exceeded the {limit} char limit] ...\n\n"
        return text[:head] + note + text[-(limit - head):]


# ============================================================================
# Parent-side client
# ============================================================================


class SandboxError(RuntimeError):
    pass


class Sandbox:
    """Owns one executor subprocess. Use as a context manager; the process is
    killed on exit even if the solve loop raised."""

    def __init__(self, train_pairs, test_input, python=None, timeout=None,
                 max_output_chars=None, max_heap_mb=None, startup_timeout=120.0,
                 stderr_path=None, extra_ns=None):
        self._train_pairs = train_pairs
        self._test_input = test_input
        # Extra names to define in the child's namespace (JSON-serialisable values),
        # e.g. `test_inputs` for a joint multi-test solve. See Session.load_task.
        self._extra_ns = dict(extra_ns) if extra_ns else {}
        self._python = python or sys.executable
        self._startup_timeout = startup_timeout
        self._env_overrides = {
            k: str(v)
            for k, v in {
                "ARC_SANDBOX_TIMEOUT": timeout,
                "ARC_SANDBOX_MAX_OUTPUT_CHARS": max_output_chars,
                "ARC_SANDBOX_MAX_HEAP_MB": max_heap_mb,
            }.items()
            if v is not None
        }
        self._stderr_path = stderr_path
        self._stderr_file = None
        self._proc = None
        self._out = None
        self._scratch = None
        self.n_restarts = 0

    def __enter__(self):
        import subprocess as _sp  # parent side only; the child cannot import this

        read_fd, write_fd = os.pipe()
        env = dict(os.environ)
        env.update(self._env_overrides)
        env["ARC_SANDBOX_OUT_FD"] = str(write_fd)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        # One BLAS thread per sandbox: dozens of these run at once, and a default
        # thread pool per process would oversubscribe the box many times over.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[var] = "1"

        # stderr goes to a file rather than /dev/null: the protocol is on its own
        # fd, so nothing here can corrupt it, and a child that dies leaves a reason.
        if self._stderr_path:
            self._stderr_file = open(self._stderr_path, "w")
            stderr_target = self._stderr_file
        else:
            stderr_target = _sp.DEVNULL
        # Run the child in a throwaway directory, never in the caller's cwd.
        #
        # RLIMIT_FSIZE (0, 0) stops the kernel from writing any *content*, but an
        # O_CREAT still succeeds, so `np.save("cov0.npy", a)` leaves a 0-byte file
        # behind. The child inherits the launcher's cwd, so for a run started from
        # the repo root that litter lands in the git working tree: the 47k-rollout
        # collection on 2026-08-30 left `cov0.npy` there, and an earlier eval left
        # `diag2.npy` in the main checkout.
        #
        # A scratch cwd is the right layer for this. Blocking more of Python's file
        # API would not help -- the creation is a kernel-level open(), and the
        # docstring above is explicit that this is a blast-radius limiter, not a
        # security boundary. Bounding *where* the blast lands is exactly its job.
        self._scratch = tempfile.mkdtemp(prefix="arc-sandbox-")
        self._proc = _sp.Popen(
            [self._python, os.path.abspath(__file__)],
            stdin=_sp.PIPE,
            stdout=_sp.DEVNULL,
            stderr=stderr_target,
            env=env,
            cwd=self._scratch,
            pass_fds=(write_fd,),
            text=True,
        )
        os.close(write_fd)
        self._out = os.fdopen(read_fd, "r")

        self._send({"init": {"train_pairs": self._train_pairs, "test_input": self._test_input,
                             "extra": self._extra_ns}})
        ready = self._recv(self._startup_timeout)
        if not ready.get("ready"):
            raise SandboxError(f"sandbox refused to start: {ready}")
        return self

    def __exit__(self, *exc):
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.kill()
            proc.wait()
        if self._out is not None:
            self._out.close()
            self._out = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None
        if self._scratch is not None:
            # Only ever holds 0-byte files the child managed to create, but leaving
            # one temp dir per rollout behind would be 47k of them over a full run.
            import shutil as _shutil
            _shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None
        return False

    def restart(self):
        """Kill the child and start a fresh one with the same task; every name the
        user defined is gone. For a caller whose `run` raised SandboxError (the
        child hung past the outer backstop -- the SIGALRM interrupt cannot stop
        code that swallows KeyboardInterrupt or sits inside a C call)."""
        self.__exit__(None, None, None)
        self.n_restarts += 1
        return self.__enter__()

    def run(self, code: str, wait: float = 300.0):
        """Returns (output_text, timed_out)."""
        self._send({"code": code})
        reply = self._recv(wait)
        return reply["output"], reply["timed_out"]

    def _send(self, obj):
        if self._proc is None:
            raise SandboxError("sandbox is not running")
        # The child process died (OOM, killed, model code crashed it) but the handle is still there: writing stdin raises
        # BrokenPipe. Turn it into SandboxError so the caller takes the same restart path as "no reply in 300 s" instead of
        # scrapping the whole task (2026-09-02 d-max-100k, db695cfb).
        try:
            self._proc.stdin.write(json.dumps(obj, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            raise SandboxError(f"sandbox process died (pipe broken: {e}; exit code {self._proc.poll()})") from e

    def _recv(self, wait: float):
        """The executor's own SIGALRM bounds each call, so a missing reply means
        the process died, not that it is slow. `wait` is the outer backstop."""
        import selectors

        sel = selectors.DefaultSelector()
        sel.register(self._out, selectors.EVENT_READ)
        try:
            if not sel.select(wait):
                raise SandboxError(f"sandbox produced no reply within {wait:g}s")
        finally:
            sel.close()
        line = self._out.readline()
        if not line:
            self._proc.poll()
            code = self._proc.wait(timeout=10)
            raise SandboxError(
                f"sandbox stopped writing (exit code {code}); stderr: {self._read_stderr()}"
            )
        return json.loads(line)

    def _read_stderr(self, limit=800):
        if self._stderr_file is not None:
            self._stderr_file.flush()
        if not self._stderr_path or not os.path.exists(self._stderr_path):
            return "(not captured)"
        with open(self._stderr_path, errors="replace") as f:
            text = f.read().strip()
        return text[-limit:] if text else "(empty)"


# ============================================================================
# Child entry point
# ============================================================================


def main():
    out_fd = int(os.environ["ARC_SANDBOX_OUT_FD"])
    out = os.fdopen(out_fd, "w")

    harden()
    session = Session(timeout=TIMEOUT, max_output_chars=MAX_OUTPUT_CHARS)

    def reply(obj):
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)

        if "init" in req:
            init = req["init"]
            session.load_task(init["train_pairs"], init["test_input"], init["extra"])
            reply({"ready": True})
            continue

        code = req.get("code")
        if code is None:
            break
        text, timed_out = session.execute(code)
        reply({"output": text, "timed_out": timed_out})


if __name__ == "__main__":
    main()

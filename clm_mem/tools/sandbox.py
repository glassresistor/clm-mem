"""Sandboxes: a bounded place to run commands and touch files per workspace.

Every session gets one sandbox rooted at ``<workdir>/workspaces/<name>/sandbox``
(local sessions root at ``<workdir>/sandbox``). Three isolation levels, chosen
by ``sandbox_mode``:

* ``off``     -- no exec/exec-shaped tools exist at all (file tools too)
* ``jail``    -- default. A *limited process*: subprocess pinned to the sandbox
                 directory with a scrubbed environment, wall-clock timeout, CPU
                 / address-space / file-size / process-count rlimits and
                 output caps. Strong enough for demos and research use; it is
                 NOT a security boundary against a hostile script (no network
                 or filesystem isolation) -- say so in the UI rather than lie.
* ``docker``  -- real container per command (``--network none``, memory/cpu/
                 volume limits) when the docker CLI is available; falls back to
                 ``jail`` with an explanatory note when it is not.

The :attr:`ExecResult.isolation` field carries which level actually ran, so the
web cards and CLI lines never claim more protection than they got.
"""
from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import resource
except ImportError:                                  # pragma: no cover - windows
    resource = None

#: env vars kept inside a sandboxed command (everything else is scrubbed, so
#: backend API keys and other host secrets never leak into a shell).
ENV_KEEP = ("PATH", "LANG", "LC_ALL", "TERM", "PYTHONUNBUFFERED",
           "PYTHONHASHSEED", "MAKEFLAGS", "CC", "CXX")

TEXT_EXTS = {".py", ".txt", ".md", ".json", ".jsonl", ".js", ".css", ".html",
            ".c", ".h", ".sh", ".yaml", ".yml", ".toml", ".csv", ".log"}


@dataclass
class ExecResult:
    ok: bool
    code: int
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    isolation: str = "process-jail"
    note: str = ""
    timed_out: bool = False

    def merged(self) -> str:
        out = self.stdout if not self.stderr else (
            self.stdout + ("\n" if self.stdout else "") + self.stderr)
        return out.strip()


@dataclass
class Sandbox:
    root: str
    mode: str = "jail"                # off | jail | docker
    timeout: float = 30.0
    output_limit: int = 8000
    disk_mb: int = 256
    mem_mb: int = 512
    image: str = "python:3.12-slim"
    _docker: Optional[bool] = field(default=None, repr=False)

    # -- basics -----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.mode != "off" and bool(self.root)

    def ensure(self) -> str:
        os.makedirs(self.root, exist_ok=True)
        return self.root

    def isolation_label(self) -> str:
        if self.mode == "off":
            return "disabled"
        if self.mode == "docker":
            return f"docker ({self.image})" if self.docker_ready() else \
                "process-jail (docker unavailable)"
        return "process-jail"

    def docker_ready(self) -> bool:
        if self._docker is None:
            self._docker = bool(shutil.which("docker"))
        return self._docker

    def info(self) -> Dict[str, object]:
        files, nbytes = self.usage()
        return {"root": os.path.relpath(self.root) if self.root else "",
                "mode": self.mode, "isolation": self.isolation_label(),
                "docker": self.docker_ready(), "files": files, "bytes": nbytes,
                "quota_mb": self.disk_mb, "timeout": self.timeout}

    # -- containment ------------------------------------------------------
    def _resolve(self, path: str) -> str:
        """Map a user/model path to a real path inside the sandbox root."""
        root = os.path.realpath(self.ensure())
        p = (path or ".").replace("\\", "/")
        if os.path.isabs(p):
            # an absolute path is only ever meaningful as a jail escape attempt;
            # accept it solely when it already points inside the root
            cand = os.path.realpath(p)
            if cand != root and os.path.commonpath([root, cand or "/"]) != root:
                raise ValueError(f"path escapes the sandbox: {path!r}")
        else:
            cand = os.path.realpath(os.path.join(root, p))
        if cand != root and os.path.commonpath([root, cand]) != root:
            raise ValueError(f"path escapes the sandbox: {path!r}")
        return cand

    # -- execution ----------------------------------------------------------
    def exec(self, command: str, timeout: Optional[float] = None,
             cwd: str = "") -> ExecResult:
        """Run one shell command inside the sandbox."""
        if not self.enabled:
            raise RuntimeError("sandbox is disabled for this session "
                               "(--sandbox-mode off)")
        start = time.time()
        limit = float(timeout) if timeout else self.timeout
        workdir = self._resolve(cwd) if cwd else self.ensure()
        os.makedirs(workdir, exist_ok=True)
        if self.mode == "docker" and self.docker_ready():
            res = self._exec_docker(command, limit, workdir, start)
            if res is not None:
                return res
        res = self._exec_jail(command, limit, workdir)
        if self.mode == "docker":
            res.note = (res.note + " " if res.note else "") \
                + "docker failed; ran in the process jail instead"
            res.isolation = "process-jail (docker failed)"
        return res

    def _exec_jail(self, command: str, limit: float,
                   workdir: str) -> ExecResult:
        t0 = time.time()
        env = {k: v for k, v in os.environ.items() if k in ENV_KEEP}
        env.update({"HOME": self.ensure(), "PWD": workdir,
                    "PYTHONUNBUFFERED": "1", "CLM_MEM_SANDBOX": "1"})
        argv = ["/bin/sh", "-c", command]

        def _limits():                              # runs in the child pre-fork
            if resource is None:
                return
            def set(r, v):
                try:
                    soft, hard = resource.getrlimit(r)
                    resource.setrlimit(r, (int(v), max(int(v), hard or 0)))
                except Exception:
                    pass
            set(resource.RLIMIT_CPU, max(2, int(limit) + 3))
            set(resource.RLIMIT_AS, self.mem_mb * 1024 * 1024)
            set(resource.RLIMIT_FSIZE, max(1_048_576,
                                           self.disk_mb * 1024 * 1024))
            if hasattr(resource, "RLIMIT_NPROC"):
                set(resource.RLIMIT_NPROC, 64)

        proc = subprocess.Popen(argv, cwd=workdir, env=env,
                                stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                preexec_fn=_limits, start_new_session=True)
        timed_out = False
        try:
            raw, _ = proc.communicate(timeout=limit + 3)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            raw, _ = proc.communicate(timeout=10)
        out = (raw or b"").decode(errors="replace")
        cap = self.output_limit
        truncated = len(out) > cap
        note = ""
        if timed_out:
            note = f"killed after {limit:.0f}s timeout"
        return ExecResult(ok=(proc.returncode == 0 and not timed_out),
                          code=-1 if proc.returncode is None else
                          int(proc.returncode),
                          stdout=out[:cap] + ("...[output capped]"
                                              if truncated else ""),
                          duration=time.time() - t0, timed_out=timed_out,
                          isolation="process-jail", note=note)

    def _exec_docker(self, command: str, limit: float, workdir: str,
                     started: float = 0.0) -> Optional[ExecResult]:
        t0 = started or time.time()
        root = self.ensure()
        argv = ["docker", "run", "--rm", "-i", "--network", "none",
                "--memory", f"{self.mem_mb}m", "--cpus", "1",
                "--workdir", "/w", "-v", f"{root}:/w",
                self.image, "/bin/sh", "-c", command]
        try:
            proc = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
        except Exception as exc:                      # docker missing/broken
            return None
        timed_out = False
        try:
            raw, _ = proc.communicate(timeout=limit + 10)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            raw, _ = proc.communicate(timeout=10)
        out = (raw or b"").decode(errors="replace")
        if "Could not find image" in out or "Cannot connect to the Docker" in out:
            return None
        cap = self.output_limit
        return ExecResult(
            ok=(proc.returncode == 0 and not timed_out),
            timed_out=timed_out,
            code=int(proc.returncode) if proc.returncode is not None else -1,
            stdout=out[:cap] + ("...[output capped]"
                                if len(out) > cap else ""),
            duration=max(0.0, time.time() - t0),
            isolation=f"docker ({self.image})",
            note="killed after timeout" if timed_out else "")

    # -- files --------------------------------------------------------------
    def write_file(self, path: str, content: str, append: bool = False) -> Dict:
        fp = self._resolve(path)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "a" if append else "w", encoding="utf-8") as f:
            f.write(content)
        return {"path": path or "(root)", "bytes": len(content.encode()),
                "mode": "append" if append else "write"}

    def read_file(self, path: str, limit: int = 20_000) -> Dict:
        fp = self._resolve(path)
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"no such file in sandbox: {path!r}")
        with open(fp, "rb") as f:
            raw = f.read(limit)
        text = raw.decode(errors="replace")
        return {"path": path, "text": text, "chars": len(text),
                "binary": os.path.splitext(fp)[1].lower() not in TEXT_EXTS
                and "\x00" in repr(raw[:64])}

    def delete(self, path: str) -> Dict:
        fp = self._resolve(path)
        if fp == os.path.realpath(self.ensure()):
            raise ValueError("refusing to delete the whole sandbox root")
        if os.path.isdir(fp):
            shutil.rmtree(fp)
        elif os.path.isfile(fp):
            os.remove(fp)
        else:
            raise FileNotFoundError(f"no such file in sandbox: {path!r}")
        return {"deleted": path}

    def list_files(self, sub: str = ".", max_entries: int = 200) -> List[Dict]:
        base = self._resolve(sub)
        out: List[Dict] = []
        if not os.path.isdir(base):
            raise NotADirectoryError(f"no such directory in sandbox: {sub!r}")
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            rel_root = os.path.relpath(dirpath, os.path.realpath(self.ensure()))
            for name in sorted(filenames):
                fp = os.path.join(dirpath, name)
                rel = os.path.normpath(os.path.join(rel_root, name)) \
                    if rel_root != "." else name
                try:
                    size = os.path.getsize(fp)
                except OSError:
                    size = 0
                out.append({"path": rel, "bytes": size,
                            "kind": "code" if os.path.splitext(name)[1]
                            in TEXT_EXTS else "file"})
                if len(out) >= max_entries:
                    return out
        return out

    def usage(self) -> tuple:
        root = os.path.realpath(self.ensure()) if self.enabled else ""
        if not root or not os.path.isdir(root):
            return 0, 0
        n = b = 0
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                try:
                    b += os.path.getsize(os.path.join(dirpath, name))
                    n += 1
                except OSError:
                    pass
        return n, b

    def reset(self) -> Dict:
        root = self.ensure()
        shutil.rmtree(root)
        os.makedirs(root, exist_ok=True)
        return {"reset": True, "root": root}

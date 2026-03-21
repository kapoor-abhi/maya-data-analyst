"""
core/sandbox.py — Secure code execution sandbox.

Runs Python code inside a Docker container with:
- Network disabled
- Read-write storage mount (FIXED: was :ro which broke all pickle saves)
- Memory and CPU limits
- Timeout enforcement (default 180s — enough for ML training)

Falls back to local subprocess if Docker unavailable.
Set SANDBOX_LOCAL=1 in .env to force local mode.
"""
import os
import re
import sys
import subprocess
import tempfile
import logging

logger = logging.getLogger(__name__)

DOCKER_TIMEOUT  = int(os.getenv("SANDBOX_TIMEOUT_SECS", "180"))   # was 60 — too short for ML
DOCKER_MEM_LIMIT = os.getenv("SANDBOX_MEM_LIMIT", "4g")
SANDBOX_IMAGE   = os.getenv("SANDBOX_IMAGE", "python-data-sandbox:latest")
SANDBOX_LOCAL   = os.getenv("SANDBOX_LOCAL", "").lower() in ("1", "true", "yes")


def _strip_code(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = text.strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    text = re.sub(r"^```(?:python)?", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r"```$", "", text, flags=re.MULTILINE).strip()
    if text.startswith("python\n"):
        text = text[7:].strip()
    return text


class DockerREPL:
    """
    Secure Python execution via Docker with read-write storage access.
    Falls back to local subprocess if Docker unavailable (dev mode).
    """

    def __init__(self, sandbox_dir: str = "sandbox", image_name: str = SANDBOX_IMAGE):
        self.sandbox_dir = os.path.abspath(sandbox_dir)
        self.image_name = image_name
        os.makedirs(self.sandbox_dir, exist_ok=True)
        self._docker_available = (not SANDBOX_LOCAL) and self._check_docker()
        if SANDBOX_LOCAL:
            logger.info("Sandbox: local mode (SANDBOX_LOCAL=1)")

    def _check_docker(self) -> bool:
        try:
            r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
            return r.returncode == 0
        except Exception:
            return False

    def run(self, code: str) -> dict:
        """Execute Python code. Returns {output, error}. Strips markdown fences automatically."""
        code = _strip_code(code)
        if not code.strip():
            return {"output": "", "error": "Empty code provided"}
        return self._run_docker(code) if self._docker_available else self._run_local(code)

    def _run_docker(self, code: str) -> dict:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         dir=self.sandbox_dir, delete=False) as f:
            f.write(code)
            script_path = f.name

        abs_sandbox   = os.path.abspath(self.sandbox_dir)
        storage_base  = os.path.abspath(os.getenv("STORAGE_BASE", "storage"))
        os.makedirs(storage_base, exist_ok=True)

        cmd = [
            "docker", "run", "--rm",
            "--network", "none",
            "--memory", DOCKER_MEM_LIMIT,
            "--cpus", "2",
            "-v", f"{abs_sandbox}:/sandbox",
            # ★ CRITICAL BUG FIX: removed :ro flag — read-only mount broke ALL df.to_pickle() calls
            "-v", f"{storage_base}:{storage_base}",
            "--workdir", "/sandbox",
            self.image_name,
            "python", os.path.basename(script_path),
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=DOCKER_TIMEOUT)
            try:
                os.unlink(script_path)
            except Exception:
                pass
            if result.returncode != 0:
                return {"output": result.stdout,
                        "error": result.stderr or f"Exit code {result.returncode}"}
            return {"output": result.stdout, "error": None}
        except subprocess.TimeoutExpired:
            try:
                os.unlink(script_path)
            except Exception:
                pass
            return {"output": "", "error": (
                f"Timeout after {DOCKER_TIMEOUT}s. "
                "Increase SANDBOX_TIMEOUT_SECS in .env for heavy ML workloads."
            )}
        except Exception as e:
            return {"output": "", "error": str(e)}

    def _run_local(self, code: str) -> dict:
        """Fallback: local subprocess (dev only — not safe for production)."""
        logger.warning("Running code locally (Docker unavailable) — NOT SAFE FOR PRODUCTION")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         dir=self.sandbox_dir, delete=False) as f:
            f.write(code)
            script_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True, text=True, timeout=DOCKER_TIMEOUT,
                cwd=os.path.dirname(self.sandbox_dir),
            )
            try:
                os.unlink(script_path)
            except Exception:
                pass
            if result.returncode != 0:
                return {"output": result.stdout, "error": result.stderr}
            return {"output": result.stdout, "error": None}
        except subprocess.TimeoutExpired:
            try:
                os.unlink(script_path)
            except Exception:
                pass
            return {"output": "", "error": f"Timeout after {DOCKER_TIMEOUT}s"}
        except Exception as e:
            return {"output": "", "error": str(e)}

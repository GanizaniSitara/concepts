from __future__ import annotations

import logging
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from tasks_mcp.config import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_PREFIX,
    default_index_dir,
    default_log_dir,
    default_tasks_root,
)

try:
    import servicemanager
    import win32event
    import win32service
    import win32serviceutil
except ImportError as exc:  # pragma: no cover - only used on service hosts
    raise SystemExit(
        "pywin32 is required for Windows service mode. "
        "Use run_http.py for standalone execution."
    ) from exc


SERVICE_NAME = "TasksMcpHttp"
DISPLAY_NAME = "Tasks MCP HTTP"
DESCRIPTION = "HTTP MCP server for the local markdown task corpus"

RUNTIME_ROOT = Path(__file__).resolve().parent
PYTHON_EXE = Path(os.environ.get("TASKS_MCP_PYTHON", sys.executable))
TASKS_ROOT = Path(os.environ.get("TASKS_ROOT", str(default_tasks_root())))
INDEX_DIR = Path(os.environ.get("TASKS_INDEX_DIR", str(default_index_dir(RUNTIME_ROOT))))
LOG_DIR = Path(os.environ.get("TASKS_MCP_LOG_DIR", str(default_log_dir(RUNTIME_ROOT))))
HOST = os.environ.get("TASKS_MCP_HOST", DEFAULT_HOST)
PORT = int(os.environ.get("TASKS_MCP_PORT", str(DEFAULT_PORT)))
DEFAULT_TASK_PREFIX = os.environ.get("TASKS_MCP_DEFAULT_PREFIX", DEFAULT_PREFIX)


def configure_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "service-wrapper.log"
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger("tasks_mcp_service")
    logger.setLevel(logging.INFO)
    return logger


LOGGER = configure_logging()


def build_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(RUNTIME_ROOT) if not existing else f"{RUNTIME_ROOT};{existing}"
    return env


def build_command() -> list[str]:
    return [
        str(PYTHON_EXE),
        "-m",
        "tasks_mcp.server",
        "--tasks-root",
        str(TASKS_ROOT),
        "--index-dir",
        str(INDEX_DIR),
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--transport",
        "streamable-http",
        "--default-prefix",
        DEFAULT_TASK_PREFIX,
    ]


class TasksMcpHttpService(win32serviceutil.ServiceFramework):
    _svc_name_ = SERVICE_NAME
    _svc_display_name_ = DISPLAY_NAME
    _svc_description_ = DESCRIPTION

    def __init__(self, args):
        super().__init__(args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        socket.setdefaulttimeout(60)
        self.is_running = True
        self.process: subprocess.Popen[str] | None = None
        self.stdout_handle = None
        self.stderr_handle = None

    def SvcStop(self):
        LOGGER.info("Stop requested")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self.is_running = False
        win32event.SetEvent(self.hWaitStop)
        self._stop_process()

    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )
        LOGGER.info("Service starting")
        self.main()

    def main(self):
        while self.is_running:
            self._start_process()
            while self.is_running and self.process and self.process.poll() is None:
                rc = win32event.WaitForSingleObject(self.hWaitStop, 1000)
                if rc == win32event.WAIT_OBJECT_0:
                    self.is_running = False
                    break

            if not self.is_running:
                break

            exit_code = self.process.poll() if self.process else None
            LOGGER.warning("Server exited unexpectedly with code %s; restarting in 5s", exit_code)
            self._stop_process()
            time.sleep(5)

        self._stop_process()
        LOGGER.info("Service stopped")

    def _start_process(self):
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        stdout_path = LOG_DIR / "server.stdout.log"
        stderr_path = LOG_DIR / "server.stderr.log"
        self.stdout_handle = open(stdout_path, "a", encoding="utf-8")
        self.stderr_handle = open(stderr_path, "a", encoding="utf-8")
        cmd = build_command()
        LOGGER.info("Starting server: %s", cmd)
        self.process = subprocess.Popen(
            cmd,
            cwd=str(RUNTIME_ROOT),
            env=build_env(),
            stdout=self.stdout_handle,
            stderr=self.stderr_handle,
        )

    def _stop_process(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.process = None
        if self.stdout_handle:
            self.stdout_handle.close()
            self.stdout_handle = None
        if self.stderr_handle:
            self.stderr_handle.close()
            self.stderr_handle = None


if __name__ == "__main__":
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(TasksMcpHttpService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(TasksMcpHttpService)

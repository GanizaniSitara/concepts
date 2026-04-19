"""Windows service wrapper for the mempalace MCP server.

Follows the LifeRunner pattern: pywin32 ``ServiceFramework`` subclass,
registered via ``win32serviceutil.HandleCommandLine``. Runs the MCP HTTP
server as a child process so the service body only has to manage lifecycle.

Install / refresh is driven by ``mempalace_install_service.py`` in this
folder. Removal by ``mempalace_uninstall_service.py``.
"""

import logging
import os
import socket
import subprocess
import sys
from pathlib import Path

import servicemanager
import win32event
import win32service
import win32serviceutil

PYTHON_EXE = r"C:\miniconda3\envs\python312\python.exe"
# All interfaces - reachable from other hosts on the LAN. The mempalace
# "privacy by architecture" promise is loopback-only by default; this
# deliberately widens that, so do not expose the box to untrusted networks.
HOST = "0.0.0.0"
PORT = 8765

# The service runs as LocalSystem, so ``%USERPROFILE%`` / ``~`` / ``Path.home()``
# all resolve to ``C:\Windows\System32\config\systemprofile`` unless we
# override. Mempalace stores its palace + entity registry + config under
# ``~/.mempalace``, so we pin the user profile to the interactive account
# that owns the data. Change this if the box's primary account ever moves.
USER_PROFILE = r"C:\Users\admin"

LOG_DIR = Path(USER_PROFILE) / ".mempalace" / "service_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_DIR / "service.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mempalace_service")


class MempalaceService(win32serviceutil.ServiceFramework):
    _svc_name_ = "MempalaceMCP"
    _svc_display_name_ = "Python Mempalace MCP server"
    _svc_description_ = f"Mempalace MCP server bound to {HOST}:{PORT} (reach via http://<host>:{PORT}/mcp)"
    _svc_startup_type_ = win32service.SERVICE_AUTO_START

    def __init__(self, args):
        super().__init__(args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        self.process = None
        self._out = None
        self._err = None
        socket.setdefaulttimeout(60)

    def SvcStop(self):
        logger.info("SvcStop received")
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=10)
            except Exception:
                logger.exception("terminate failed, killing")
                try:
                    self.process.kill()
                except Exception:
                    logger.exception("kill failed")
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )
        logger.info("starting mempalace MCP on %s:%s via %s", HOST, PORT, PYTHON_EXE)
        try:
            self._out = open(LOG_DIR / "server.out.log", "ab", buffering=0)
            self._err = open(LOG_DIR / "server.err.log", "ab", buffering=0)
            child_env = os.environ.copy()
            child_env["USERPROFILE"] = USER_PROFILE
            child_env["HOMEDRIVE"] = USER_PROFILE[:2]
            child_env["HOMEPATH"] = USER_PROFILE[2:]
            self.process = subprocess.Popen(
                [
                    PYTHON_EXE,
                    "-m",
                    "mempalace.mcp_server",
                    "--host", HOST,
                    "--port", str(PORT),
                ],
                stdout=self._out,
                stderr=self._err,
                creationflags=subprocess.CREATE_NO_WINDOW,
                env=child_env,
            )
            logger.info("child started pid=%s", self.process.pid)
            win32event.WaitForSingleObject(self.hWaitStop, win32event.INFINITE)
            logger.info("stop signalled, exiting SvcDoRun")
        except Exception:
            logger.exception("service failure")
            raise
        finally:
            for fh in (self._out, self._err):
                if fh:
                    try:
                        fh.close()
                    except Exception:
                        pass


if __name__ == "__main__":
    if len(sys.argv) == 1:
        servicemanager.Initialize()
        servicemanager.PrepareToHostSingle(MempalaceService)
        servicemanager.StartServiceCtrlDispatcher()
    else:
        win32serviceutil.HandleCommandLine(MempalaceService)

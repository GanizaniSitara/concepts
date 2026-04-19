# mempalace-ops

Ops packaging for running the [mempalace](https://github.com/GanizaniSitara/mempalace) MCP
server on Windows in two interchangeable modes:

1. **Windows service** (home machine, always-on) — pywin32
   `ServiceFramework`, bound to `127.0.0.1:8765`.
2. **Console mode** (work machine, or any box where installing services is
   restricted) — same bind, same module, just launched in a visible window
   from your login script.

No data lives here — only code. Palace contents, WAL, service logs all live
under `%USERPROFILE%\.mempalace\`.

## Two locations: source vs runtime

- **Source (this folder, under git)** — canonical code. Edit here.
- **Runtime — `C:\Tools\mempalace-ops\`** — what actually runs. The service
  is registered against this path; the startup console hook also launches
  from here.

The split exists so pulling / switching branches in the git working tree can
never yank the running binary out from under SCM. `deploy.py` copies source
→ runtime; the installer calls it automatically before registering the
service. Refreshing after a code change is therefore always the same
command — the installer itself.

## Files

- `mempalace_service.py` — the `win32serviceutil.ServiceFramework` subclass.
  This is the actual service body. It spawns
  `python -m mempalace.mcp_server --host 127.0.0.1 --port 8765` as a child
  and manages its lifecycle against SCM signals.
- `mempalace_console.py` — port-listening check, then `os.execv` into the
  same MCP server command. Idempotent: if something's already on 8765 it
  exits 0. Safe to wire into a startup script.
- `mempalace_install_service.py` — self-elevating installer / refresher.
  Calls `deploy.mirror()` first (git → `C:\Tools`), then runs
  `stop → remove → install → start → poll /health` against the runtime copy.
- `mempalace_uninstall_service.py` — self-elevating remover.
- `deploy.py` — mirrors the source tree to `C:\Tools\mempalace-ops\`.
  Idempotent, safe to run standalone.
- `mempalace-refresh.cmd` — double-click convenience around the installer.
  Not committed (`.gitignore`); one is auto-created in the runtime copy by
  `deploy.py`.

## Prereqs

- Python 3.12 with `pywin32` and `mempalace` installed. On this machine
  that's `C:\miniconda3\envs\python312\python.exe` — adjust `PYTHON_EXE` in
  each script if your env is elsewhere.

## Install (home)

```
python C:\git\concepts\mempalace-ops\mempalace_install_service.py
```

Or double-click `C:\Tools\mempalace-ops\mempalace-refresh.cmd` once the
runtime copy exists. The installer self-elevates, deploys source → runtime,
then registers and starts the service. On success,
`http://127.0.0.1:8765/health` answers 200 before the script exits — no
reboot.

## Console (work)

First deploy the runtime copy (one-off, no admin needed):

```
python C:\git\concepts\mempalace-ops\deploy.py
```

Then point your login script at the runtime copy:

```powershell
Start-Process -FilePath 'C:\miniconda3\envs\python312\python.exe' `
              -ArgumentList 'C:\Tools\mempalace-ops\mempalace_console.py'
```

The console launcher checks the port first, so it's safe to run this even
on a machine where the service is already up — it'll no-op.

## Uninstall

```
python C:\git\concepts\mempalace-ops\mempalace_uninstall_service.py
```

Self-elevates, stops and removes the service. Leaves the runtime copy under
`C:\Tools\mempalace-ops\` in place — delete that folder manually if you
want to fully tear it down.

## Pattern

Modelled after [LifeRunner](../../../Solutions/Python/LifeRunnerV2prod/)'s
`app_service.py` + `refresh.cmd`. Notable differences:

- Service body spawns the HTTP server as a `subprocess.Popen` child rather
  than `multiprocessing.Process`. Reason: `mempalace.mcp_server` parses
  `argv` at import time, so an in-process launch would require patching
  `sys.argv`. A subprocess is cleaner to stop.
- Install is a Python script with `ShellExecuteW runas` self-elevation,
  rather than a `.cmd` with `powershell Start-Process -Verb RunAs`. Pure
  Python, no shell.

Explicitly **not** using NSSM.

## Local convenience

`deploy.py` auto-creates `C:\Tools\mempalace-ops\mempalace-refresh.cmd`, a
one-line wrapper around the installer, so you have a double-click target
without anything to commit. The in-repo `.gitignore` additionally prevents a
`mempalace-*.cmd` / `.ps1` / `.bat` / `.sh` sneaking into the source tree.

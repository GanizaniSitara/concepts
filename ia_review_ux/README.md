# AI Review UX

Installs a local git `pre-push` hook that calls the bundled Python review
helper. The installer is intentionally reversible: every hook it changes is
tracked in `~/.ai-review/install-manifest.json`, and uninstall restores backups
when the hook slot is still owned by this tool.

## Install

```bash
python3 setup_ai_review.py install
```

Useful flags:

```bash
python3 setup_ai_review.py --dry-run install
python3 setup_ai_review.py install --env wsl
python3 setup_ai_review.py install --non-interactive
```

By default, `install` registers the current git repository if it is run from
inside one. Registered repos are stored in `~/.ai-review/config.json`.

## Manage Repositories

```bash
python3 setup_ai_review.py list
python3 setup_ai_review.py add ~/projects/example-repo
python3 setup_ai_review.py remove ~/projects/example-repo
python3 setup_ai_review.py reinstall-hooks
```

`--dry-run` can be used with any subcommand to preview changes.

## Uninstall

```bash
python3 setup_ai_review.py uninstall
```

Uninstall removes only hooks that still point at this tool's installed hook
target. If a repository's `pre-push` hook was replaced after install, the
installer leaves it in place and prints a warning instead of overwriting user
work.

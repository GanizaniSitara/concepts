import os
import subprocess

def run_git_cmd(repo_path, args):
    return subprocess.run(
        ["git"] + args,
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True
    ).stdout.strip()

def uncommitted_files(repo_path):
    result = run_git_cmd(repo_path, ["status", "--porcelain"])
    return [line[3:] for line in result.splitlines() if line]

def remote_diff_files(repo_path):
    subprocess.run(["git", "fetch"], cwd=repo_path,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    branch_info = run_git_cmd(repo_path, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if not branch_info:
        return []
    diff = run_git_cmd(repo_path, ["diff", "--name-only", f"{branch_info}...HEAD"])
    return diff.splitlines()

def find_git_repos(root):
    for dirpath, dirnames, filenames in os.walk(root):
        # skip Windows recycle bin and other system folders
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in {"$recycle.bin", "recycler", "system volume information"}
        ]
        if ".git" in dirnames:
            yield dirpath
            dirnames[:] = [d for d in dirnames if d != ".git"]

if __name__ == "__main__":
    drive = os.getcwd().split(":")[0] + ":\\"
    for repo in find_git_repos(drive):
        uncommitted = uncommitted_files(repo)
        remote_diff = remote_diff_files(repo)
        if uncommitted or remote_diff:
            print(f"\n{repo}")
            if uncommitted:
                print("   Uncommitted / staged changes:")
                for f in uncommitted:
                    print(f"      {f}")
            if remote_diff:
                print("   Differs from remote:")
                for f in remote_diff:
                    print(f"      {f}")

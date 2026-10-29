# -*- coding: utf-8 -*-
"""把本机 PSP 代码通过 GitHub Git Data API 推送到远端仓库。

本环境 github.com 直连不通,只有 api.github.com 可达,因此不走 git push,
改用 REST API 直接构造 tree -> commit -> 更新 refs。

用法:
    python scripts/push_to_github.py <token> [--dry-run]
"""
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OWNER = "Eas0N333/"
REPO = "PSP-"
BRANCH = "main"
API = "https://api.github.com/repos/{}{}".format(OWNER, REPO)

EXCLUDE_DIRS = {
    "data", "checkpoint", ".workbuddy", "setup", "__pycache__",
    ".git", "exps_on_abdomen", "result", "runs", ".venv", ".idea",
}
EXCLUDE_EXT = {".pth", ".pt", ".ckpt", ".zip", ".exe", ".whl", ".pyc", ".log", ".tar", ".gz"}
EXCLUDE_FILES = {"logs_chaos.txt"}


def collect_files():
    out = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel_dir = os.path.relpath(dirpath, ROOT).replace("\\", "/")
        if rel_dir == ".":
            rel_dir = ""
        parts = [p for p in rel_dir.split("/") if p]
        if any(p in EXCLUDE_DIRS for p in parts):
            continue
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn in EXCLUDE_FILES:
                continue
            if os.path.splitext(fn)[1].lower() in EXCLUDE_EXT:
                continue
            rel = (rel_dir + "/" + fn) if rel_dir else fn
            full = os.path.join(dirpath, fn)
            out.append((rel.replace("\\", "/"), full))
    return sorted(out)


def api(path, token, method="GET", payload=None):
    url = "https://api.github.com" + path if path.startswith("/") else path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "psp-push-script")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return json.loads(r.read().decode() or "null")
        except Exception as e:  # noqa: BLE001
            print("[retry {}] {} {}".format(attempt, method, e), flush=True)
            time.sleep(3)
    raise SystemExit("API failed: " + url)


def main():
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("GITHUB_TOKEN", "")
    dry = "--dry-run" in sys.argv
    if not token and not dry:
        raise SystemExit("缺少 token")

    files = collect_files()
    total = sum(os.path.getsize(f) for _, f in files)
    skipped = [(r, os.path.getsize(f)) for r, f in files if os.path.getsize(f) > 20 * 1024 * 1024]
    print("待推送 {} 个文件,合计 {:.2f} MB".format(len(files), total / 1e6), flush=True)
    for rel, full in files:
        print("  {:>9.1f} KB  {}".format(os.path.getsize(full) / 1024, rel))
    if skipped:
        print("[warn] 超过 20MB 的文件已跳过: {}".format([r for r, _ in skipped]))
    files = [f for f in files if os.path.getsize(f[1]) <= 20 * 1024 * 1024]
    if dry:
        return

    tree = []
    for rel, full in files:
        with open(full, "rb") as f:
            raw = f.read()
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            content = base64.b64encode(raw).decode()
        tree.append({"path": rel, "mode": "100644", "type": "blob", "content": content})
        print("  + {}".format(rel), flush=True)

    info = api("/repos/{}{}".format(OWNER, REPO), token)
    parent = None
    try:
        ref = api("/repos/{}{}/git/ref/heads/{}".format(OWNER, REPO, BRANCH), token)
        parent = ref["object"]["sha"]
    except SystemExit:
        pass

    print("创建 tree...", flush=True)
    tree_obj = api("/repos/{}{}/git/trees".format(OWNER, REPO), token, "POST", {"tree": tree})
    payload = {"message": "Add PSP reproduction code with environment fixes", "tree": tree_obj["sha"]}
    if parent:
        payload["parents"] = [parent]
    print("创建 commit...", flush=True)
    commit = api("/repos/{}{}/git/commits".format(OWNER, REPO), token, "POST", payload)
    sha = commit["sha"]
    if parent:
        api("/repos/{}{}/git/refs/heads/{}".format(OWNER, REPO, BRANCH), token, "PATCH", {"sha": sha})
    else:
        api("/repos/{}{}/git/refs".format(OWNER, REPO), token, "POST", {"ref": "refs/heads/" + BRANCH, "sha": sha})
    print("\n推送完成: https://github.com/{}{}\ncommit {}".format(OWNER, REPO, sha[:10]))


if __name__ == "__main__":
    main()

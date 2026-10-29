# -*- coding: utf-8 -*-
"""多线程分块下载器(支持断点续传),用于慢速国际链路。
用法: python dl_thread.py <url> <输出文件> [线程数]
"""
import os
import sys
import threading
import time
import urllib.request

CHUNK = 4 * 1024 * 1024


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def download_part(url, start, end, path, idx, progress):
    have = os.path.getsize(path) if os.path.exists(path) else 0
    need = end - start + 1
    if have >= need:
        progress[idx] = need
        return
    for attempt in range(8):
        try:
            req = urllib.request.Request(
                url, headers={"Range": f"bytes={start + have}-{end}"}
            )
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "ab") as f:
                while True:
                    buf = r.read(65536)
                    if not buf:
                        break
                    f.write(buf)
                    have += len(buf)
                    progress[idx] = have
            if have >= need:
                return
        except Exception as e:
            print(f"[w{idx}] retry {attempt}: {e}", flush=True)
            time.sleep(3)
            have = os.path.getsize(path) if os.path.exists(path) else 0
            progress[idx] = have
    if have < need:
        raise RuntimeError(f"part {idx} incomplete {have}/{need}")


def main():
    url, out = sys.argv[1], sys.argv[2]
    nthread = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    total = head_size(url)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tmpdir = out + ".parts"
    os.makedirs(tmpdir, exist_ok=True)

    # 分块边界依赖线程数:线程数变化时必须丢弃旧分块,否则会拼出损坏文件
    meta = os.path.join(tmpdir, "meta.txt")
    if os.path.exists(meta):
        saved = open(meta).read().strip()
        if saved != f"{total}:{nthread}":
            import shutil

            shutil.rmtree(tmpdir)
            os.makedirs(tmpdir)
            print("nthread/size changed -> reset parts", flush=True)
    open(meta, "w").write(f"{total}:{nthread}")
    print(f"total={total/1e6:.1f}MB threads={nthread}", flush=True)

    bounds = []
    step = (total + nthread - 1) // nthread
    for i in range(nthread):
        s = i * step
        e = min(s + step - 1, total - 1)
        if s <= e:
            bounds.append((i, s, e))

    progress = {i: 0 for i, _, _ in bounds}
    for i, s, e in bounds:
        p = os.path.join(tmpdir, f"part{i:03d}")
        if os.path.exists(p):
            progress[i] = min(os.path.getsize(p), e - s + 1)

    threads = []
    for i, s, e in bounds:
        t = threading.Thread(
            target=download_part,
            args=(url, s, e, os.path.join(tmpdir, f"part{i:03d}"), i, progress),
            daemon=True,
        )
        t.start()
        threads.append(t)

    t0 = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        done = sum(progress.values())
        el = time.time() - t0
        spd = done / el / 1024 if el > 0 else 0
        print(
            f"progress {done/1e6:.1f}/{total/1e6:.1f}MB "
            f"({done*100/total:.1f}%) {spd:.0f}KB/s",
            flush=True,
        )
    for t in threads:
        t.join()
    errs = [t for t in threads if t.is_alive()]
    if errs:
        raise SystemExit("some threads hung")

    with open(out, "wb") as fo:
        for i, _, _ in bounds:
            with open(os.path.join(tmpdir, f"part{i:03d}"), "rb") as fi:
                while True:
                    b = fi.read(1 << 20)
                    if not b:
                        break
                    fo.write(b)
    size = os.path.getsize(out)
    print(f"DONE {out} {size/1e6:.1f}MB (expect {total/1e6:.1f}MB)", flush=True)
    if size == total:
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()

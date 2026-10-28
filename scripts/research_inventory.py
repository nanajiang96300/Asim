#!/usr/bin/env python3
"""Inventory evidence without moving it. No external dependencies or remote access."""
import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOTS = ("results", "DOCS/zhongqi/projects")
RAW_FILES = ("DOCS/zhongqi/Communication_Operators_on_NPU_edit(6).pdf",)
BUILD_FILES = ("build/bin/Simulator", "build/bin/Simulator_test", "build/CMakeCache.txt")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def inventory(output):
    # Reserve a new directory: never overwrite an earlier snapshot.
    output.mkdir(parents=True, exist_ok=False)
    tracked = set(os.fsdecode(p) for p in git("ls-files", "-z").split(b"\0") if p)
    paths = set(tracked)
    # Dependencies are a separate toolchain concern; do not recursively hash extern.
    paths = {p for p in paths if not p.startswith(("extern/", "experiments/", "scripts/research_"))
             and not p.startswith("DOCS/research/")}
    # Existing research context is fixed explicitly, excluding concurrent audit outputs.
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / "DOCS/research").glob("*.md"))
    for name in RAW_ROOTS:
        base = ROOT / name
        if base.exists():
            for directory, subdirs, files in os.walk(base, followlinks=False):
                for entry in files:
                    paths.add(str((Path(directory) / entry).relative_to(ROOT)))
                for entry in list(subdirs):
                    if (Path(directory) / entry).is_symlink():
                        paths.add(str((Path(directory) / entry).relative_to(ROOT)))
                        subdirs.remove(entry)
    paths.update(p for p in (*RAW_FILES, *BUILD_FILES) if (ROOT / p).exists())
    entries, totals, archive = [], {}, []
    for name in sorted(paths):
        path = ROOT / name
        before = path.lstat()
        if stat.S_ISDIR(before.st_mode):  # gitlinks, if present
            continue
        is_link = stat.S_ISLNK(before.st_mode)
        if not (is_link or stat.S_ISREG(before.st_mode)):
            raise ValueError(f"unsupported file type: {name}")
        sha = None if is_link else digest(path)
        after = path.lstat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError(f"file changed during inventory: {name}")
        raw = name.startswith(("results/", "projects/", "DOCS/zhongqi/", "scripts/47_trae/"))
        group = "raw_or_historical" if raw else "build_provenance_unknown" if name in BUILD_FILES else "source_and_context"
        entry = {"path": name, "bytes": after.st_size, "sha256": sha,
                 "tracked": name in tracked, "kind": "symlink_not_followed" if is_link else "file",
                 "category": group, "mtime_ns_not_experiment_date": after.st_mtime_ns}
        entries.append(entry)
        bucket = totals.setdefault(group, {"files": 0, "bytes": 0})
        bucket["files"] += 1
        bucket["bytes"] += after.st_size
        if raw:
            archive.append({"source_path": name, "bytes": after.st_size, "sha256": sha,
                            "action": "retain_in_place", "proposed_destination": None,
                            "copy_status": "not_requested", "cleanup_authorized": False})
    (output / "files.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries))
    (output / "archive_map.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in archive))
    disk = shutil.disk_usage(ROOT)
    def version(command):
        try:
            r = subprocess.run(command, capture_output=True, text=True, timeout=10)
            return {"returncode": r.returncode, "first_line": (r.stdout or r.stderr).splitlines()[0]}
        except (OSError, IndexError, subprocess.TimeoutExpired) as exc:
            return {"unavailable": type(exc).__name__}
    env = {"schema_version": "research-inventory-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
           "root": str(ROOT), "hostname": platform.node(), "platform": platform.platform(),
           "cpu_count": os.cpu_count(), "available_cpu_affinity": len(os.sched_getaffinity(0)),
           "meminfo_kib": {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:"))},
           "disk_bytes": {"total": disk.total, "free": disk.free},
           "audit_base_commit": git("rev-parse", "HEAD").decode().strip(),
           "branch": git("branch", "--show-current").decode().strip(),
           "git_status": git("status", "--short", "--untracked-files=normal").decode(),
           "submodule_status": git("submodule", "status").decode(),
           "versions": {name: version(cmd) for name, cmd in {
               "python": ["python3", "--version"], "cmake": ["cmake", "--version"],
               "cxx": ["c++", "--version"], "conan": ["conan", "--version"],
               "venv_numpy": [str(ROOT / ".venv/bin/python"), "-c", "import numpy; print(numpy.__version__)"],
           }.items()},
           "scope": {"included": "tracked project files (except exclusions), top-level research context, raw roots, selected existing build files",
                     "raw_roots": list(RAW_ROOTS), "excluded": ["extern/**", ".git/**", ".venv/**", "experiments/**", "scripts/research_*", "new audit subdirectories", "other ignored build/dependency files"],
                     "limitations": ["build hashes do not establish source provenance", "mtime is not an experiment date", "no remote backup or copying performed"]},
           "totals": totals, "file_count": len(entries), "archive_entry_count": len(archive)}
    write_json(output / "environment.json", env)
    write_json(output / "manifest.json", {"schema_version": "research-inventory-v1", "audit_base_commit": env["audit_base_commit"],
               "files": [{"path": name, "sha256": digest(output / name), "bytes": (output / name).stat().st_size}
                         for name in ("files.jsonl", "archive_map.jsonl", "environment.json")]})
    print(json.dumps({"status": "INVENTORIED", "files": len(entries), "totals": totals, "output": str(output)}, ensure_ascii=False))


def check(directory, raw_only=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    errors = []
    for item in manifest["files"]:
        path = directory / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            errors.append("manifest mismatch: " + item["path"])
    if errors:
        raise ValueError("; ".join(errors))
    for line in (directory / "files.jsonl").read_text().splitlines():
        item = json.loads(line)
        if raw_only and item["category"] != "raw_or_historical":
            continue
        path = ROOT / item["path"]
        if item["kind"] != "file":
            errors.append("unsupported symlink check: " + item["path"])
        elif not path.is_file() or path.is_symlink() or path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            errors.append("source missing or changed: " + item["path"])
    print(json.dumps({"status": "MISMATCH" if errors else "INTEGRITY_OK", "errors": errors}, ensure_ascii=False))
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "check"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--raw-only", action="store_true", help="verify original raw/historical files after context docs change")
    args = parser.parse_args()
    try:
        if args.action == "create":
            inventory(args.directory.resolve())
            return 0
        return check(args.directory.resolve(), args.raw_only)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "INVALID", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

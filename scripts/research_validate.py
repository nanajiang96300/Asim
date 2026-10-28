#!/usr/bin/env python3
"""Check research evidence consistency. This does not validate operator numerics."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
BASE = "7cf9c5088ccc030ee3bf478468660b3cdf46cee3"


def load(path):
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite JSON: " + value)))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    errors, counts = [], {}
    cache = {}

    def require(condition, message):
        if not condition:
            errors.append(message)

    def source(path):
        if path not in cache:
            # Historical citations/hashes refer to the frozen source, not revised README.
            r = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASE}:{path}"], capture_output=True)
            cache[path] = r.stdout if r.returncode == 0 else (ROOT / path).read_bytes()
        return cache[path]

    def location(item):
        path, line = item["path"], item.get("line")
        try:
            data = source(path)
            if line is not None:
                require(isinstance(line, int) and 1 <= line <= len(data.splitlines()), f"invalid line: {path}:{line}")
        except OSError:
            errors.append("missing citation: " + path)

    try:
        json_files = list((ROOT / "experiments").rglob("*.json")) + list((ROOT / "DOCS/research").rglob("*.json"))
        for path in json_files:
            load(path)
        for path in (ROOT / "experiments").rglob("*.jsonl"):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                except ValueError:
                    errors.append(f"invalid JSONL: {path.relative_to(ROOT)}:{number}")
        entries = [json.loads(line) for line in (ROOT / "experiments/history/ledger.jsonl").read_text().splitlines()]
        ids = {e["id"] for e in entries}
        require(len(ids) == len(entries), "duplicate history id")
        for entry in entries:
            require(entry["trusted_metrics"] is None and entry["trusted_score"] is None, "unverified historical score: " + entry["id"])
            require(entry["promotion_status"] == "legacy_unverified", "unexpected historical promotion: " + entry["id"])
            require(not set(entry["evidence_levels"]) & {"validated", "reproduced"}, "historical evidence overclaim: " + entry["id"])
            require(entry["baseline_id"] is None or entry["baseline_id"] in ids, "unknown historical parent: " + entry["id"])
            for item in entry["sources"]:
                location(item)
            for field in ("implementation_commit", "candidate_commit", "audit_base_commit"):
                commit = entry[field]
                if commit:
                    r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", commit + "^{commit}"], capture_output=True)
                    require(r.returncode == 0, "missing commit: " + str(commit))
        counts["history_entries"] = len(entries)
        counts["ns_entries"] = sum(e["algorithm"] == "newton_schulz" for e in entries)

        tasks = load(ROOT / "experiments/tasks.json")["tasks"]
        task_ids = {t["id"] for t in tasks}
        require(len(task_ids) == len(tasks), "duplicate task id")
        visited, active = set(), set()
        def visit(task_id):
            if task_id in active:
                raise ValueError("task dependency cycle: " + task_id)
            if task_id in visited:
                return
            active.add(task_id)
            task = next(t for t in tasks if t["id"] == task_id)
            for dependency in task["depends_on"]:
                require(dependency in task_ids, "missing task dependency: " + dependency)
                if dependency in task_ids:
                    visit(dependency)
            for evidence in task["evidence"]:
                require((ROOT / evidence).exists(), "missing task evidence: " + evidence)
            if task["status"] == "done":
                require(bool(task["evidence"]), "done without evidence: " + task_id)
            active.remove(task_id)
            visited.add(task_id)
        for task_id in task_ids:
            visit(task_id)

        issues = load(ROOT / "DOCS/research/audits/issues.json")["issues"]
        require(len({i["id"] for i in issues}) == len(issues), "duplicate issue id")
        for issue in issues:
            for item in issue["locations"]:
                location(item)
            require(set(issue["followup_tasks"]) <= task_ids, "unknown issue task: " + issue["id"])
            require(issue["operator_validation_status"] == "NOT_PERFORMED", "issue overclaims validation")
        counts["issues"] = len(issues)
        counts["tasks"] = len(tasks)

        runs = load(ROOT / "experiments/history/legacy_runs.json")["runs"]
        artifact_count = 0
        for run in runs:
            for item in run["artifacts"]:
                data = (ROOT / item["path"]).read_bytes()
                require(len(data) == item["bytes"] and sha(data) == item["sha256"], "raw artifact changed: " + item["path"])
                artifact_count += 1
        counts["legacy_runs"] = len(runs)
        counts["legacy_artifacts"] = artifact_count
        for item in load(ROOT / "experiments/history/sources.json")["sources"]:
            if item.get("status") == "present":
                data = source(item["path"])
                require(len(data) == item["bytes"] and sha(data) == item["sha256"], "historical source snapshot mismatch: " + item["path"])

        diagnostic = load(ROOT / "experiments/audit/2026-10-28/diagnostics.json")
        require(diagnostic["operator_validation_status"] == "NOT_PERFORMED" and diagnostic["candidate_score"] is None, "diagnostic overclaim")
        for item in diagnostic["sources"]:
            require(sha((ROOT / item["path"]).read_bytes()) == item["sha256"], "diagnostic source changed: " + item["path"])
        counts["diagnostics"] = len(diagnostic["checks"])
        for name in ("campaign", "run_manifest"):
            template = load(ROOT / "experiments/templates" / (name + ".json"))
            require(template["status"] == "PROPOSED" and template["promotion_allowed"] is False, "unsafe template: " + name)

        # Check only current research/new evidence docs; historical reports remain untouched.
        documents = list((ROOT / "DOCS/research").rglob("*.md")) + list((ROOT / "experiments").rglob("*.md"))
        for path in documents:
            content = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
            for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", content):
                target = unquote(target.strip("<>"))
                if re.match(r"[a-zA-Z]+:", target) or target.startswith("#"):
                    continue
                target = target.split("#", 1)[0]
                target = re.sub(r":\d+$", "", target)
                require((path.parent / target).exists(), f"broken link: {path.relative_to(ROOT)} -> {target}")
        counts["markdown_documents"] = len(documents)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        errors.append(str(exc))

    report = {"schema_version": "research-metadata-check-v1", "audit_base_commit": BASE,
              "status": "METADATA_OK" if not errors else "INVALID", "counts": counts, "errors": errors,
              "operator_validation_status": "NOT_PERFORMED",
              "limitations": ["checks metadata, original artifact hashes and citation bounds; does not certify mathematical claims or implement numerical gates"]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())

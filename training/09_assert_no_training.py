"""Refuse to start when a training process is already running.

    uv run python scripts/09_assert_no_training.py                 # default pattern
    uv run python scripts/09_assert_no_training.py --pattern 3X_train

Exit 0 means nothing matched and it is safe to launch. Exit 1 means something matched, or
**the check could not be carried out** -- see below for why those share an exit code.

This exists because the guard it replaces never ran. `.scratch/run_ablation.sh` opened with

    if pgrep -f "3[13]_train" >/dev/null 2>&1; then ... exit 1; fi

and `pgrep` is not installed in this machine's Git Bash. A missing command exits 127, the
`if` reads that as false, and the guard waves every launch through. The project notes recorded
that the script "refuses to start if a training process is running" and cited two occasions
where concurrent training destroyed a result, so the belief that the project was protected
here was load-bearing and wrong for as long as the line existed.

The lesson is in the failure mode, not the tool: **a guard that cannot tell must fail
closed.** Every path below either answers the question or exits non-zero saying it could
not. Silence is never treated as "nothing is running".

No new dependency: the process table is read through PowerShell's CIM provider, which also
yields full command lines, so the pattern can match the *script* being run rather than just
"some python". `ps -W` sees the processes but not their arguments, and psutil is only
present transitively, so neither is relied on.
"""

# Role: Guard to run before a training launch. It lists the running Python processes and exits 1
#   if a command line matches DEFAULT_PATTERN or if the list cannot be read; it exits 0 only when
#   the list was read and nothing matched.
# Used by: run by hand before a launch; _psutil_processes() is also called by
#   47_matched_continuation.assert_idle(), which the training entry points 47, 101, 103, 108 and
#   124 in this folder call at start-up; training (not used by the delivered run).
# In this repository the script lives in training/, not in scripts/ as the usage lines say.
# The process list comes from PowerShell's CIM provider (Windows) or, failing that, from psutil.
# Optional input, not included in the repository: work/cpu-coexistence.json, which describes one
#   reviewed CPU-only job that may run alongside training; that job's source files must still
#   have the SHA256 values recorded there. Without the file no exception is made.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# The training entry points. 63_predict_fused is included because stage 4 launches it as its
# fourth step, and the pattern it inherited from run_ablation.sh ("3[13]_train") missed it.
# The pattern also names development scripts that are not in this repository. The entry points
# here are guarded more strictly by 47_matched_continuation.assert_idle(), which refuses to start
# next to any other Python process, whether or not it matches this pattern.
DEFAULT_PATTERN = (
    r"(3[013]_train|6[03]_predict|44_self_training|47_matched_continuation"
    r"|100_a_delivery|101_matched_replay|102_b_quantize|103_b_final_student|106_weight_trials"
    r"|108_finalize_calibrated|109_a1_for_t2|110_ensemble_teacher_student"
    r"|112_t3_matched_bncal|113_h_fourfold|114_h_delivery|118_shared_view_pilot"
    r"|120_c_complete_folds|122_c_quant_fourfold)"
)

# PowerShell command: process ID and full command line of every process whose name starts with
# "python", as compact UTF-8 JSON. Any error ends the command with a non-zero exit code, which
# _powershell_processes() treats as a failure.
QUERY = (
    "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new(); "
    "$ErrorActionPreference='Stop'; "
    "Get-CimInstance Win32_Process -Filter \"Name like 'python%'\" | "
    "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
)


def _powershell_processes() -> list[dict]:
    """Every running python process as ``{ProcessId, CommandLine}``.

    Raises on anything unexpected rather than returning an empty list: an empty list is the
    answer "nothing is running", and this function must never invent it.
    """
    executable = shutil.which("powershell") or shutil.which("pwsh")
    if executable is None:
        raise RuntimeError("neither powershell nor pwsh is on PATH")

    finished = subprocess.run(  # noqa: S603
        [executable, "-NoProfile", "-NonInteractive", "-Command", QUERY],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
    )
    if finished.returncode != 0:
        raise RuntimeError(f"process query failed ({finished.returncode}): {finished.stderr}")

    payload = finished.stdout.strip()
    if not payload:
        # No python at all: ConvertTo-Json on an empty result prints nothing. This is the
        # one case where "no output" genuinely means "nothing running", and it is only safe
        # to say so because the command itself succeeded.
        return []

    parsed = json.loads(payload)
    # ConvertTo-Json writes a single object instead of a list when exactly one process matches.
    return [parsed] if isinstance(parsed, dict) else parsed


def _cpu_coexistence_ids(rows, contract=None):
    """Allow only the reviewed, explicitly CPU-limited sibling job; unknowns still block."""
    # rows come from _psutil_processes(). Returns the process IDs that may keep running; without
    # an unexpired coexistence file (work/cpu-coexistence.json) the set is empty.
    if contract is None:
        contract_file = Path(__file__).resolve().parents[1] / "work/cpu-coexistence.json"
        if not contract_file.exists():
            return set()
        contract = json.loads(contract_file.read_text(encoding="utf-8"))
    if time.time() >= contract["expires_at"]:
        return set()
    # A candidate has the recorded argument list, except that the argument at video_index may be
    # any .mp4 file under video_root; it must also run in the recorded working directory, with the
    # recorded launcher or interpreter as argv[0].
    matches = []
    for row in rows:
        args = row["Arguments"]
        if len(args) != len(contract["arguments"]) + 1:
            continue
        actual = args[1:].copy()
        video = Path(actual[contract["video_index"]])
        try:
            video.relative_to(Path(contract["video_root"]))
        except ValueError:
            continue
        if video.suffix.lower() != ".mp4":
            continue
        actual[contract["video_index"]] = "<video>"
        if actual != contract["arguments"] or row["WorkingDirectory"] != contract["cwd"]:
            continue
        if args[0] not in (contract["launcher"], contract["interpreter"]):
            continue
        matches.append(row)
    if not matches:
        return set()
    # A single known venv launcher + interpreter pair, with exactly matching arguments.
    if len(matches) != 2:
        return set()
    launcher = next((r for r in matches if r["Arguments"][0] == contract["launcher"]), None)
    child = next((r for r in matches if r["Arguments"][0] == contract["interpreter"]), None)
    if launcher is None or child is None or child["ParentProcessId"] != launcher["ProcessId"]:
        return set()
    if child["Arguments"][1:] != launcher["Arguments"][1:]:
        return set()
    # No exception is granted unless an actual matching parent/child pair exists.
    # An inactive unrelated project's edits must not block an exclusive run; unknown
    # Python processes remain in the caller's list and are still refused there.
    for file, digest in contract["source_sha256"].items():
        with Path(file).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                raise RuntimeError("reviewed CPU sibling implementation changed")
    # The job's resident memory and the machine's free memory must stay within the recorded
    # limits. Each failed check raises, and the caller then refuses to start.
    if sum(r["ResidentBytes"] for r in matches) > contract["max_resident_bytes"]:
        raise RuntimeError("reviewed CPU sibling exceeded its resident RAM allowance")
    import psutil

    if psutil.virtual_memory().available < contract["min_available_bytes"]:
        raise RuntimeError("insufficient RAM headroom for coexistence")
    print("reviewed CPU-only sibling allowed: " + str([r["ProcessId"] for r in matches]))
    return {r["ProcessId"] for r in matches}


def _psutil_processes() -> list[dict]:
    """Alternative OS process API when CIM is unavailable; never ignore unreadable Python.

    psutil is already installed here. Its absence or access failure is a hard failure,
    not permission to launch. Include creation time/parent for identity audits.
    """
    import psutil

    # Same selection as the CIM query: processes named python*. A process that exits during the
    # scan is skipped; any other error, such as psutil.AccessDenied, propagates to the caller,
    # which then refuses to start.
    found = []
    for process in psutil.process_iter():
        try:
            name = process.name().lower()
            if not name.startswith("python"):
                continue
            command = process.cmdline()
            if not command:
                raise RuntimeError(f"unreadable Python command line: pid {process.pid}")
            found.append(
                {
                    "ProcessId": process.pid,
                    "ParentProcessId": process.ppid(),
                    "CreationDate": process.create_time(),
                    "CommandLine": " ".join(command),
                    "Arguments": command,
                    "WorkingDirectory": process.cwd(),
                    "ResidentBytes": process.memory_info().rss,
                }
            )
        except psutil.NoSuchProcess:
            continue
    # Drop the reviewed CPU-only job if the coexistence file allows it.
    allowed = _cpu_coexistence_ids(found)
    return [row for row in found if row["ProcessId"] not in allowed]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default=DEFAULT_PATTERN, help=f"default: {DEFAULT_PATTERN}")
    args = ap.parse_args()

    # CIM first, psutil if PowerShell is missing or fails. Only the psutil path applies the
    # coexistence exception, because the CIM rows carry no arguments or working directory. If both
    # fail, the answer is unknown and the guard refuses (exit 1).
    try:
        try:
            processes = _powershell_processes()
        except Exception as cim_failure:  # noqa: BLE001
            print(f"CIM unavailable; checking via psutil: {cim_failure}", file=sys.stderr)
            processes = _psutil_processes()
    except Exception as failure:  # noqa: BLE001
        print(
            f"refusing to start: cannot determine whether training is running -- {failure}\n"
            "  a guard that cannot tell must fail closed. Check by hand with\n"
            '  powershell -Command "Get-Process python* | Select Id,StartTime,CPU"',
            file=sys.stderr,
        )
        return 1

    pattern = re.compile(args.pattern)
    # The guard's own command line can contain --pattern. Only exclude its own PID;
    # arbitrary ancestors/children must still be checked.
    matches = [
        p
        for p in processes
        if p["ProcessId"] != os.getpid() and pattern.search(p.get("CommandLine") or "")
    ]

    if matches:
        print("refusing to start: a training process is already running", file=sys.stderr)
        for process in matches:
            print(f"  pid {process['ProcessId']}  {process['CommandLine']}", file=sys.stderr)
        print(
            "\n  kill it by pid and verify, do not assume:\n"
            '    powershell -Command "Stop-Process -Id <pid> -Force"\n'
            '    powershell -Command "Get-Process python* | Select Id,CPU"\n'
            "    nvidia-smi   # VRAM must come back down",
            file=sys.stderr,
        )
        return 1

    print(f"no training process matching /{args.pattern}/ ({len(processes)} python running)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

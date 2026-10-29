#!/usr/bin/env python3
"""
Dev supervisor for test_peripherals.c on DebugTetheredEE290Config.

Usage (from the chipyard repo root):

    # One-time prereqs:
    (cd sims/vcs && make CONFIG=DebugTetheredEE290Config LOADMEM=1 BINARY=base_schema)
    (cd tests && cmake -S . -B build && cmake --build build)

    # Run the supervisor — exits 0 once all six expected stores are seen
    # (typically ~75 s of wall-clock on this design):
    python3 tests/run_test_peripherals_ci.py

Launches `make run-binary`, tails the generated .out file for [PERIPH]
lines from WithPrintfTLMonitor, and exits 0 as soon as all six expected
(address, data) stores have been observed. Exits nonzero on timeout or
process failure. Kills the whole simv pipeline (simv | spike-dasm | tee)
via process group so no orphan survives.

WARNING: cleanup uses `pkill -f simv-chipyard.harness-DebugTetheredEE290Config`
as a safety sweep for simv subshells that escape the process group via
bash process substitution. Do NOT run this concurrently with any other
sim of the same HW_CONFIG owned by the same user (another shell, another
dev tool) — the sweep will kill those too. Intended for manual dev use,
not parallel CI.
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO     = Path(__file__).resolve().parent.parent
SIMS_DIR = REPO / "sims" / "vcs"
OUT_FILE = SIMS_DIR / "output" / "chipyard.harness.TestHarness.DebugTetheredEE290Config" / "test_peripherals.out"

ATLAS  = 0x40000000
SATURN = 0x880000000

EXPECTED = {
    (ATLAS,  1111), (ATLAS,  3333), (ATLAS,  5555),
    (SATURN, 2222), (SATURN, 4444), (SATURN, 6666),
}

PERIPH_RE = re.compile(
    r"\[PERIPH\].*addr=0x([0-9a-fA-F]+).*data=0x([0-9a-fA-F]+).*mask=0x([0-9a-fA-F]+)"
)


def parse_line(line: str):
    m = PERIPH_RE.search(line)
    if not m:
        return None
    addr = int(m.group(1), 16)
    data = int(m.group(2), 16)
    data_lo = data & 0xFFFFFFFF
    return (addr, data_lo)


def launch_sim(binary: Path, timeout_cycles: int) -> subprocess.Popen:
    cmd = [
        "make",
        f"CONFIG=DebugTetheredEE290Config",
        f"LOADMEM=1",
        f"TIMEOUT_CYCLES={timeout_cycles}",
        f"BINARY={binary}",
        "run-binary",
    ]
    print(f"[ci] launching: {' '.join(cmd)}", flush=True)
    return subprocess.Popen(
        cmd,
        cwd=SIMS_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


SIMV_NAME = "simv-chipyard.harness-DebugTetheredEE290Config"


def kill_group(proc: subprocess.Popen):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            pass
        # bash process substitution (2> >(spike-dasm)) spawns a subshell that
        # may land in its own pgid, escaping killpg. Sweep by simv name too.
        subprocess.run(["pkill", "-" + sig.name.removeprefix("SIG"), "-f", SIMV_NAME],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(30):
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        if subprocess.run(["pgrep", "-f", SIMV_NAME],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
            return


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, default=REPO / "tests" / "build" / "test_peripherals.riscv")
    ap.add_argument("--wall-timeout", type=int, default=600, help="wall-clock seconds before giving up")
    ap.add_argument("--timeout-cycles", type=int, default=10_000_000, help="+max-cycles passed to simv")
    args = ap.parse_args()

    args.binary = args.binary.resolve()
    if not args.binary.exists():
        print(f"[ci] FAIL: binary not found: {args.binary}", file=sys.stderr)
        return 2

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        OUT_FILE.unlink()

    proc = launch_sim(args.binary, args.timeout_cycles)

    start = time.monotonic()
    seen: set = set()
    fh = None
    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed > args.wall_timeout:
                print(f"[ci] FAIL: wall-clock timeout after {elapsed:.0f}s", file=sys.stderr)
                print(f"[ci]   seen:    {sorted(seen)}", file=sys.stderr)
                print(f"[ci]   missing: {sorted(EXPECTED - seen)}", file=sys.stderr)
                return 1

            if proc.poll() is not None and not OUT_FILE.exists():
                print(f"[ci] FAIL: sim process exited ({proc.returncode}) with no .out file", file=sys.stderr)
                return 1

            if fh is None:
                if OUT_FILE.exists():
                    fh = OUT_FILE.open("r")
                else:
                    time.sleep(0.5)
                    continue

            line = fh.readline()
            if not line:
                time.sleep(0.5)
                continue

            hit = parse_line(line)
            if hit is None or hit not in EXPECTED or hit in seen:
                continue
            seen.add(hit)
            print(f"[ci] observed {len(seen)}/6: addr=0x{hit[0]:x} data={hit[1]}", flush=True)
            if seen == EXPECTED:
                print(f"[ci] PASS: all 6 expected stores observed in {elapsed:.1f}s", flush=True)
                return 0
    finally:
        if fh is not None:
            fh.close()
        kill_group(proc)


if __name__ == "__main__":
    sys.exit(main())

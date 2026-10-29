#!/usr/bin/env bash
set -u
set -o pipefail

# Usage:
#   ./run_saturn_atlas_tests.sh test1 test2 test3
#   ./run_saturn_atlas_tests.sh --all
#   ./run_saturn_atlas_tests.sh --parallel test1 test2 test3   # parallelize VCS sims
#   ./run_saturn_atlas_tests.sh -jN  test1 test2 test3         # cap concurrency at N
#
# Warm-up: if the VCS simv isn't built yet (fresh clone, branch switch,
# stale generated-src), run ONE test serially first to build simv, then
# -jN the rest. Parallel sweeps race on simv regen and corrupt each other.
#
# Behavior:
#   - Tests are hand-written C drivers at tests/saturn_atlas_*.c (not .S + gen_*.py).
#   - CMake configure + build runs serially once; any required golden headers
#     (smolvla_compose_*, tri_phase_*) are auto-generated via CMake's
#     add_custom_command dependencies.
#   - Simulations run either serially (default) or in parallel.
#   - A test is marked failed if:
#       * any build step exits nonzero, or
#       * the simulator log contains explicit failure signatures, or
#       * the log does not contain a PASS marker.
#
# Assumes this script lives at: chipyard/tests/run_saturn_atlas_tests.sh

RUN_ALL=0
PARALLEL=0
MAX_JOBS=0   # 0 = unbounded (one job per test)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      RUN_ALL=1
      shift
      ;;
    --parallel)
      PARALLEL=1
      shift
      ;;
    -j*)
      MAX_JOBS="${1#-j}"
      PARALLEL=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHIPYARD_ROOT="$(cd "$TESTS_DIR/.." && pwd)"
BUILD_DIR="$TESTS_DIR/build"
VCS_DIR="$CHIPYARD_ROOT/sims/vcs"
CONFIG="EE290SimConfig"

if [[ ! -d "$VCS_DIR" ]]; then
  echo "Error: could not find Chipyard VCS directory at:"
  echo "  $VCS_DIR"
  exit 1
fi

TEST_NAMES=()

if [[ $RUN_ALL -eq 1 ]]; then
  while IFS= read -r file; do
    name="$(basename "$file" .c)"
    TEST_NAMES+=("$name")
  done < <(find "$TESTS_DIR" -maxdepth 1 -name "saturn_atlas_*.c" | sort)
else
  if [[ $# -eq 0 ]]; then
    echo "Error: provide test names or use --all"
    exit 1
  fi
  TEST_NAMES=("$@")
fi

if [[ ${#TEST_NAMES[@]} -eq 0 ]]; then
  echo "No tests found."
  exit 1
fi

echo "Tests dir:     $TESTS_DIR"
echo "Chipyard root: $CHIPYARD_ROOT"
echo "Config:        $CONFIG"
echo "Tests:         ${TEST_NAMES[*]}"
if [[ $PARALLEL -eq 1 ]]; then
  if [[ $MAX_JOBS -gt 0 ]]; then
    echo "Sim mode: parallel (max $MAX_JOBS concurrent)"
  else
    echo "Sim mode: parallel (all concurrent)"
  fi
else
  echo "Sim mode: serial"
fi
echo

# Holds the test-name / build-status pair.
# BUILD_STATUS[i]=0 means ok-to-sim, 1 means build failed (sim skipped).
declare -a BUILD_STATUS=()

# ── Phase 1: CMake configure + build ──────────────────────────────────
# Build only the targets the user asked for (default --target all would
# build every CMake target in tests/, which is wasted work). With --all
# we still restrict to the saturn_atlas_* targets we enumerated above.
# Parallelism follows -jN when given, otherwise nproc.
echo "============================================================"
echo "CMake configure + build"
echo "============================================================"
cmake -S "$TESTS_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Debug || exit 1

if [[ ${MAX_JOBS:-0} -gt 0 ]]; then
  BUILD_PAR="$MAX_JOBS"
else
  BUILD_PAR="$(nproc)"
fi

build_targets=()
for tn in "${TEST_NAMES[@]}"; do
  build_targets+=("$tn")
done
if [[ ${#build_targets[@]} -gt 0 ]]; then
  cmake --build "$BUILD_DIR" --target "${build_targets[@]}" -j"$BUILD_PAR" || exit 1
fi

# Verify binaries exist.
for tn in "${TEST_NAMES[@]}"; do
  bin="$BUILD_DIR/${tn}.riscv"
  if [[ -f "$bin" ]]; then
    BUILD_STATUS+=("0")
  else
    echo "  Missing binary after build: $bin"
    BUILD_STATUS+=("1")
  fi
done

# ── Phase 2: simulations ──────────────────────────────────────────────
SIM_LOG_DIR="$(mktemp -d)"
declare -a SIM_LOG=()
declare -a SIM_PID=()
declare -a SIM_EXIT=()

FAIL_RE='FAIL:|Assertion failed: \*\*\* FAILED \*\*\*|Fatal:|FAILED \(\*+\)|\*\*\* FAILED \*\*\*|ASSERT FAIL|Assertion .* failed|\bMISMATCH\b|\b[Mm]ismatch\b'
# A test passes only if VCS emits *** PASSED *** OR the driver emits its
# final "PASS: <test_name>" line. Earlier sub-check PASS lines are not
# enough on their own.
VCS_PASS_RE='\*\*\* PASSED \*\*\*'

sim_one_test() {
  local test_name="$1"
  local bin="$BUILD_DIR/${test_name}.riscv"
  local log="$2"

  (
    cd "$VCS_DIR" &&
    make run-binary GEN_COVERAGE=1 \
      BINARY="$bin" \
      CONFIG="$CONFIG" \
      LOADMEM=1
  ) > "$log" 2>&1
}

# Throttle helper: block until outstanding PIDs drop below MAX_JOBS.
await_slot() {
  local -n _pids=$1
  local cap=$2
  while [[ ${#_pids[@]} -ge $cap ]]; do
    local pid=${_pids[0]}
    wait "$pid"
    EXIT_OF_PID[$pid]=$?
    _pids=("${_pids[@]:1}")
  done
}

declare -A EXIT_OF_PID

running_pids=()
for i in "${!TEST_NAMES[@]}"; do
  tn="${TEST_NAMES[$i]}"
  if [[ "${BUILD_STATUS[$i]}" -ne 0 ]]; then
    SIM_LOG+=("")
    SIM_PID+=("0")
    SIM_EXIT+=("1")
    continue
  fi

  log="$SIM_LOG_DIR/${tn}.simlog"
  SIM_LOG+=("$log")

  if [[ $PARALLEL -eq 1 ]]; then
    if [[ $MAX_JOBS -gt 0 ]]; then
      await_slot running_pids "$MAX_JOBS"
    fi
    echo "  [launch] $tn → $log"
    sim_one_test "$tn" "$log" &
    pid=$!
    SIM_PID+=("$pid")
    SIM_EXIT+=("0")   # overwritten after wait
    running_pids+=("$pid")
  else
    echo "  [serial] $tn → $log"
    sim_one_test "$tn" "$log"
    SIM_PID+=("0")
    SIM_EXIT+=("$?")
  fi
done

# Drain parallel PIDs and capture exit codes explicitly.
if [[ $PARALLEL -eq 1 ]]; then
  echo "  Waiting for all sims to complete..."
  for pid in "${running_pids[@]}"; do
    wait "$pid"
    EXIT_OF_PID[$pid]=$?
  done
  for i in "${!TEST_NAMES[@]}"; do
    [[ "${BUILD_STATUS[$i]}" -eq 0 ]] || continue
    p="${SIM_PID[$i]}"
    if [[ "$p" != "0" && -n "${EXIT_OF_PID[$p]:-}" ]]; then
      SIM_EXIT[$i]="${EXIT_OF_PID[$p]}"
    fi
  done
fi

# ── Phase 3: tally ────────────────────────────────────────────────────
# A test PASSES only when:
#   1) build succeeded (BUILD_STATUS[i] == 0)
#   2) sim exit status was 0
#   3) log contains a PASS marker ("PASS: ..." or "*** PASSED ***")
#   4) log contains no FAIL_RE match
PASSED=()
FAILED=()

for i in "${!TEST_NAMES[@]}"; do
  tn="${TEST_NAMES[$i]}"
  if [[ "${BUILD_STATUS[$i]}" -ne 0 ]]; then
    FAILED+=("$tn (build)")
    continue
  fi

  log="${SIM_LOG[$i]}"
  if [[ -z "$log" || ! -f "$log" ]]; then
    FAILED+=("$tn (no log)")
    continue
  fi

  echo "============================================================"
  echo "Sim log tail: $tn"
  echo "============================================================"
  tail -30 "$log"
  echo

  if [[ "${SIM_EXIT[$i]}" -ne 0 ]]; then
    FAILED+=("$tn (sim exit ${SIM_EXIT[$i]})")
    continue
  fi

  if grep -Eq "$FAIL_RE" "$log"; then
    FAILED+=("$tn (failure signature)")
    continue
  fi

  # FAIL_RE above catches real failures (mismatch, ASSERT FAIL, etc.) before
  # we reach this PASS check, so accepting any ^PASS: marker is safe.
  if ! grep -Eq "$VCS_PASS_RE|^PASS:" "$log"; then
    FAILED+=("$tn (no PASS marker)")
    continue
  fi

  PASSED+=("$tn")
done

rm -rf "$SIM_LOG_DIR"

echo "==================== SUMMARY ===================="
echo "Passed: ${#PASSED[@]}"
for t in "${PASSED[@]}"; do
  echo "  PASS  $t"
done

echo "Failed: ${#FAILED[@]}"
for t in "${FAILED[@]}"; do
  echo "  FAIL  $t"
done

if [[ ${#FAILED[@]} -ne 0 ]]; then
  exit 1
fi

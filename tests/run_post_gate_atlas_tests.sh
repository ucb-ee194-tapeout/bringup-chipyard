#!/usr/bin/env bash
set -u
set -o pipefail

usage() {
  cat <<'EOF'
Usage:
  tests/run_post_gate_atlas_tests.sh [options]

This script:
  1. builds the top-level handwritten Atlas baremetal assembly tests from
     generators/atlas-npu/baremetal/assembly/*.S
  2. builds all tests/saturn_atlas_*.c drivers
  3. runs both post-synthesis and post-PAR gate-level sims with redo

Options:
  --only-syn       Run redo-sim-syn only
  --only-par       Run redo-sim-par only
  --skip-build     Reuse existing test binaries
  --dry-run        Print the commands without executing them
  -j, --jobs N     Parallelism for build steps
  -h, --help       Show this help

Environment overrides:
  CONFIG        VLSI config to use for gate-level sim      (default: EE290TapeoutConfig)
  TOP           Top module for vlsi/Makefile              (default: ChipTop)
  VLSI_TOP      VLSI top override                         (default: $TOP)
  VLSI_OBJ_DIR  VLSI obj dir                              (default: build)
  LOADMEM       Passed through to vlsi make               (default: 1)
  LOG_ROOT      Directory for script-owned logs

Notes:
  - The handwritten Atlas assembly sweep intentionally follows the current
    baremetal flow, which wires in the top-level assembly/*.S tests.
  - The cooperative Saturn/Atlas coverage comes from tests/saturn_atlas_*.c.
  - The gate-level redo targets share sim run directories, so this script
    runs the sims serially on purpose.
EOF
}

die() {
  echo "Error: $*" >&2
  exit 1
}

default_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN
  else
    echo 1
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHIPYARD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ASM_ROOT="$CHIPYARD_ROOT/generators/atlas-npu/baremetal"
ASM_TESTS_DIR="$ASM_ROOT/tests"
SATURN_TESTS_DIR="$CHIPYARD_ROOT/tests"
VLSI_DIR="$CHIPYARD_ROOT/vlsi"

CONFIG="${CONFIG:-EE290TapeoutConfig}"
TOP="${TOP:-ChipTop}"
VLSI_TOP="${VLSI_TOP:-$TOP}"
VLSI_OBJ_DIR="${VLSI_OBJ_DIR:-build}"
LOADMEM="${LOADMEM:-1}"
BUILD_JOBS="${BUILD_JOBS:-$(default_jobs)}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_ROOT="${LOG_ROOT:-$SATURN_TESTS_DIR/post-gate-logs/$TIMESTAMP}"

RUN_SYN=1
RUN_PAR=1
SKIP_BUILD=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only-syn)
      RUN_PAR=0
      shift
      ;;
    --only-par)
      RUN_SYN=0
      shift
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -j|--jobs)
      [[ $# -ge 2 ]] || die "missing value for $1"
      BUILD_JOBS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[[ "$BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || die "jobs must be a positive integer"
[[ $RUN_SYN -eq 1 || $RUN_PAR -eq 1 ]] || die "nothing to do: both stages are disabled"

[[ -d "$ASM_ROOT" ]] || die "missing baremetal directory: $ASM_ROOT"
[[ -d "$SATURN_TESTS_DIR" ]] || die "missing tests directory: $SATURN_TESTS_DIR"
[[ -d "$VLSI_DIR" ]] || die "missing vlsi directory: $VLSI_DIR"

mkdir -p "$LOG_ROOT"

run_and_tee() {
  local log="$1"
  shift

  mkdir -p "$(dirname "$log")"

  if [[ $DRY_RUN -eq 1 ]]; then
    {
      printf '[dry-run] '
      printf '%q ' "$@"
      printf '\n'
    } | tee "$log"
    return 0
  fi

  (
    set -o pipefail
    "$@" 2>&1 | tee "$log"
  )
}

FAIL_RE='FAIL:|Assertion failed: \*\*\* FAILED \*\*\*|Fatal:|FAILED \(\*+\)|\*\*\* FAILED \*\*\*|ASSERT FAIL|Assertion .* failed|\bMISMATCH\b|\b[Mm]ismatch\b'
PASS_RE='\*\*\* PASSED \*\*\*|^PASS:'

classify_log() {
  local rc="$1"
  local log="$2"

  if [[ $DRY_RUN -eq 1 ]]; then
    echo "pass"
    return 0
  fi
  if [[ "$rc" -ne 0 ]]; then
    echo "sim exit $rc"
    return 0
  fi
  if grep -Eq "$FAIL_RE" "$log"; then
    echo "failure signature"
    return 0
  fi
  if ! grep -Eq "$PASS_RE" "$log"; then
    echo "no PASS marker"
    return 0
  fi
  echo "pass"
}

declare -a TEST_LABELS=()
declare -a TEST_BINARIES=()
declare -a TEST_IDS=()
declare -a SATURN_TARGETS=()

mapfile -t ASM_SOURCES < <(find "$ASM_ROOT/assembly" -maxdepth 1 -type f -name '*.S' | sort)
mapfile -t SATURN_SOURCES < <(find "$SATURN_TESTS_DIR" -maxdepth 1 -type f -name 'saturn_atlas_*.c' | sort)

[[ ${#ASM_SOURCES[@]} -gt 0 ]] || die "no top-level assembly tests found in $ASM_ROOT/assembly"
[[ ${#SATURN_SOURCES[@]} -gt 0 ]] || die "no saturn_atlas_*.c tests found in $SATURN_TESTS_DIR"

for src in "${ASM_SOURCES[@]}"; do
  stem="$(basename "$src" .S)"
  TEST_LABELS+=("assembly/$stem")
  TEST_BINARIES+=("$ASM_TESTS_DIR/build/atlas_${stem}.riscv")
  TEST_IDS+=("assembly__atlas_${stem}")
done

for src in "${SATURN_SOURCES[@]}"; do
  stem="$(basename "$src" .c)"
  SATURN_TARGETS+=("$stem")
  TEST_LABELS+=("saturn/$stem")
  TEST_BINARIES+=("$SATURN_TESTS_DIR/build/${stem}.riscv")
  TEST_IDS+=("saturn__${stem}")
done

echo "Chipyard root:      $CHIPYARD_ROOT"
echo "Gate-level config:  $CONFIG"
echo "TOP / VLSI_TOP:     $TOP / $VLSI_TOP"
echo "VLSI obj dir:       $VLSI_OBJ_DIR"
echo "Assembly tests:     ${#ASM_SOURCES[@]}"
echo "Saturn tests:       ${#SATURN_SOURCES[@]}"
echo "Total sims/stage:   ${#TEST_LABELS[@]}"
if [[ $RUN_SYN -eq 1 && $RUN_PAR -eq 1 ]]; then
  echo "Stages:             redo-sim-syn, redo-sim-par"
elif [[ $RUN_SYN -eq 1 ]]; then
  echo "Stages:             redo-sim-syn"
else
  echo "Stages:             redo-sim-par"
fi
if [[ $DRY_RUN -eq 1 ]]; then
  echo "Mode:               dry-run"
fi
echo "Log root:           $LOG_ROOT"
echo

if [[ $SKIP_BUILD -eq 0 ]]; then
  echo "============================================================"
  echo "Build handwritten assembly tests"
  echo "============================================================"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] CMAKE_BUILD_PARALLEL_LEVEL=$BUILD_JOBS make -C $ASM_ROOT build-assembly"
  else
    (
      export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
      make -C "$ASM_ROOT" build-assembly
    ) || exit 1
  fi
  echo

  echo "============================================================"
  echo "Build saturn_atlas tests"
  echo "============================================================"
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "[dry-run] cmake -S $SATURN_TESTS_DIR -B $SATURN_TESTS_DIR/build -DCMAKE_BUILD_TYPE=Debug"
    printf '[dry-run] cmake --build %q --target ' "$SATURN_TESTS_DIR/build"
    printf '%q ' "${SATURN_TARGETS[@]}"
    printf -- '-j%s\n' "$BUILD_JOBS"
  else
    cmake -S "$SATURN_TESTS_DIR" -B "$SATURN_TESTS_DIR/build" -DCMAKE_BUILD_TYPE=Debug || exit 1
    cmake --build "$SATURN_TESTS_DIR/build" --target "${SATURN_TARGETS[@]}" -j"$BUILD_JOBS" || exit 1
  fi
  echo
fi

if [[ $DRY_RUN -eq 0 ]]; then
  for bin in "${TEST_BINARIES[@]}"; do
    [[ -f "$bin" ]] || die "expected binary not found: $bin"
  done
fi

echo "============================================================"
echo "Prepare vlsi buildfile"
echo "============================================================"
BUILD_PREP_LOG="$LOG_ROOT/buildfile.log"
VLSI_PREP_CMD=(
  make -C "$VLSI_DIR" buildfile
  "CONFIG=$CONFIG"
  "TOP=$TOP"
  "VLSI_TOP=$VLSI_TOP"
  "VLSI_OBJ_DIR=$VLSI_OBJ_DIR"
)
run_and_tee "$BUILD_PREP_LOG" "${VLSI_PREP_CMD[@]}" || exit 1
echo

declare -a SYN_PASSED=()
declare -a SYN_FAILED=()
declare -a PAR_PASSED=()
declare -a PAR_FAILED=()

run_stage() {
  local target="$1"
  local stage_dir="$2"
  local stage_name="$3"
  local total="${#TEST_LABELS[@]}"
  local i rc outcome log
  local -a stage_cmd

  mkdir -p "$stage_dir"

  for i in "${!TEST_LABELS[@]}"; do
    local label="${TEST_LABELS[$i]}"
    local binary="${TEST_BINARIES[$i]}"
    local test_id="${TEST_IDS[$i]}"
    log="$stage_dir/${test_id}.log"

    echo "============================================================"
    printf '[%s %d/%d] %s\n' "$stage_name" "$((i + 1))" "$total" "$label"
    echo "Binary: $binary"
    echo "Log:    $log"
    echo "============================================================"

    stage_cmd=(
      make -C "$VLSI_DIR" "$target"
      "CONFIG=$CONFIG"
      "TOP=$TOP"
      "VLSI_TOP=$VLSI_TOP"
      "VLSI_OBJ_DIR=$VLSI_OBJ_DIR"
      "BINARY=$binary"
      "LOADMEM=$LOADMEM"
    )

    if run_and_tee "$log" "${stage_cmd[@]}"; then
      rc=0
    else
      rc=$?
    fi

    outcome="$(classify_log "$rc" "$log")"
    if [[ "$outcome" == "pass" ]]; then
      if [[ "$stage_name" == "post-syn" ]]; then
        SYN_PASSED+=("$label")
      else
        PAR_PASSED+=("$label")
      fi
      echo "Result: PASS"
    else
      if [[ "$stage_name" == "post-syn" ]]; then
        SYN_FAILED+=("$label :: $outcome :: $log")
      else
        PAR_FAILED+=("$label :: $outcome :: $log")
      fi
      echo "Result: FAIL ($outcome)"
      if [[ $DRY_RUN -eq 0 ]]; then
        echo "Log tail:"
        tail -30 "$log"
      fi
    fi
    echo
  done
}

if [[ $RUN_SYN -eq 1 ]]; then
  run_stage "redo-sim-syn" "$LOG_ROOT/post-syn" "post-syn"
fi

if [[ $RUN_PAR -eq 1 ]]; then
  run_stage "redo-sim-par" "$LOG_ROOT/post-par" "post-par"
fi

echo "==================== SUMMARY ===================="
echo "Log root: $LOG_ROOT"
if [[ $RUN_SYN -eq 1 ]]; then
  echo "post-syn passed: ${#SYN_PASSED[@]}"
  for item in "${SYN_PASSED[@]}"; do
    echo "  PASS  $item"
  done
  echo "post-syn failed: ${#SYN_FAILED[@]}"
  for item in "${SYN_FAILED[@]}"; do
    echo "  FAIL  $item"
  done
fi
if [[ $RUN_PAR -eq 1 ]]; then
  echo "post-par passed: ${#PAR_PASSED[@]}"
  for item in "${PAR_PASSED[@]}"; do
    echo "  PASS  $item"
  done
  echo "post-par failed: ${#PAR_FAILED[@]}"
  for item in "${PAR_FAILED[@]}"; do
    echo "  FAIL  $item"
  done
fi

if [[ ${#SYN_FAILED[@]} -ne 0 || ${#PAR_FAILED[@]} -ne 0 ]]; then
  exit 1
fi

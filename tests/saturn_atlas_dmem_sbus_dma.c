#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"
#include "saturn_atlas_stress_common.h"
#include "saturn_atlas_dmem_sbus_dma_golden.h"
#include "kernel_sas_dmem_sbus_dma.h"

/*
 * T4: Saturn cached data memory (private dcache → SBUS → DRAM)
 * hammered by an RVV stride-1 sweep, while Atlas runs sustained
 * 8-channel DMA on disjoint DRAM tiles. Both TL masters contend for
 * SBUS xbar slots but never touch each other's destination memory —
 * so the contention is at the SBUS arbiter, not at DRAM.
 *
 * "DMEM" here means Saturn's *cached* data memory backed by DRAM via
 * the dcache, not a separately addressable TCM (EE290SimConfig does
 * not enable WithTCM; the TCM-coverage variant lives in the
 * EE290SimConfigTCM Chipyard config).
 *
 * Three-pass A/B/A measurement protocol:
 *   Pass 1 (solo_a):     Saturn alone, Atlas idle. Baseline.
 *   Pass 2 (contended):  Saturn + Atlas DMA pump concurrent.
 *   Pass 3 (solo_b):     Saturn alone again. Bracketing solo run
 *                        cross-checks pass 1 and reveals
 *                        dcache-residency drift between baselines.
 *
 * Pass criteria:
 *   1. Saturn `big_arr` final state matches scalar-replay golden
 *      across all three passes (Saturn's RVV math correct).
 *   2. Atlas DRAM tiles match DMA round-trip golden (Atlas DMA
 *      correct under Saturn SBUS pressure).
 *   3. Atlas was still running when Saturn's contended sweep
 *      finished (direct overlap proof; sampled via atlas_halted()
 *      immediately after Saturn's RVV body, not inferred from
 *      cycle ratios).
 *
 * The contended-vs-solo cycle ratio is reported as a DIAGNOSTIC and
 * does not fail the test on its own. A hardcoded 1.10x threshold
 * is uncalibrated for this config; treat the printed ratio as a
 * calibration target until a known-good baseline anchors it.
 *
 * Target config: EE290SimConfig.
 */

#define SAS_T4_INPUT_OFF_BASE   0x0000U
#define SAS_T4_OUTPUT_OFF_BASE  0x2000U
#define SAS_T4_TILE_OFF_STRIDE  0x0400U

/*
 * Saturn scratch buffer size. EE290SimConfig's Shuttle L1d is 16 KiB
 * (64 sets × 4 ways × 64 B lines), so 64 KiB is 4× L1d capacity and
 * defeats it on a stride-1 sweep. Provisional first-land number;
 * tuning knob via -DSAS_T4_BUF_BYTES_OVERRIDE if calibration shows
 * the contention diagnostic ratio is too close to 1.0×.
 */
#ifndef SAS_T4_BUF_BYTES
#define SAS_T4_BUF_BYTES        (64U * 1024U)
#endif
#define SAS_T4_BUF_LEN          (SAS_T4_BUF_BYTES / 4U)

/*
 * RVV multiplier used by both the kernel and the scalar replay. Any
 * coprime to 2^32 works; pick a small odd prime so the scalar replay
 * is cheap.
 */
#define SAS_T4_MULTIPLIER       7919U

/*
 * Static-storage scratch buffer. .bss → DRAM at link time. 128-B
 * alignment matches RVV LMUL=4 line stride (32-B vector lines × 4 =
 * 128 B), keeping vle32/vse32 timing stable across runs.
 */
static int32_t big_arr[SAS_T4_BUF_LEN] __attribute__((aligned(128)));

/* Pre-fill seed: deterministic so both passes start from the same state. */
static int32_t saturn_seed_value(uint32_t i)
{
    return (int32_t)((i * 2654435761U) ^ 0xC0FFEE00U);
}

/* ── Helpers ────────────────────────────────────────────────────────── */

static void sas_t4_seed_dram_inputs(void)
{
    for (uint32_t t = 0; t < SAS_DMEM_N_TILES; t++) {
        uintptr_t base = SAS_DRAM_BASE + SAS_T4_INPUT_OFF_BASE
                       + t * SAS_T4_TILE_OFF_STRIDE;
        for (uint32_t w = 0; w < SAS_DMEM_WORDS_PER_TILE; w++) {
            atlas_write32(base + w * 4, sas_dmem_input_tiles[t][w]);
        }
    }
    for (uint32_t t = 0; t < SAS_DMEM_N_TILES; t++) {
        uintptr_t base = SAS_DRAM_BASE + SAS_T4_OUTPUT_OFF_BASE
                       + t * SAS_T4_TILE_OFF_STRIDE;
        for (uint32_t w = 0; w < SAS_DMEM_WORDS_PER_TILE; w++) {
            atlas_write32(base + w * 4, 0);
        }
    }
    asm volatile ("fence" ::: "memory");
}

static void sas_t4_seed_big_arr(void)
{
    for (uint32_t i = 0; i < SAS_T4_BUF_LEN; i++) {
        big_arr[i] = saturn_seed_value(i);
    }
    asm volatile ("fence" ::: "memory");
}

/*
 * Saturn RVV sweep: big_arr[i] = big_arr[i] * MULTIPLIER for every i.
 * Returns elapsed cycles measured via rdcycle.
 *
 * Barrier discipline (memory clobber + fence both before and after)
 * keeps the compiler from hoisting RVV ops outside the rdcycle
 * pair and forces in-flight stores to drain before the timestamps.
 */
static uint64_t sas_t4_saturn_sweep(void)
{
    uint64_t start, end;
    asm volatile ("fence" ::: "memory");
    asm volatile ("rdcycle %0" : "=r"(start) :: "memory");

    size_t remaining = SAS_T4_BUF_LEN;
    int32_t *p = big_arr;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m4(remaining);
        vint32m4_t v = __riscv_vle32_v_i32m4(p, vl);
        v = __riscv_vmul_vx_i32m4(v, (int32_t)SAS_T4_MULTIPLIER, vl);
        __riscv_vse32_v_i32m4(p, v, vl);
        remaining -= vl;
        p += vl;
    }

    asm volatile ("fence" ::: "memory");
    asm volatile ("rdcycle %0" : "=r"(end) :: "memory");
    return end - start;
}

static int sas_t4_check_big_arr(void)
{
    int fails = 0;
    for (uint32_t i = 0; i < SAS_T4_BUF_LEN; i++) {
        int32_t expected =
            (int32_t)((uint32_t)saturn_seed_value(i) * SAS_T4_MULTIPLIER);
        if (big_arr[i] != expected) {
            if (fails < 4) {
                printf("  BIG_ARR MISMATCH i=%u: got %d exp %d\n",
                       i, big_arr[i], expected);
            }
            fails++;
        }
    }
    if (fails > 4) {
        printf("  ... and %d more big_arr mismatches\n", fails - 4);
    }
    return fails;
}

static int sas_t4_check_dram_outputs(void)
{
    int fails = 0;
    for (uint32_t t = 0; t < SAS_DMEM_N_TILES; t++) {
        uintptr_t base = SAS_DRAM_BASE + SAS_T4_OUTPUT_OFF_BASE
                       + t * SAS_T4_TILE_OFF_STRIDE;
        for (uint32_t w = 0; w < SAS_DMEM_WORDS_PER_TILE; w++) {
            uint32_t got = atlas_read32(base + w * 4);
            uint32_t exp = sas_dmem_expected_tiles[t][w];
            if (got != exp) {
                if (fails < 4) {
                    printf("  DMA OUTPUT MISMATCH tile=%u w=%u: got 0x%08x exp 0x%08x\n",
                           t, w, got, exp);
                }
                fails++;
            }
        }
    }
    if (fails > 4) {
        printf("  ... and %d more DMA mismatches\n", fails - 4);
    }
    return fails;
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("Saturn <-> Atlas T4: Saturn cached DMEM RVV sweep + Atlas DMA pump\n");
    printf("  big_arr len=%u (=%u KiB), multiplier=%u\n",
           SAS_T4_BUF_LEN, SAS_T4_BUF_BYTES / 1024U, SAS_T4_MULTIPLIER);

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);

    int fails = 0;

    /* ── Pass 1: solo_a baseline ────────────────────────────────── */
    printf("Pass 1 (solo_a): Saturn alone, Atlas idle ...\n");
    sas_t4_seed_big_arr();
    uint64_t cycles_solo_a = sas_t4_saturn_sweep();
    int big_fails_a = sas_t4_check_big_arr();
    if (big_fails_a) {
        printf("FAIL: solo_a big_arr mismatch (%d errors)\n", big_fails_a);
        return 1;
    }
    printf("  cycles_solo_a=%lu\n", (unsigned long)cycles_solo_a);

    /* ── Pass 2: contended ─────────────────────────────────────── */
    printf("Pass 2 (contended): Saturn + Atlas DMA concurrent ...\n");
    sas_t4_seed_dram_inputs();
    sas_t4_seed_big_arr();

    printf("Staging Atlas kernel (%u words) ...\n",
           KERNEL_SAS_DMEM_SBUS_DMA_LEN);
    if (sas_stage_imem_from_array(kernel_sas_dmem_sbus_dma,
                                  KERNEL_SAS_DMEM_SBUS_DMA_LEN)) {
        printf("FAIL: IMEM readback mismatch\n");
        return 1;
    }

    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint64_t cycles_contended = sas_t4_saturn_sweep();

    /*
     * Direct overlap proof: sample atlas_halted() the moment Saturn's
     * sweep ends. If Atlas is still running, Saturn was contended
     * with Atlas DMA for the full sweep duration. If Atlas already
     * halted, Saturn finished slower than Atlas — the contention
     * window may have been only partial.
     */
    int atlas_still_running_at_sweep_end = atlas_halted() ? 0 : 1;

    if (sas_wait_halted_with_dbg0(1, "dmem-sbus-dma")) {
        printf("  cycles_contended=%lu cycles_solo_a=%lu\n",
               (unsigned long)cycles_contended,
               (unsigned long)cycles_solo_a);
        return 1;
    }

    uint32_t atlas_cycles  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t atlas_instcnt = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);

    int big_fails_b = sas_t4_check_big_arr();
    if (big_fails_b == 0) {
        printf("PASS: contended big_arr (Saturn RVV math correct under SBUS pressure)\n");
    } else {
        printf("FAIL: %d contended big_arr mismatches\n", big_fails_b);
        fails += big_fails_b;
    }

    int dma_fails = sas_t4_check_dram_outputs();
    if (dma_fails == 0) {
        printf("PASS: Atlas DMA round-trip echo (8 tiles × %u B)\n",
               SAS_DMEM_TILE_BYTES);
    } else {
        printf("FAIL: %d DMA mismatches\n", dma_fails);
        fails += dma_fails;
    }

    /* ── Pass 3: solo_b baseline (bracketing solo) ─────────────── */
    printf("Pass 3 (solo_b): Saturn alone again, Atlas halted ...\n");
    sas_t4_seed_big_arr();
    uint64_t cycles_solo_b = sas_t4_saturn_sweep();
    int big_fails_c = sas_t4_check_big_arr();
    if (big_fails_c == 0) {
        printf("PASS: solo_b big_arr (Saturn RVV math still correct)\n");
    } else {
        printf("FAIL: %d solo_b big_arr mismatches\n", big_fails_c);
        fails += big_fails_c;
    }
    printf("  cycles_solo_b=%lu\n", (unsigned long)cycles_solo_b);

    /* ── Direct overlap check (pass criterion) ─────────────────── */
    if (atlas_still_running_at_sweep_end) {
        printf("PASS: Atlas was still running when contended sweep ended "
               "(direct overlap proof)\n");
    } else {
        printf("FAIL: Atlas halted before Saturn's contended sweep finished — "
               "no sustained overlap\n");
        fails++;
    }

    /* ── Diagnostic: cycle ratios (no fail criterion) ──────────── */
    uint64_t solo_avg = (cycles_solo_a + cycles_solo_b) / 2ULL;
    uint64_t ratio_x100 = (cycles_contended * 100ULL) / solo_avg;
    printf("  cycles_solo_a=%lu cycles_solo_b=%lu cycles_contended=%lu\n",
           (unsigned long)cycles_solo_a,
           (unsigned long)cycles_solo_b,
           (unsigned long)cycles_contended);
    printf("  atlas_cycles=%u atlas_instcnt=%u\n", atlas_cycles, atlas_instcnt);
    printf("  DIAGNOSTIC: contended/avg_solo = %lu / 100\n",
           (unsigned long)ratio_x100);
    if (ratio_x100 < 105ULL) {
        printf("  NOTE: ratio < 105/100 — SBUS contention may not be "
               "reaching the bus (try -DSAS_T4_BUF_BYTES_OVERRIDE=larger)\n");
    }

    if (fails) {
        printf("FAIL: T4 (%d issues)\n", fails);
        return 1;
    }
    printf("PASS: T4 saturn_atlas_dmem_sbus_dma\n");
    return 0;
}

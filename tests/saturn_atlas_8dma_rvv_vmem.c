#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"
#include "saturn_atlas_stress_common.h"
#include "saturn_atlas_8dma_rvv_vmem_golden.h"
#include "kernel_sas_8dma_rvv_vmem.h"

/*
 * T2: 8-channel DMA echo + Saturn RVV in the highest-numbered VMEM bank.
 *
 * Atlas drives all 8 DMA channels through the lower banks (channel 7 reuses
 * bank 0 at slot 1) inside an 8-round outer loop, giving Saturn a wide
 * overlap window.  The final VMEM bank is reserved for Saturn — this driver
 * runs an RVV stride-1 sweep there *during* the Atlas DMA window
 * to stress SBUS xbar fairness and per-bank arbiter independence.
 *
 * Coverage scope (narrower than the original master plan said):
 *   - All 8 DMA channels in flight per round (queue saturation).
 *   - Per-bank arbiter independence (DMA touches lower banks, Saturn
 *     touches the final bank; if any bank's pressure backed up onto another,
 *     Saturn's iteration count would drop or its final state would
 *     diverge).
 *   - SBUS xbar fairness between Saturn (TL master) and Atlas DMA
 *     (TL master).
 *   - Pure DMA echo, no VPU/MXU compute. T1 covers LSU+VPU
 *     contention; T5 covers compute under multi-bank pressure.
 *
 * Pass criteria:
 *   1. DRAM output tiles match generator golden (DMA round-trip
 *      integrity preserved despite Saturn TL contention).
 *   2. Final-bank VMEM state == triangular sum (Saturn writes
 *      reached VMEM despite lowest priority).
 *   3. Saturn iteration counter >= SAS_FLOOR while Atlas was still
 *      running (proves sustained Saturn-during-Atlas overlap; floor
 *      derived from Atlas's 8-round budget, see below).
 *
 * Target config: EE290SimConfig.
 */

#define SAS_INPUT_OFF_BASE     0x0000U   /* DRAM offsets for IN tiles */
#define SAS_OUTPUT_OFF_BASE    0x2000U   /* DRAM offsets for OUT tiles */
#define SAS_TILE_OFF_STRIDE    0x0400U   /* 1024 B between tiles */

/*
 * VMEM region given to Saturn in the highest-numbered bank. Use only slot 0
 * (the first 1024 words = 4 KiB) so the rest of that bank stays a guard band.
 */
#define SAS_SATURN_BANK_IDX    (ATLAS_VMEM_NUM_BANKS - 1UL)
#define SAS_SATURN_VMEM_OFF    (SAS_SATURN_BANK_IDX * SAS_BANK_BYTES)
#define SAS_SATURN_LEN         1024U   /* 1024 i32 elements = 4 KiB */

/*
 * Atlas runs N_ROUNDS=8 DMA rounds; each round is ~1.5K Atlas cycles
 * for 16 transfers through one shared TileLinkAdapter, giving roughly
 * a 12K-cycle Atlas window.  One Saturn iteration is a 256-i32 RVV
 * sweep at ~1K cycles via SBUS, so we expect ~12 iters of overlap;
 * floor of 4 leaves ample slack but still detects starvation.  The
 * cap is high so it's clearly a safety bound, not a normal stop.
 */
#define SAS_FLOOR              4U
#define SAS_TARGET_ITERS       256U

/* ── Helpers ────────────────────────────────────────────────────────── */

static void sas_seed_dram_inputs(void)
{
    for (uint32_t t = 0; t < SAS_8DMA_N_TILES; t++) {
        uintptr_t base = SAS_DRAM_BASE + SAS_INPUT_OFF_BASE
                       + t * SAS_TILE_OFF_STRIDE;
        for (uint32_t w = 0; w < SAS_8DMA_WORDS_PER_TILE; w++) {
            atlas_write32(base + w * 4, sas_8dma_input_tiles[t][w]);
        }
    }
    /* Zero output region so a stuck-at-old-value would also fail. */
    for (uint32_t t = 0; t < SAS_8DMA_N_TILES; t++) {
        uintptr_t base = SAS_DRAM_BASE + SAS_OUTPUT_OFF_BASE
                       + t * SAS_TILE_OFF_STRIDE;
        for (uint32_t w = 0; w < SAS_8DMA_WORDS_PER_TILE; w++) {
            atlas_write32(base + w * 4, 0);
        }
    }
    asm volatile ("fence" ::: "memory");
}

static int sas_check_dram_outputs(void)
{
    int fails = 0;
    for (uint32_t t = 0; t < SAS_8DMA_N_TILES; t++) {
        uintptr_t base = SAS_DRAM_BASE + SAS_OUTPUT_OFF_BASE
                       + t * SAS_TILE_OFF_STRIDE;
        for (uint32_t w = 0; w < SAS_8DMA_WORDS_PER_TILE; w++) {
            uint32_t got = atlas_read32(base + w * 4);
            uint32_t exp = sas_8dma_expected_tiles[t][w];
            if (got != exp) {
                if (fails < 4) {
                    printf("  DRAM OUTPUT MISMATCH tile=%u w=%u: got 0x%08x exp 0x%08x\n",
                           t, w, got, exp);
                }
                fails++;
            }
        }
    }
    if (fails > 4) {
        printf("  ... and %d more DRAM mismatches\n", fails - 4);
    }
    return fails;
}

/*
 * Saturn RVV sweep over the highest-numbered VMEM bank. Each iteration loads
 * a stride-1 vector from that bank, adds the iteration index, and stores it
 * back.
 * Returns iterations completed before Atlas halted.
 */
static uint32_t sas_saturn_last_bank_sweep(uint32_t target_iters)
{
    int32_t * const bank = (int32_t *)SAS_BANK_BASE(SAS_SATURN_BANK_IDX);
    uint32_t completed = 0;

    for (uint32_t it = 0; it < target_iters; it++) {
        size_t remaining = SAS_SATURN_LEN;
        int32_t *p = bank;
        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e32m4(remaining);
            vint32m4_t v = __riscv_vle32_v_i32m4(p, vl);
            v = __riscv_vadd_vx_i32m4(v, (int32_t)it, vl);
            __riscv_vse32_v_i32m4(p, v, vl);
            remaining -= vl;
            p += vl;
        }
        completed++;

        /* Bail early if Atlas already halted — we have enough progress. */
        if (atlas_halted()) break;
    }

    asm volatile ("fence" ::: "memory");
    return completed;
}

static int sas_check_last_bank_final(uint32_t iters_done)
{
    /*
     * Saturn writes start with that bank zero (we initialise it below) and
     * each iteration adds `it` to every word.  Final value at word w =
     * sum_{it=0}^{iters_done-1} it = iters_done * (iters_done - 1) / 2.
     * Holds for every word since the seed was uniform zero.
     */
    int32_t expected = (int32_t)((iters_done * (iters_done - 1)) / 2);
    int fails = 0;
    int32_t *bank = (int32_t *)SAS_BANK_BASE(SAS_SATURN_BANK_IDX);
    for (uint32_t w = 0; w < SAS_SATURN_LEN; w++) {
        int32_t got = (int32_t)atlas_read32((uintptr_t)&bank[w]);
        if (got != expected) {
            if (fails < 4) {
                printf("  BANK%lu MISMATCH w=%u: got %d exp %d (iters=%u)\n",
                       (unsigned long)SAS_SATURN_BANK_IDX,
                       w, got, expected, iters_done);
            }
            fails++;
        }
    }
    if (fails > 4) {
        printf("  ... and %d more bank-%lu mismatches\n",
               fails - 4, (unsigned long)SAS_SATURN_BANK_IDX);
    }
    return fails;
}

static void sas_zero_last_bank_region(void)
{
    int32_t *bank = (int32_t *)SAS_BANK_BASE(SAS_SATURN_BANK_IDX);
    for (uint32_t w = 0; w < SAS_SATURN_LEN; w++) {
        atlas_write32((uintptr_t)&bank[w], 0);
    }
    asm volatile ("fence" ::: "memory");
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("Saturn <-> Atlas T2: 8-channel DMA echo + Saturn RVV bank-%lu sweep\n",
           (unsigned long)SAS_SATURN_BANK_IDX);

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);

    sas_seed_dram_inputs();
    sas_zero_last_bank_region();

    printf("Staging Atlas kernel (%u words) ...\n",
           KERNEL_SAS_8DMA_RVV_VMEM_LEN);
    if (sas_stage_imem_from_array(kernel_sas_8dma_rvv_vmem,
                                  KERNEL_SAS_8DMA_RVV_VMEM_LEN)) {
        printf("FAIL: IMEM readback mismatch\n");
        return 1;
    }

#ifdef SAS_T2_SKIP_SATURN
    printf("Starting Atlas (Saturn sweep DISABLED — diagnostic mode) ...\n");
#else
    printf("Starting Atlas + Saturn RVV bank-%lu sweep ...\n",
           (unsigned long)SAS_SATURN_BANK_IDX);
#endif
    atlas_start();
    asm volatile ("fence" ::: "memory");

#ifdef SAS_T2_SKIP_SATURN
    /*
     * Diagnostic bisect mode: skip the Saturn-side RVV sweep entirely.
     * If Atlas halts with dbg0=1 in this mode, the Atlas DMA pump
     * itself is fine; the live-lock is in the Saturn-vs-VMEM-TL path.
     * If Atlas still hangs, the bug is on the Atlas DMA side.
     */
    uint32_t iters_done = 0;
#else
    /* Saturn runs its sweep in parallel with Atlas's DMA storm. */
    uint32_t iters_done = sas_saturn_last_bank_sweep(SAS_TARGET_ITERS);
#endif

    if (sas_wait_halted_with_dbg0(1, "8dma+rvv")) {
        printf("  Saturn iters_done=%u (capped at %u)\n",
               iters_done, SAS_TARGET_ITERS);
        return 1;
    }

    uint32_t cycles  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t instcnt = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    printf("  cycles=%u instcnt=%u saturn_iters=%u\n",
           cycles, instcnt, iters_done);

    int fails = 0;

    /* Pass criterion 1: DRAM round-trip echo correct. */
    int dram_fails = sas_check_dram_outputs();
    if (dram_fails == 0) {
        printf("PASS: DRAM round-trip echo (%u tiles × %u bytes)\n",
               SAS_8DMA_N_TILES, SAS_8DMA_TILE_BYTES);
    } else {
        printf("FAIL: %d DRAM word mismatches\n", dram_fails);
        fails += dram_fails;
    }

#ifdef SAS_T2_SKIP_SATURN
    printf("PASS: Atlas DMA pump completed in skip-Saturn diagnostic mode\n");
    if (fails) {
        printf("FAIL: T2 (skip-Saturn) (%d issues)\n", fails);
        return 1;
    }
    printf("PASS: T2 saturn_atlas_8dma_rvv_vmem (skip-Saturn)\n");
    return 0;
#endif

    /* Pass criterion 2: reserved-bank final state matches Saturn's expected sum. */
    int bank_fails = sas_check_last_bank_final(iters_done);
    if (bank_fails == 0) {
        printf("PASS: bank-%lu final state (Saturn writes reached VMEM)\n",
               (unsigned long)SAS_SATURN_BANK_IDX);
    } else {
        printf("FAIL: %d bank-%lu word mismatches\n",
               bank_fails, (unsigned long)SAS_SATURN_BANK_IDX);
        fails += bank_fails;
    }

    /* Pass criterion 3: Saturn forward-progress floor. */
    if (iters_done >= SAS_FLOOR) {
        printf("PASS: Saturn forward-progress floor (%u >= %u)\n",
               iters_done, SAS_FLOOR);
    } else {
        printf("FAIL: Saturn iters %u below floor %u — possible starvation\n",
               iters_done, SAS_FLOOR);
        fails++;
    }

    if (fails) {
        printf("FAIL: T2 (%d issues)\n", fails);
        return 1;
    }
    printf("PASS: T2 saturn_atlas_8dma_rvv_vmem\n");
    return 0;
}

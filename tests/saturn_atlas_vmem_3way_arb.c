#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"
#include "saturn_atlas_stress_common.h"
#include "saturn_atlas_vmem_3way_arb_golden.h"
#include "kernel_sas_vmem_3way_arb.h"

/*
 * T1: VMEM 3-way arbitration, single bank (bank 3).
 *
 * Three VMEM client classes hammer bank 3 at the same time, on
 * disjoint sub-regions so the goldens stay deterministic:
 *   - vector LSU sub-region (Atlas VLOAD/VSTORE identity loop)
 *   - DMA  sub-region (Atlas DMA.LOAD/STORE round-trip)
 *   - Saturn sub-region (host RVV vle32/vse32 sweep, lowest priority)
 *
 * Coverage scope: this test exercises the vector-LSU vs DMA vs TL
 * rungs of the priority ladder. It does NOT exercise scalar LSU
 * (scalar SW/LW); adding scalar traffic on bank 3 would risk
 * tripping the same-cycle scalar/vector bank-conflict assertions.
 *
 * Pass criteria:
 *   1. DMA round-trip echo — DMA OUT region == DMA IN tile (DMA wasn't
 *      corrupted by vector-LSU/Saturn contention on bank 3).
 *   2. LSU sub-region final state == initial pattern (VLOAD/VSTORE
 *      identity preserved despite DMA + Saturn pressure).
 *   3. Saturn sub-region final state == sum_{i=0}^{iters-1} i (Saturn
 *      writes reached VMEM despite lowest priority).
 *   4. Saturn forward-progress floor (>=16) — calibrated against the
 *      Atlas window of ~128K cycles (2 outer × 64 inner LSU+DMA
 *      iterations). Saturn's 1 KiB sweep at ~500 cycles → 256 iters
 *      of overlap expected; floor is generous slack against scheduler
 *      noise.
 *   5. Sim-log assertion check (caught by run_saturn_atlas_tests.sh
 *      FAIL_RE; baremetal driver cannot directly observe HW asserts).
 *
 * Bank 3 host base = ATLAS_VMEM_BASE + 3 * 0x30000 = 0x20090000.
 *   LSU sub-region:     0x20090000..0x200903FF   (1024 B = 1 mreg tile)
 *   DMA sub-region:     0x20090400..0x200907FF   (1024 B)
 *   Saturn sub-region:  0x20090800..0x20090BFF   (1024 B)
 *
 * Target config: EE290SimConfig.
 */

#define SAS_3WAY_BANK            3U
#define SAS_3WAY_BANK_BASE       SAS_BANK_BASE(SAS_3WAY_BANK)
#define SAS_3WAY_LSU_OFF         0x000UL
#define SAS_3WAY_DMA_OFF         0x400UL
#define SAS_3WAY_SATURN_OFF      0x800UL

#define SAS_3WAY_DMA_DRAM_IN     0x0000UL
#define SAS_3WAY_DMA_DRAM_OUT    0x1000UL

#define SAS_3WAY_FLOOR           32U     /* calibrated to ~200K cycle Atlas window (4x128 LSU+DMA loop) */
#define SAS_3WAY_TARGET_ITERS    4096U   /* safety cap chosen so Saturn does NOT exhaust before Atlas halts */

/* ── Helpers ────────────────────────────────────────────────────────── */

static void sas_3way_seed_dram_input(void)
{
    uintptr_t in_base = SAS_DRAM_BASE + SAS_3WAY_DMA_DRAM_IN;
    uintptr_t out_base = SAS_DRAM_BASE + SAS_3WAY_DMA_DRAM_OUT;
    for (uint32_t w = 0; w < SAS_3WAY_WORDS_PER_TILE; w++) {
        atlas_write32(in_base + w * 4, sas_3way_dma_input_tile[w]);
        atlas_write32(out_base + w * 4, 0);
    }
    asm volatile ("fence" ::: "memory");
}

static void sas_3way_seed_lsu_subregion(void)
{
    uintptr_t base = SAS_3WAY_BANK_BASE + SAS_3WAY_LSU_OFF;
    for (uint32_t w = 0; w < SAS_3WAY_WORDS_PER_TILE; w++) {
        atlas_write32(base + w * 4, sas_3way_lsu_init_pattern[w]);
    }
    asm volatile ("fence" ::: "memory");
}

static void sas_3way_zero_saturn_subregion(void)
{
    uintptr_t base = SAS_3WAY_BANK_BASE + SAS_3WAY_SATURN_OFF;
    for (uint32_t w = 0; w < SAS_3WAY_WORDS_PER_TILE; w++) {
        atlas_write32(base + w * 4, 0);
    }
    asm volatile ("fence" ::: "memory");
}

static int sas_3way_check_dram_output(void)
{
    int fails = 0;
    uintptr_t base = SAS_DRAM_BASE + SAS_3WAY_DMA_DRAM_OUT;
    for (uint32_t w = 0; w < SAS_3WAY_WORDS_PER_TILE; w++) {
        uint32_t got = atlas_read32(base + w * 4);
        uint32_t exp = sas_3way_dma_expected_tile[w];
        if (got != exp) {
            if (fails < 4) {
                printf("  DMA OUTPUT MISMATCH w=%u: got 0x%08x exp 0x%08x\n",
                       w, got, exp);
            }
            fails++;
        }
    }
    if (fails > 4) {
        printf("  ... and %d more DMA mismatches\n", fails - 4);
    }
    return fails;
}

static int sas_3way_check_lsu_subregion(void)
{
    int fails = 0;
    uintptr_t base = SAS_3WAY_BANK_BASE + SAS_3WAY_LSU_OFF;
    for (uint32_t w = 0; w < SAS_3WAY_WORDS_PER_TILE; w++) {
        uint32_t got = atlas_read32(base + w * 4);
        uint32_t exp = sas_3way_lsu_init_pattern[w];
        if (got != exp) {
            if (fails < 4) {
                printf("  LSU MISMATCH w=%u: got 0x%08x exp 0x%08x\n",
                       w, got, exp);
            }
            fails++;
        }
    }
    if (fails > 4) {
        printf("  ... and %d more LSU mismatches\n", fails - 4);
    }
    return fails;
}

static int sas_3way_check_saturn_subregion(uint32_t iters_done)
{
    int fails = 0;
    int32_t expected = (int32_t)((iters_done * (iters_done - 1)) / 2);
    int32_t *base = (int32_t *)(SAS_3WAY_BANK_BASE + SAS_3WAY_SATURN_OFF);
    for (uint32_t w = 0; w < SAS_3WAY_WORDS_PER_TILE; w++) {
        int32_t got = (int32_t)atlas_read32((uintptr_t)&base[w]);
        if (got != expected) {
            if (fails < 4) {
                printf("  SATURN MISMATCH w=%u: got %d exp %d (iters=%u)\n",
                       w, got, expected, iters_done);
            }
            fails++;
        }
    }
    if (fails > 4) {
        printf("  ... and %d more Saturn mismatches\n", fails - 4);
    }
    return fails;
}

/*
 * Saturn RVV stride-1 sweep over the Saturn sub-region. Each iteration
 * adds `it` to every word.  Loops until target_iters or Atlas halts.
 */
static uint32_t sas_3way_saturn_sweep(uint32_t target_iters)
{
    int32_t * const region =
        (int32_t *)(SAS_3WAY_BANK_BASE + SAS_3WAY_SATURN_OFF);
    uint32_t completed = 0;

    for (uint32_t it = 0; it < target_iters; it++) {
        size_t remaining = SAS_3WAY_WORDS_PER_TILE;
        int32_t *p = region;
        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e32m4(remaining);
            vint32m4_t v = __riscv_vle32_v_i32m4(p, vl);
            v = __riscv_vadd_vx_i32m4(v, (int32_t)it, vl);
            __riscv_vse32_v_i32m4(p, v, vl);
            remaining -= vl;
            p += vl;
        }
        completed++;

        if (atlas_halted()) break;
    }

    asm volatile ("fence" ::: "memory");
    return completed;
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("Saturn <-> Atlas T1: VMEM 3-way arbitration on bank 3\n");

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);

    sas_3way_seed_dram_input();
    sas_3way_seed_lsu_subregion();
    sas_3way_zero_saturn_subregion();

    printf("Staging Atlas kernel (%u words) ...\n",
           KERNEL_SAS_VMEM_3WAY_ARB_LEN);
    if (sas_stage_imem_from_array(kernel_sas_vmem_3way_arb,
                                  KERNEL_SAS_VMEM_3WAY_ARB_LEN)) {
        printf("FAIL: IMEM readback mismatch\n");
        return 1;
    }

    printf("Starting Atlas + Saturn RVV bank-3 sweep ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t iters_done = sas_3way_saturn_sweep(SAS_3WAY_TARGET_ITERS);

    if (sas_wait_halted_with_dbg0(1, "3way-arb")) {
        printf("  Saturn iters_done=%u (capped at %u)\n",
               iters_done, SAS_3WAY_TARGET_ITERS);
        return 1;
    }

    uint32_t cycles  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t instcnt = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    printf("  cycles=%u instcnt=%u saturn_iters=%u\n",
           cycles, instcnt, iters_done);

    int fails = 0;

    int dma_fails = sas_3way_check_dram_output();
    if (dma_fails == 0) {
        printf("PASS: DMA round-trip echo on bank 3 (1024 B)\n");
    } else {
        printf("FAIL: %d DMA mismatches\n", dma_fails);
        fails += dma_fails;
    }

    int lsu_fails = sas_3way_check_lsu_subregion();
    if (lsu_fails == 0) {
        printf("PASS: LSU sub-region preserved (VLOAD/VSTORE identity)\n");
    } else {
        printf("FAIL: %d LSU mismatches\n", lsu_fails);
        fails += lsu_fails;
    }

    int sat_fails = sas_3way_check_saturn_subregion(iters_done);
    if (sat_fails == 0) {
        printf("PASS: Saturn sub-region final state (Saturn writes reached VMEM)\n");
    } else {
        printf("FAIL: %d Saturn mismatches\n", sat_fails);
        fails += sat_fails;
    }

    if (iters_done >= SAS_3WAY_FLOOR) {
        printf("PASS: Saturn forward-progress floor (%u >= %u)\n",
               iters_done, SAS_3WAY_FLOOR);
    } else {
        printf("FAIL: Saturn iters %u below floor %u — possible starvation\n",
               iters_done, SAS_3WAY_FLOOR);
        fails++;
    }

    if (fails) {
        printf("FAIL: T1 (%d issues)\n", fails);
        return 1;
    }
    printf("PASS: T1 saturn_atlas_vmem_3way_arb\n");
    return 0;
}

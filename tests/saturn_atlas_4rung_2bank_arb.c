#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"
#include "saturn_atlas_stress_common.h"
#include "saturn_atlas_4rung_2bank_arb_golden.h"
#include "kernel_sas_4rung_2bank_arb.h"

/*
 * Saturn <-> Atlas 4-rung 2-bank arbitration test.
 *
 * Bank 3: Atlas vector LSU + Atlas DMA (ch0) + Saturn TL.
 * Bank 4: Atlas scalar LSU + Atlas DMA (ch1) + Saturn TL.
 *
 * Together, the test exercises both legal sub-chains of the per-bank
 * priority ladder concurrently: vector > DMA > TL on bank 3, and
 * scalar > DMA > TL on bank 4. Same-bank scalar+vector LSU is
 * intentionally avoided (it's hardware-asserted illegal).
 *
 * Pass criteria:
 *   1. Atlas halts cleanly (dbg0=1, halt_reason=ECALL).
 *   2. DRAM round-trip echo correct on both banks.
 *   3. Bank 3 vector LSU sub-region preserved (VLOAD/VSTORE identity).
 *   4. Bank 4 scalar LSU: byte 8 == scalar_init[0] AND byte 12 ==
 *      scalar_init[1] (proves LW fired and result was consumed by SW).
 *   5. Saturn dual-bank sweep reached both banks (triangular sums
 *      match on both Saturn sub-regions).
 *   6. Saturn iters_done >= floor (forward-progress).
 *
 * Target config: EE290SimConfig.
 */

#define BANK3                    3U
#define BANK4                    4U
#define BANK3_BASE               SAS_BANK_BASE(BANK3)   /* 0x20060000 */
#define BANK4_BASE               SAS_BANK_BASE(BANK4)   /* 0x20080000 */

#define LSU_OFF                  0x000UL                /* +0x000: vector LSU (b3) / scalar LSU (b4) */
#define DMA_OFF                  0x400UL                /* +0x400: DMA sub-region */
#define SATURN_OFF               0x800UL                /* +0x800: Saturn TL sub-region */

#define DMA_DRAM_IN_B3           0x0000UL
#define DMA_DRAM_IN_B4           0x0400UL
#define DMA_DRAM_OUT_B3          0x2000UL
#define DMA_DRAM_OUT_B4          0x2400UL

#define SATURN_REGION_LEN        SAS_4RUNG_WORDS_PER_TILE   /* 256 words / 1 KiB */

#define SAS_4RUNG_FLOOR          16U     /* dual-bank Saturn floor */
#define SAS_4RUNG_TARGET_ITERS   4096U   /* safety cap chosen so Saturn does NOT exhaust before Atlas halts */

/* Bank 4 scalar LSU pass-criterion offsets (must match the kernel). */
#define SCALAR_LW_OFF_A          0       /* kernel reads word at byte 0 */
#define SCALAR_LW_OFF_B          4       /* kernel reads word at byte 4 */
#define SCALAR_SW_OFF_A          8       /* kernel writes loaded[A] here */
#define SCALAR_SW_OFF_B          12      /* kernel writes loaded[B] here */

/* ── Pre-fill helpers (host TL Put through vmemNode) ──────────────── */

static void seed_dram_inputs(void)
{
    uintptr_t in3 = SAS_DRAM_BASE + DMA_DRAM_IN_B3;
    uintptr_t in4 = SAS_DRAM_BASE + DMA_DRAM_IN_B4;
    for (uint32_t w = 0; w < SAS_4RUNG_WORDS_PER_TILE; w++) {
        atlas_write32(in3 + w * 4, sas_4rung_dma_input_b3[w]);
        atlas_write32(in4 + w * 4, sas_4rung_dma_input_b4[w]);
    }
    /* Zero outputs so a stuck-at-old-value also fails. */
    uintptr_t out3 = SAS_DRAM_BASE + DMA_DRAM_OUT_B3;
    uintptr_t out4 = SAS_DRAM_BASE + DMA_DRAM_OUT_B4;
    for (uint32_t w = 0; w < SAS_4RUNG_WORDS_PER_TILE; w++) {
        atlas_write32(out3 + w * 4, 0);
        atlas_write32(out4 + w * 4, 0);
    }
    asm volatile ("fence" ::: "memory");
}

static void seed_lsu_b3(void)
{
    uintptr_t base = BANK3_BASE + LSU_OFF;
    for (uint32_t w = 0; w < SAS_4RUNG_WORDS_PER_TILE; w++) {
        atlas_write32(base + w * 4, sas_4rung_lsu_init_b3[w]);
    }
    asm volatile ("fence" ::: "memory");
}

static void seed_scalar_b4(void)
{
    uintptr_t base = BANK4_BASE + LSU_OFF;
    for (uint32_t w = 0; w < SAS_4RUNG_WORDS_PER_TILE; w++) {
        atlas_write32(base + w * 4, sas_4rung_scalar_init_b4[w]);
    }
    asm volatile ("fence" ::: "memory");
}

static void zero_saturn_regions(void)
{
    uintptr_t s3 = BANK3_BASE + SATURN_OFF;
    uintptr_t s4 = BANK4_BASE + SATURN_OFF;
    for (uint32_t w = 0; w < SATURN_REGION_LEN; w++) {
        atlas_write32(s3 + w * 4, 0);
        atlas_write32(s4 + w * 4, 0);
    }
    asm volatile ("fence" ::: "memory");
}

/* ── Saturn dual-bank RVV sweep ───────────────────────────────────── */

static uint32_t saturn_dual_bank_sweep(uint32_t target_iters)
{
    int32_t * const region_b3 = (int32_t *)(BANK3_BASE + SATURN_OFF);
    int32_t * const region_b4 = (int32_t *)(BANK4_BASE + SATURN_OFF);
    uint32_t completed = 0;

    for (uint32_t it = 0; it < target_iters; it++) {
        /* Sweep bank 3 sub-region */
        size_t remaining = SATURN_REGION_LEN;
        int32_t *p = region_b3;
        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e32m4(remaining);
            vint32m4_t v = __riscv_vle32_v_i32m4(p, vl);
            v = __riscv_vadd_vx_i32m4(v, (int32_t)it, vl);
            __riscv_vse32_v_i32m4(p, v, vl);
            remaining -= vl;
            p += vl;
        }

        /* Sweep bank 4 sub-region */
        remaining = SATURN_REGION_LEN;
        p = region_b4;
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

/* ── Post-halt verification helpers ───────────────────────────────── */

static int check_dram_echo(uintptr_t out_base, const uint32_t *expected,
                           const char *label)
{
    int fails = 0;
    for (uint32_t w = 0; w < SAS_4RUNG_WORDS_PER_TILE; w++) {
        uint32_t got = atlas_read32(out_base + w * 4);
        if (got != expected[w]) {
            if (fails < 4) {
                printf("  DMA OUT %s mismatch w=%u: got 0x%08x exp 0x%08x\n",
                       label, w, got, expected[w]);
            }
            fails++;
        }
    }
    if (fails > 4) printf("  ... and %d more %s mismatches\n", fails - 4, label);
    return fails;
}

static int check_lsu_b3(void)
{
    int fails = 0;
    uintptr_t base = BANK3_BASE + LSU_OFF;
    for (uint32_t w = 0; w < SAS_4RUNG_WORDS_PER_TILE; w++) {
        uint32_t got = atlas_read32(base + w * 4);
        if (got != sas_4rung_lsu_init_b3[w]) {
            if (fails < 4) {
                printf("  LSU b3 mismatch w=%u: got 0x%08x exp 0x%08x\n",
                       w, got, sas_4rung_lsu_init_b3[w]);
            }
            fails++;
        }
    }
    if (fails > 4) printf("  ... and %d more LSU b3 mismatches\n", fails - 4);
    return fails;
}

static int check_scalar_b4(void)
{
    int fails = 0;
    uintptr_t base = BANK4_BASE + LSU_OFF;

    /* Bytes 0 and 4 are read by the kernel but never written; should
     * still equal scalar_init pre-fill. */
    uint32_t got_a = atlas_read32(base + SCALAR_LW_OFF_A);
    uint32_t got_b = atlas_read32(base + SCALAR_LW_OFF_B);
    uint32_t exp_a = sas_4rung_scalar_init_b4[SCALAR_LW_OFF_A / 4];
    uint32_t exp_b = sas_4rung_scalar_init_b4[SCALAR_LW_OFF_B / 4];
    if (got_a != exp_a) {
        printf("  SCALAR b4 byte 0 (LW source) mismatch: got 0x%08x exp 0x%08x\n",
               got_a, exp_a);
        fails++;
    }
    if (got_b != exp_b) {
        printf("  SCALAR b4 byte 4 (LW source) mismatch: got 0x%08x exp 0x%08x\n",
               got_b, exp_b);
        fails++;
    }

    /* Bytes 8 and 12 receive the LW result via SW. They must equal the
     * loaded values (= scalar_init[0] and [1] respectively). */
    uint32_t got_sa = atlas_read32(base + SCALAR_SW_OFF_A);
    uint32_t got_sb = atlas_read32(base + SCALAR_SW_OFF_B);
    if (got_sa != exp_a) {
        printf("  SCALAR b4 byte 8 (SW dest, LW source byte 0) mismatch: "
               "got 0x%08x exp 0x%08x (proves LW fired and was consumed)\n",
               got_sa, exp_a);
        fails++;
    }
    if (got_sb != exp_b) {
        printf("  SCALAR b4 byte 12 (SW dest, LW source byte 4) mismatch: "
               "got 0x%08x exp 0x%08x (proves LW fired and was consumed)\n",
               got_sb, exp_b);
        fails++;
    }

    return fails;
}

static int check_saturn_region(uintptr_t base, uint32_t iters_done,
                               const char *label)
{
    int32_t expected = (int32_t)((iters_done * (iters_done - 1)) / 2);
    int fails = 0;
    int32_t *region = (int32_t *)base;
    for (uint32_t w = 0; w < SATURN_REGION_LEN; w++) {
        int32_t got = (int32_t)atlas_read32((uintptr_t)&region[w]);
        if (got != expected) {
            if (fails < 4) {
                printf("  SATURN %s mismatch w=%u: got %d exp %d (iters=%u)\n",
                       label, w, got, expected, iters_done);
            }
            fails++;
        }
    }
    if (fails > 4) printf("  ... and %d more %s mismatches\n", fails - 4, label);
    return fails;
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("Saturn <-> Atlas 4-rung 2-bank arbitration\n");

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);

    seed_dram_inputs();
    seed_lsu_b3();
    seed_scalar_b4();
    zero_saturn_regions();

    printf("Staging Atlas kernel (%u words) ...\n", KERNEL_SAS_4RUNG_2BANK_ARB_LEN);
    if (sas_stage_imem_from_array(kernel_sas_4rung_2bank_arb,
                                  KERNEL_SAS_4RUNG_2BANK_ARB_LEN)) {
        printf("FAIL: IMEM readback mismatch\n");
        return 1;
    }

    printf("Starting Atlas + Saturn dual-bank sweep ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t iters_done = saturn_dual_bank_sweep(SAS_4RUNG_TARGET_ITERS);

    if (sas_wait_halted_with_dbg0(1, "4rung-2bank")) {
        printf("  saturn_iters=%u (cap %u)\n", iters_done, SAS_4RUNG_TARGET_ITERS);
        return 1;
    }

    uint32_t cycles  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t instcnt = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    printf("  cycles=%u instcnt=%u saturn_iters=%u\n",
           cycles, instcnt, iters_done);

    int fails = 0;

    /* (2) DRAM round-trip echo on both banks */
    int dma_b3_fails = check_dram_echo(SAS_DRAM_BASE + DMA_DRAM_OUT_B3,
                                       sas_4rung_dma_expected_b3, "b3");
    int dma_b4_fails = check_dram_echo(SAS_DRAM_BASE + DMA_DRAM_OUT_B4,
                                       sas_4rung_dma_expected_b4, "b4");
    if (dma_b3_fails == 0 && dma_b4_fails == 0) {
        printf("PASS: DMA round-trip echo on bank 3 + bank 4\n");
    } else {
        fails += dma_b3_fails + dma_b4_fails;
    }

    /* (3) Bank 3 vector LSU identity preserved */
    int lsu_fails = check_lsu_b3();
    if (lsu_fails == 0) {
        printf("PASS: bank 3 vector LSU sub-region preserved\n");
    } else {
        fails += lsu_fails;
    }

    /* (4) Bank 4 scalar LSU consumed-load proof */
    int scalar_fails = check_scalar_b4();
    if (scalar_fails == 0) {
        printf("PASS: bank 4 scalar LSU LW result consumed by SW (load fired)\n");
    } else {
        fails += scalar_fails;
    }

    /* (5) Saturn dual-bank sweeps reached both banks */
    int sat_b3_fails = check_saturn_region(BANK3_BASE + SATURN_OFF,
                                           iters_done, "b3");
    int sat_b4_fails = check_saturn_region(BANK4_BASE + SATURN_OFF,
                                           iters_done, "b4");
    if (sat_b3_fails == 0 && sat_b4_fails == 0) {
        printf("PASS: Saturn dual-bank sweep reached both banks\n");
    } else {
        fails += sat_b3_fails + sat_b4_fails;
    }

    /* (6) Saturn forward-progress floor */
    if (iters_done >= SAS_4RUNG_FLOOR) {
        printf("PASS: Saturn forward-progress floor (%u >= %u)\n",
               iters_done, SAS_4RUNG_FLOOR);
    } else {
        printf("FAIL: Saturn iters %u below floor %u\n",
               iters_done, SAS_4RUNG_FLOOR);
        fails++;
    }

    if (fails) {
        printf("FAIL: 4-rung 2-bank arb (%d issues)\n", fails);
        return 1;
    }
    printf("PASS: saturn_atlas_4rung_2bank_arb\n");
    return 0;
}

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"
#include "saturn_atlas_stress_common.h"
#include "saturn_atlas_vmem_dram_dual_golden.h"
#include "kernel_sas_vmem_dram_dual.h"

/*
 * T5: Saturn↔Atlas VMEM + DRAM dual-target stress.
 *
 * Closes the coverage gap flagged by PI: Saturn issuing concurrent
 * vload/vstore to BOTH VMEM (bank 3 + 0x1000, 4 KiB) and DRAM (64 KiB
 * ring rotated 4 KiB-per-iteration) while Atlas runs T1.5-style
 * 4-rung 2-bank LSU + DMA on banks 3 and 4.
 *
 * Saturn loop chunk-interleaves VMEM and DRAM RVV ops within each
 * iteration: for each vsetvl chunk, do vle/vadd/vse on VMEM, memory
 * barrier, then vle/vadd/vse on DRAM, memory barrier, advance both
 * pointers.  This locks the "Saturn keeps both target paths hot at
 * vector-chunk granularity" claim by construction.
 *
 * Pass criteria (8):
 *   1. Atlas halts cleanly (dbg0=1, halt_reason=ECALL).
 *   2. Atlas DMA round-trip echo on bank 3 + bank 4.
 *   3. Atlas vector LSU identity on bank 3 +0x000.
 *   4. Atlas scalar LSU consumed-load proof on bank 4 (bytes 0/4
 *      untouched, bytes 8/12 == loaded values).
 *   5. Saturn VMEM final state == triangular sum (closed-form).
 *   6. Saturn DRAM final state == seed + per-chunk last-touch sum
 *      (closed-form, uint32_t arithmetic to mirror RVV vadd.vx
 *      modulo 2^32 wrap).
 *   7. Saturn forward-progress floor: iters_done >= 16 (one full
 *      DRAM-ring traversal).
 *   8. Direct overlap proof: atlas_alive_after_floor == 1 — Saturn
 *      completed >= 16 dual-target iterations while Atlas was live.
 *
 * OoO DRAM-response probing intentionally out of scope.
 *
 * Target config: EE290SimConfig.
 */

/* ── Constants ──────────────────────────────────────────────────────── */

#define SAS_T5_VMEM_BANK          3U
#define SAS_T5_VMEM_OFF           0x1000UL
#define SAS_T5_VMEM_WORDS         1024U          /* 4 KiB */
#define SAS_T5_DRAM_WORDS         (16U * 1024U)  /* 64 KiB total */
#define SAS_T5_DRAM_CHUNK_WORDS   1024U          /* 4 KiB per chunk */
#define SAS_T5_NUM_CHUNKS         16U

#define SAS_T5_FLOOR              16U            /* one full ring traversal */
#define SAS_T5_TARGET_ITERS       4096U          /* generous safety cap */

#define SAS_T5_DMA_DRAM_IN_B3     0x0000UL
#define SAS_T5_DMA_DRAM_IN_B4     0x0400UL
#define SAS_T5_DMA_DRAM_OUT_B3    0x2000UL
#define SAS_T5_DMA_DRAM_OUT_B4    0x2400UL
#define SAS_T5_ATLAS_DMA_END      0x2800UL       /* OUT_b4 + 1 KiB */

#define SAS_T5_VMEM_BASE          (SAS_BANK_BASE(SAS_T5_VMEM_BANK) + SAS_T5_VMEM_OFF)

/* Atlas-internal regions (T1.5 layout reused) */
#define BANK3                    3U
#define BANK4                    4U
#define BANK3_BASE               SAS_BANK_BASE(BANK3)
#define BANK4_BASE               SAS_BANK_BASE(BANK4)
#define LSU_OFF                  0x000UL
#define DMA_OFF                  0x400UL
#define SCALAR_LW_OFF_A          0
#define SCALAR_LW_OFF_B          4
#define SCALAR_SW_OFF_A          8
#define SCALAR_SW_OFF_B          12

/* ── Saturn DRAM scratch ring ───────────────────────────────────────── */

/*
 * Static .bss → DRAM at link time. 4 KiB alignment so chunk boundaries
 * land on page boundaries (mirrors DMA's 4 KiB max transfer size).
 * 64 KiB total is 4× Shuttle L1d (16 KiB), defeating L1d residency on
 * stride-1 sweeps. L2/DRAMSim reach is a diagnostic property — see
 * master plan risk #10.
 */
static int32_t saturn_dram_ring[SAS_T5_DRAM_WORDS]
    __attribute__((aligned(4096)));

static inline uint32_t saturn_dram_seed(uint32_t w)
{
    /* deterministic per-word seed; cheap, low collision */
    return (w * 0x9E3779B1U) ^ 0xA5A5A5A5U;
}

/* ── Pre-start helpers ──────────────────────────────────────────────── */

static int sas_t5_check_ring_disjoint(void)
{
    uintptr_t ring_lo = (uintptr_t)&saturn_dram_ring[0];
    uintptr_t ring_hi = (uintptr_t)&saturn_dram_ring[SAS_T5_DRAM_WORDS];
    uintptr_t dma_lo  = SAS_DRAM_BASE + 0x0000UL;
    uintptr_t dma_hi  = SAS_DRAM_BASE + SAS_T5_ATLAS_DMA_END;

    printf("Saturn DRAM ring: [0x%016lx, 0x%016lx) (%u KiB)\n",
           (unsigned long)ring_lo, (unsigned long)ring_hi,
           (unsigned)(SAS_T5_DRAM_WORDS * 4U / 1024U));
    printf("Atlas DMA window: [0x%016lx, 0x%016lx)\n",
           (unsigned long)dma_lo, (unsigned long)dma_hi);

    if (!(ring_hi <= dma_lo || ring_lo >= dma_hi)) {
        printf("FAIL: Saturn DRAM ring overlaps Atlas DMA window\n");
        return 1;
    }
    return 0;
}

static void sas_t5_seed_dram_inputs(void)
{
    uintptr_t in3  = SAS_DRAM_BASE + SAS_T5_DMA_DRAM_IN_B3;
    uintptr_t in4  = SAS_DRAM_BASE + SAS_T5_DMA_DRAM_IN_B4;
    uintptr_t out3 = SAS_DRAM_BASE + SAS_T5_DMA_DRAM_OUT_B3;
    uintptr_t out4 = SAS_DRAM_BASE + SAS_T5_DMA_DRAM_OUT_B4;
    for (uint32_t w = 0; w < SAS_T5_WORDS_PER_TILE; w++) {
        atlas_write32(in3 + w * 4, sas_t5_dma_input_b3[w]);
        atlas_write32(in4 + w * 4, sas_t5_dma_input_b4[w]);
        atlas_write32(out3 + w * 4, 0);
        atlas_write32(out4 + w * 4, 0);
    }
    asm volatile ("fence" ::: "memory");
}

static void sas_t5_seed_atlas_lsu(void)
{
    uintptr_t lsu_b3    = BANK3_BASE + LSU_OFF;
    uintptr_t scalar_b4 = BANK4_BASE + LSU_OFF;
    for (uint32_t w = 0; w < SAS_T5_WORDS_PER_TILE; w++) {
        atlas_write32(lsu_b3 + w * 4, sas_t5_lsu_init_b3[w]);
        atlas_write32(scalar_b4 + w * 4, sas_t5_scalar_init_b4[w]);
    }
    asm volatile ("fence" ::: "memory");
}

static void sas_t5_zero_saturn_vmem(void)
{
    uintptr_t base = SAS_T5_VMEM_BASE;
    for (uint32_t w = 0; w < SAS_T5_VMEM_WORDS; w++) {
        atlas_write32(base + w * 4, 0);
    }
    asm volatile ("fence" ::: "memory");
}

static void sas_t5_seed_saturn_dram(void)
{
    for (uint32_t w = 0; w < SAS_T5_DRAM_WORDS; w++) {
        saturn_dram_ring[w] = (int32_t)saturn_dram_seed(w);
    }
    asm volatile ("fence" ::: "memory");
}

/* ── Saturn dual-target loop (chunk-interleaved) ────────────────────── */

static uint32_t saturn_vmem_dram_dual_loop(uint32_t target_iters,
                                           int *atlas_alive_after_floor)
{
    int32_t * const vmem = (int32_t *)SAS_T5_VMEM_BASE;
    uint32_t completed = 0;

    for (uint32_t it = 0; it < target_iters; it++) {
        int32_t *dram = &saturn_dram_ring[(it & 0xFU) * SAS_T5_DRAM_CHUNK_WORDS];
        size_t remaining = SAS_T5_VMEM_WORDS;   /* == SAS_T5_DRAM_CHUNK_WORDS */
        int32_t *pv = vmem;
        int32_t *pd = dram;

        while (remaining > 0) {
            size_t vl = __riscv_vsetvl_e32m4(remaining);

            /* VMEM half */
            vint32m4_t vv = __riscv_vle32_v_i32m4(pv, vl);
            vv = __riscv_vadd_vx_i32m4(vv, (int32_t)it, vl);
            __riscv_vse32_v_i32m4(pv, vv, vl);
            asm volatile ("" ::: "memory");  /* lock VMEM-then-DRAM ordering */

            /* DRAM half — chunk-interleaved with VMEM half */
            vint32m4_t vd = __riscv_vle32_v_i32m4(pd, vl);
            vd = __riscv_vadd_vx_i32m4(vd, (int32_t)it, vl);
            __riscv_vse32_v_i32m4(pd, vd, vl);
            asm volatile ("" ::: "memory");  /* preserve chunk granularity */

            remaining -= vl;
            pv += vl;
            pd += vl;
        }

        completed++;

        /* Snapshot Atlas alive-state right when Saturn finishes one
         * full 64 KiB DRAM-ring traversal — direct overlap proof. */
        if (completed == SAS_T5_FLOOR) {
            *atlas_alive_after_floor = atlas_halted() ? 0 : 1;
        }

        if (atlas_halted()) break;
    }

    asm volatile ("fence" ::: "memory");
    return completed;
}

/* ── Verification ───────────────────────────────────────────────────── */

static int sas_t5_check_dram_echo(uintptr_t out_base, const uint32_t *expected,
                                  const char *label)
{
    int fails = 0;
    for (uint32_t w = 0; w < SAS_T5_WORDS_PER_TILE; w++) {
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

static int sas_t5_check_lsu_b3(void)
{
    int fails = 0;
    uintptr_t base = BANK3_BASE + LSU_OFF;
    for (uint32_t w = 0; w < SAS_T5_WORDS_PER_TILE; w++) {
        uint32_t got = atlas_read32(base + w * 4);
        if (got != sas_t5_lsu_init_b3[w]) {
            if (fails < 4) {
                printf("  LSU b3 mismatch w=%u: got 0x%08x exp 0x%08x\n",
                       w, got, sas_t5_lsu_init_b3[w]);
            }
            fails++;
        }
    }
    if (fails > 4) printf("  ... and %d more LSU b3 mismatches\n", fails - 4);
    return fails;
}

static int sas_t5_check_scalar_b4(void)
{
    int fails = 0;
    uintptr_t base = BANK4_BASE + LSU_OFF;

    /* Bytes 0 and 4 are read but never written — should equal pre-fill. */
    uint32_t got_a = atlas_read32(base + SCALAR_LW_OFF_A);
    uint32_t got_b = atlas_read32(base + SCALAR_LW_OFF_B);
    uint32_t exp_a = sas_t5_scalar_init_b4[SCALAR_LW_OFF_A / 4];
    uint32_t exp_b = sas_t5_scalar_init_b4[SCALAR_LW_OFF_B / 4];
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

    /* Bytes 8 and 12 receive the LW result via SW. */
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

static int sas_t5_check_saturn_vmem(uint32_t iters_done)
{
    int fails = 0;
    /* Closed-form triangular sum: every word receives +0 +1 +... +(N-1) */
    int32_t expected = (int32_t)((iters_done * (iters_done - 1U)) / 2U);
    int32_t *region = (int32_t *)SAS_T5_VMEM_BASE;
    for (uint32_t w = 0; w < SAS_T5_VMEM_WORDS; w++) {
        int32_t got = (int32_t)atlas_read32((uintptr_t)&region[w]);
        if (got != expected) {
            if (fails < 4) {
                printf("  SATURN VMEM mismatch w=%u: got %d exp %d (iters=%u)\n",
                       w, got, expected, iters_done);
            }
            fails++;
        }
    }
    if (fails > 4) printf("  ... and %d more Saturn VMEM mismatches\n", fails - 4);
    return fails;
}

static int sas_t5_check_saturn_dram(uint32_t iters_done)
{
    int fails = 0;
    for (uint32_t w = 0; w < SAS_T5_DRAM_WORDS; w++) {
        uint32_t c = w / SAS_T5_DRAM_CHUNK_WORDS;     /* chunk index 0..15 */

        /* Closed-form per-chunk last-touch sum (uint32_t throughout to
         * mirror RVV vadd.vx modulo 2^32). */
        uint32_t n_hits = (iters_done > c)
            ? (iters_done - c + 15U) / 16U
            : 0U;
        uint32_t inner_sum = n_hits * (uint32_t)c
                           + 16U * (n_hits * (n_hits - 1U) / 2U);
        uint32_t seed = saturn_dram_seed(w);
        uint32_t exp  = seed + inner_sum;             /* wraps mod 2^32 */
        uint32_t got  = (uint32_t)saturn_dram_ring[w];

        if (got != exp) {
            if (fails < 4) {
                printf("  SATURN DRAM mismatch w=%u c=%u: got 0x%08x exp 0x%08x "
                       "(seed=0x%08x n_hits=%u inner_sum=%u iters=%u)\n",
                       w, c, got, exp, seed, n_hits, inner_sum, iters_done);
            }
            fails++;
        }
    }
    if (fails > 4) printf("  ... and %d more Saturn DRAM mismatches\n", fails - 4);
    return fails;
}

/* ── Main ───────────────────────────────────────────────────────────── */

int main(void)
{
    printf("Saturn <-> Atlas T5: VMEM (bank 3 +0x1000) + DRAM (64 KiB ring) dual-target\n");

    /* Pre-start step 1: hard-fail address sanity check. */
    if (sas_t5_check_ring_disjoint()) {
        return 1;
    }

    /* Step 2: stop Atlas, clear debug regs. */
    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);

    /* Steps 3-6: seed Atlas regions, zero Saturn VMEM, seed Saturn DRAM. */
    sas_t5_seed_dram_inputs();
    sas_t5_seed_atlas_lsu();
    sas_t5_zero_saturn_vmem();
    sas_t5_seed_saturn_dram();

    /* Step 7: stage kernel. */
    printf("Staging Atlas kernel (%u words) ...\n", KERNEL_SAS_VMEM_DRAM_DUAL_LEN);
    if (sas_stage_imem_from_array(kernel_sas_vmem_dram_dual,
                                  KERNEL_SAS_VMEM_DRAM_DUAL_LEN)) {
        printf("FAIL: IMEM readback mismatch\n");
        return 1;
    }

    /* Step 8: explicit-zero the overlap-proof state before the loop. */
    int atlas_alive_after_floor = 0;

    printf("Starting Atlas + Saturn VMEM/DRAM dual sweep ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t iters_done = saturn_vmem_dram_dual_loop(SAS_T5_TARGET_ITERS,
                                                     &atlas_alive_after_floor);

    if (sas_wait_halted_with_dbg0(1, "vmem-dram-dual")) {
        printf("  saturn_iters=%u (cap %u) atlas_alive_after_floor=%d\n",
               iters_done, SAS_T5_TARGET_ITERS, atlas_alive_after_floor);
        return 1;
    }

    uint32_t cycles  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t instcnt = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    printf("  cycles=%u instcnt=%u saturn_iters=%u atlas_alive_after_floor=%d\n",
           cycles, instcnt, iters_done, atlas_alive_after_floor);

    int fails = 0;

    /* (2) DMA round-trip echo on both banks */
    int dma_b3_fails = sas_t5_check_dram_echo(SAS_DRAM_BASE + SAS_T5_DMA_DRAM_OUT_B3,
                                              sas_t5_dma_expected_b3, "b3");
    int dma_b4_fails = sas_t5_check_dram_echo(SAS_DRAM_BASE + SAS_T5_DMA_DRAM_OUT_B4,
                                              sas_t5_dma_expected_b4, "b4");
    if (dma_b3_fails == 0 && dma_b4_fails == 0) {
        printf("PASS: Atlas DMA round-trip echo on bank 3 + bank 4\n");
    } else {
        fails += dma_b3_fails + dma_b4_fails;
    }

    /* (3) Bank 3 vector LSU identity */
    int lsu_fails = sas_t5_check_lsu_b3();
    if (lsu_fails == 0) {
        printf("PASS: Atlas vector LSU identity preserved (bank 3 +0x000)\n");
    } else {
        fails += lsu_fails;
    }

    /* (4) Bank 4 scalar LSU consumed-load proof */
    int scalar_fails = sas_t5_check_scalar_b4();
    if (scalar_fails == 0) {
        printf("PASS: Atlas scalar LSU LW result consumed by SW (load fired)\n");
    } else {
        fails += scalar_fails;
    }

    /* (5) Saturn VMEM triangular sum */
    int sat_vmem_fails = sas_t5_check_saturn_vmem(iters_done);
    if (sat_vmem_fails == 0) {
        printf("PASS: Saturn VMEM final state (triangular sum, bank 3 +0x1000)\n");
    } else {
        fails += sat_vmem_fails;
    }

    /* (6) Saturn DRAM per-chunk last-touch sum */
    int sat_dram_fails = sas_t5_check_saturn_dram(iters_done);
    if (sat_dram_fails == 0) {
        printf("PASS: Saturn DRAM final state (per-chunk sum, 64 KiB ring)\n");
    } else {
        fails += sat_dram_fails;
    }

    /* (7) Forward-progress floor */
    if (iters_done >= SAS_T5_FLOOR) {
        printf("PASS: Saturn forward-progress floor (%u >= %u)\n",
               iters_done, SAS_T5_FLOOR);
    } else {
        printf("FAIL: Saturn iters %u below floor %u\n",
               iters_done, SAS_T5_FLOOR);
        fails++;
    }

    /* (8) Direct overlap proof */
    if (atlas_alive_after_floor == 1) {
        printf("PASS: direct overlap proof (Atlas live at Saturn iter %u)\n",
               SAS_T5_FLOOR);
    } else {
        printf("FAIL: Atlas had already halted by Saturn iter %u "
               "(atlas_alive_after_floor=%d)\n",
               SAS_T5_FLOOR, atlas_alive_after_floor);
        fails++;
    }

    if (fails) {
        printf("FAIL: T5 saturn_atlas_vmem_dram_dual (%d issues)\n", fails);
        return 1;
    }
    printf("PASS: saturn_atlas_vmem_dram_dual\n");
    return 0;
}

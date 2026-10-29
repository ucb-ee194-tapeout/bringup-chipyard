#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"
#include "saturn_atlas_smolvla_compose_large_golden.h"

/*
 * Saturn <-> Atlas smolvla composition — multi-tile variant.
 *
 * Runs the same pipeline as saturn_atlas_smolvla_compose.c, but in a
 * host-driven outer loop over SAC_L_N_TILES distinct input tiles. Each
 * iteration re-seeds DRAM, reloads IMEM for both phases, and verifies
 * the reduced output against SAC_L_EXPECT_S[i].
 *
 * Tile count is controlled by SAC_L_N_TILES, which defaults to
 * SAC_L_GOLDEN_N_TILES (= 16) from the generated header. Override at
 * compile time via -DSAC_L_N_TILES=N for dev runs. Tiles 0 and 1 are
 * crafted BF16 edge cases (subnormal-heavy scale, FP32-subnormal
 * product); tiles 2..15 are randn throughput tiles with tile 2
 * bit-identical to the single-tile test's seed=100 anchor.
 */

/* Default tile count: use the full golden set unless overridden. */
#ifndef SAC_L_N_TILES
#define SAC_L_N_TILES SAC_L_GOLDEN_N_TILES
#endif

_Static_assert(SAC_L_N_TILES >= 1,
               "SAC_L_N_TILES must be >= 1");
_Static_assert(SAC_L_N_TILES <= SAC_L_GOLDEN_N_TILES,
               "SAC_L_N_TILES must not exceed SAC_L_GOLDEN_N_TILES");

#define SACL_DRAM_BASE   ((uint64_t)ATLAS_DRAM_BASE)

/* DRAM layout matches single-tile test; see saturn_atlas_smolvla_compose.c
 * for rationale. The outer loop fully overwrites these slots each tile. */
#define SACL_OFF_A_B0    0x0000U
#define SACL_OFF_A_B1    0x0400U
#define SACL_OFF_B_B0    0x0800U
#define SACL_OFF_B_B1    0x0C00U
#define SACL_OFF_Y_B0    0x1000U
#define SACL_OFF_Y_B1    0x1400U
#define SACL_OFF_S_B0    0x0800U
#define SACL_OFF_S_B1    0x0C00U

#define SACL_BANK_ROWS   32U
#define SACL_ROW_BYTES   32U    /* 16 BF16 lanes = 32 bytes */

#define POLL_ITERS       200000000U

/* ── Assembled smolvla kernels (byte-identical to single-tile test) ────── */

#define KERNEL_MUL_LEN 54
static const uint32_t kernel_mul[KERNEL_MUL_LEN] = {
    0x00100E13U, 0x00000293U, 0x0002807FU, 0x40000113U, 0x00000313U, 0x900000B7U, 0x0020837BU, 0x0200007FU,
    0x10000313U, 0x900000B7U, 0x40008093U, 0x0020837BU, 0x0200007FU, 0x00000313U, 0x00030007U, 0x02101067U,
    0x00830087U, 0x02101067U, 0x20000393U, 0x900010B7U, 0x80008093U, 0x002083FBU, 0x0200007FU, 0x30000393U,
    0x900010B7U, 0xC0008093U, 0x002083FBU, 0x0200007FU, 0x20000393U, 0x00038107U, 0x02101067U, 0x00838187U,
    0x02101067U, 0x06100257U, 0x04001067U, 0x40000413U, 0x00042207U, 0x02101067U, 0x00842287U, 0x02101067U,
    0x40000413U, 0x900011B7U, 0x022411FBU, 0x0200107FU, 0x50000413U, 0x900011B7U, 0x40018193U, 0x022411FBU,
    0x0200107FU, 0x00100093U, 0xC1009073U, 0x00000073U, 0xC10E1073U, 0x00000073U,
};

#define KERNEL_RED_LEN 40
static const uint32_t kernel_red[KERNEL_RED_LEN] = {
    0x00100E13U, 0x00000293U, 0x0002807FU, 0x40000113U, 0x00000313U, 0x900000B7U, 0x0020837BU, 0x0200007FU,
    0x10000313U, 0x900000B7U, 0x40008093U, 0x0020837BU, 0x0200007FU, 0x00000313U, 0x00030007U, 0x02101067U,
    0x00830087U, 0x02101067U, 0x42000257U, 0x02501067U, 0x20000413U, 0x00042207U, 0x02101067U, 0x00842287U,
    0x02101067U, 0x20000413U, 0x900011B7U, 0x80018193U, 0x022411FBU, 0x0200107FU, 0x30000413U, 0x900011B7U,
    0xC0018193U, 0x022411FBU, 0x0200107FU, 0x00100093U, 0xC1009073U, 0x00000073U, 0xC10E1073U, 0x00000073U,
};

/* ── Helpers ───────────────────────────────────────────────────────────── */

static void seed_bank(uintptr_t dram_off, const uint16_t rows[][SAC_L_LANES], uint32_t first_row)
{
    uintptr_t base = SACL_DRAM_BASE + dram_off;
    for (uint32_t r = 0; r < SACL_BANK_ROWS; r++) {
        const uint16_t *row = rows[first_row + r];
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t val = ((uint32_t)row[2 * w + 1] << 16) | (uint32_t)row[2 * w];
            atlas_write32(base + r * SACL_ROW_BYTES + w * 4, val);
        }
    }
}

static int load_kernel(const uint32_t *words, uint32_t n, const char *label, uint32_t tile)
{
    for (uint32_t i = 0; i < n; i++) {
        atlas_imem_write(i, words[i]);
    }
    asm volatile ("fence" ::: "memory");
    int fail = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t got = atlas_imem_read(i);
        if (got != words[i]) {
            printf("  IMEM MISMATCH[tile=%u %s word=%u]: got 0x%08x exp 0x%08x\n",
                   tile, label, i, got, words[i]);
            fail++;
        }
    }
    return fail;
}

static int wait_halted(uint32_t *status_out)
{
    uint32_t status = 0;
    for (uint32_t i = 0; i < POLL_ITERS; i++) {
        status = atlas_status();
        if (status & ATLAS_STATUS_HALTED_BIT) break;
    }
    *status_out = status;
    return (status & ATLAS_STATUS_HALTED_BIT) ? 0 : 1;
}

/* ── Saturn BF16 interlude (per-row, identical semantics to single-tile) ─ */

static void saturn_interlude_row(uintptr_t y_addr, uintptr_t yprime_addr, uint16_t scale_bits)
{
    uint32_t scale_u32 = ((uint32_t)scale_bits) << 16;
    float scale_f;
    memcpy(&scale_f, &scale_u32, sizeof(float));

    size_t vl = __riscv_vsetvl_e16m1(SAC_L_LANES);
    vuint16m1_t y_u16  = __riscv_vle16_v_u16m1((const uint16_t *)y_addr, vl);
    vuint32m2_t y_u32  = __riscv_vzext_vf2_u32m2(y_u16, vl);
    vuint32m2_t y_bits = __riscv_vsll_vx_u32m2(y_u32, 16, vl);
    vfloat32m2_t y_f32 = __riscv_vreinterpret_v_u32m2_f32m2(y_bits);
    vfloat32m2_t p_f32 = __riscv_vfmul_vf_f32m2(y_f32, scale_f, vl);
    vuint32m2_t p_bits = __riscv_vreinterpret_v_f32m2_u32m2(p_f32);
    vuint32m2_t p_hi   = __riscv_vsrl_vx_u32m2(p_bits, 16, vl);
    vuint16m1_t p_u16  = __riscv_vnsrl_wx_u16m1(p_hi, 0, vl);
    __riscv_vse16_v_u16m1((uint16_t *)yprime_addr, p_u16, vl);
}

static void run_interlude(const uint16_t scale[SAC_L_TILE_ROWS])
{
    for (uint32_t bank = 0; bank < 2; bank++) {
        uintptr_t y_base  = SACL_DRAM_BASE + (bank ? SACL_OFF_Y_B1 : SACL_OFF_Y_B0);
        uintptr_t yp_base = SACL_DRAM_BASE + (bank ? SACL_OFF_A_B1 : SACL_OFF_A_B0);
        for (uint32_t r = 0; r < SACL_BANK_ROWS; r++) {
            uintptr_t y_addr  = y_base  + r * SACL_ROW_BYTES;
            uintptr_t yp_addr = yp_base + r * SACL_ROW_BYTES;
            saturn_interlude_row(y_addr, yp_addr, scale[r]);
        }
    }
}

/* ── Verification ──────────────────────────────────────────────────────── */

static int verify_bank(uintptr_t dram_off, const uint16_t expected[][SAC_L_LANES],
                       uint32_t first_row, uint32_t tile, const char *label)
{
    uintptr_t base = SACL_DRAM_BASE + dram_off;
    int mism = 0;   /* per-lane count */
    for (uint32_t r = 0; r < SACL_BANK_ROWS; r++) {
        const uint16_t *exp = expected[first_row + r];
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t got = atlas_read32(base + r * SACL_ROW_BYTES + w * 4);
            uint16_t got_lo = (uint16_t)(got & 0xFFFF);
            uint16_t got_hi = (uint16_t)(got >> 16);
            uint16_t exp_lo = exp[2 * w];
            uint16_t exp_hi = exp[2 * w + 1];
            if (got_lo != exp_lo) {
                if (mism < 4) {
                    printf("  mismatch: tile=%u bank=%s row=%u lane=%u got=0x%04x exp=0x%04x\n",
                           tile, label, r, 2 * w, got_lo, exp_lo);
                }
                mism++;
            }
            if (got_hi != exp_hi) {
                if (mism < 4) {
                    printf("  mismatch: tile=%u bank=%s row=%u lane=%u got=0x%04x exp=0x%04x\n",
                           tile, label, r, 2 * w + 1, got_hi, exp_hi);
                }
                mism++;
            }
        }
    }
    return mism;
}

/* ── Per-tile execution ────────────────────────────────────────────────── */

/* Returns 0 on clean (match or data-only mismatch), non-zero on unrecoverable
 * failure (non-ECALL halt, dbg0 wrong, timeout, IMEM readback fail). On
 * data-only mismatch `*tile_mism_out` is populated and caller continues. */
static int run_one_tile(uint32_t tile, int *tile_mism_out)
{
    *tile_mism_out = 0;

    seed_bank(SACL_OFF_A_B0, SAC_L_A[tile], 0);
    seed_bank(SACL_OFF_A_B1, SAC_L_A[tile], SACL_BANK_ROWS);
    seed_bank(SACL_OFF_B_B0, SAC_L_B[tile], 0);
    seed_bank(SACL_OFF_B_B1, SAC_L_B[tile], SACL_BANK_ROWS);
    asm volatile ("fence" ::: "memory");

    /* Phase 1: Atlas mul */
    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    asm volatile ("fence" ::: "memory");

    if (load_kernel(kernel_mul, KERNEL_MUL_LEN, "mul", tile)) {
        printf("FAIL(tile=%u): kernel_mul IMEM readback\n", tile);
        return 1;
    }

    /* Sample CSRs immediately before phase 1 kickoff and immediately
     * after phase 2 halt, so the delta measures only the two Atlas
     * phases + the host-side Saturn interlude between them. */
    uint32_t cycle_before = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t inst_before  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);

    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t status1 = 0;
    if (wait_halted(&status1)) {
        printf("FAIL(tile=%u): phase 1 did not halt (status=0x%08x)\n", tile, status1);
        return 1;
    }
    uint32_t reason1 = atlas_halt_reason();
    uint32_t dbg0_1  = atlas_dbg0();
    if (reason1 != ATLAS_HALT_ECALL || dbg0_1 != 1) {
        uint32_t ill = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_ILLEGAL_PC);
        printf("FAIL(tile=%u): phase 1 halt_reason=%u dbg0=%u illegal_pc=0x%08x\n",
               tile, reason1, dbg0_1, ill);
        return 1;
    }

    /* Saturn interlude */
    asm volatile ("fence" ::: "memory");
    run_interlude(SAC_L_SCALE[tile]);
    asm volatile ("fence" ::: "memory");

    /* Phase 2: Atlas reduce */
    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    asm volatile ("fence" ::: "memory");

    if (load_kernel(kernel_red, KERNEL_RED_LEN, "red", tile)) {
        printf("FAIL(tile=%u): kernel_red IMEM readback\n", tile);
        return 1;
    }

    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t status2 = 0;
    if (wait_halted(&status2)) {
        printf("FAIL(tile=%u): phase 2 did not halt (status=0x%08x)\n", tile, status2);
        return 1;
    }
    uint32_t reason2 = atlas_halt_reason();
    uint32_t dbg0_2  = atlas_dbg0();
    if (reason2 != ATLAS_HALT_ECALL || dbg0_2 != 1) {
        uint32_t ill = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_ILLEGAL_PC);
        printf("FAIL(tile=%u): phase 2 halt_reason=%u dbg0=%u illegal_pc=0x%08x\n",
               tile, reason2, dbg0_2, ill);
        return 1;
    }

    uint32_t cycle_after = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t inst_after  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);

    /* Verify */
    asm volatile ("fence" ::: "memory");
    int mism = 0;
    mism += verify_bank(SACL_OFF_S_B0, SAC_L_EXPECT_S[tile], 0,                tile, "S bank0");
    mism += verify_bank(SACL_OFF_S_B1, SAC_L_EXPECT_S[tile], SACL_BANK_ROWS,   tile, "S bank1");

    printf("  tile %2u: cycles=%u insts=%u lane_mismatches=%d\n",
           tile, cycle_after - cycle_before, inst_after - inst_before, mism);

    *tile_mism_out = mism;
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("Saturn <-> Atlas smolvla composition LARGE (N_TILES=%u, GOLDEN=%u)\n",
           (unsigned)SAC_L_N_TILES, (unsigned)SAC_L_GOLDEN_N_TILES);

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);
    asm volatile ("fence" ::: "memory");

    int tiles_with_mism = 0;
    int total_lane_mism = 0;

    for (uint32_t tile = 0; tile < (uint32_t)SAC_L_N_TILES; tile++) {
        int tile_mism = 0;
        int rc = run_one_tile(tile, &tile_mism);
        if (rc) {
            /* Unrecoverable — abort the whole test. */
            return 1;
        }
        if (tile_mism) {
            tiles_with_mism++;
            total_lane_mism += tile_mism;
        }
    }

    /* Raw end-of-test CSR totals (monotonic across all tiles). */
    uint32_t total_cycles = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t total_insts  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    printf("End-of-test CSR totals: CYCLE=%u INSTCNT=%u\n", total_cycles, total_insts);

    if (tiles_with_mism) {
        printf("FAIL: %d/%u tiles had data mismatches (%d lane-level total)\n",
               tiles_with_mism, (unsigned)SAC_L_N_TILES, total_lane_mism);
        return 1;
    }

    printf("PASS: saturn_atlas_smolvla_compose_large (N_TILES=%u)\n",
           (unsigned)SAC_L_N_TILES);
    return 0;
}

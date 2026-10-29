#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"
#include "saturn_atlas_tri_phase_large_golden.h"

/*
 * Saturn <-> Atlas tri-phase composition — multi-tile variant.
 *
 * Runs the same pipeline as saturn_atlas_tri_phase.c in a host-driven
 * outer loop over SAC_TRI_L_N_TILES distinct input tiles. Each iteration:
 *   (1) re-seed DRAM A,B slots
 *   (2) stage+run kernel_mul, verify phase 1 halt
 *   (3) Saturn RVV BF16 row-scale: Y -> Y'
 *   (4) stage kernel_red_jal (lower IMEM) + kernel_add (upper IMEM)
 *   (5) restart Atlas: kernel_red_jal -> JALR 0x4000 -> kernel_add -> ECALL
 *   (6) Saturn RVV BF16 post-scale: Z -> Z'' (into a fresh DRAM slot)
 *   (7) verify Z'' against SAC_TRI_L_EXPECT_Z2[tile]
 *
 * Tile count is SAC_TRI_L_N_TILES, default = SAC_TRI_L_GOLDEN_N_TILES (= 16
 * from the generated header). Override at compile time via
 * -DSAC_TRI_L_N_TILES=N (e.g. N=2 covers both edge-case tiles).
 * Tiles 0 and 1 are crafted BF16 edge cases (subnormal-heavy scale,
 * FP32-subnormal product); tiles 2..N-1 are randn throughput tiles
 * with tile 2 bit-identical to the single-tile test's seed=100 anchor.
 */

#ifndef SAC_TRI_L_N_TILES
#define SAC_TRI_L_N_TILES SAC_TRI_L_GOLDEN_N_TILES
#endif

_Static_assert(SAC_TRI_L_N_TILES >= 1,
               "SAC_TRI_L_N_TILES must be >= 1");
_Static_assert(SAC_TRI_L_N_TILES <= SAC_TRI_L_GOLDEN_N_TILES,
               "SAC_TRI_L_N_TILES must not exceed SAC_TRI_L_GOLDEN_N_TILES");

#define SAC_TRI_L_DRAM_BASE       ((uint64_t)ATLAS_DRAM_BASE)

/* DRAM layout, same slots as the single-tile test. The outer loop fully
 * overwrites these each tile. */
#define SAC_TRI_L_OFF_A_B0        0x0000U
#define SAC_TRI_L_OFF_A_B1        0x0400U
#define SAC_TRI_L_OFF_B_B0        0x0800U
#define SAC_TRI_L_OFF_B_B1        0x0C00U
#define SAC_TRI_L_OFF_Y_B0        0x1000U
#define SAC_TRI_L_OFF_Y_B1        0x1400U
#define SAC_TRI_L_OFF_S_B0        0x0800U
#define SAC_TRI_L_OFF_S_B1        0x0C00U
#define SAC_TRI_L_OFF_Z_B0        0x1000U
#define SAC_TRI_L_OFF_Z_B1        0x1400U
#define SAC_TRI_L_OFF_Z2_B0       0x1800U
#define SAC_TRI_L_OFF_Z2_B1       0x1C00U

#define SAC_TRI_L_BANK_ROWS       32U
#define SAC_TRI_L_ROW_BYTES       32U

#define SAC_TRI_L_IMEM_UPPER_WORD 0x4000U

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

/* NB: word 36 is a NOP (ADDI x0,x0,0) absorbing the delay slot after
 * JALR. Atlas PcControl has 1 architectural delay slot
 * (PcControl.scala:8,43-77); a JAL/JALR in a delay slot is flagged
 * illegal in ScalarCore.scala:226. See sab_a_to_high.S:25-28 for the
 * same convention on the Stage-2 toy kernels. */
#define KERNEL_RED_JAL_LEN 40
static const uint32_t kernel_red_jal[KERNEL_RED_JAL_LEN] = {
    0x00100E13U, 0x00000293U, 0x0002807FU, 0x40000113U, 0x00000313U, 0x900000B7U, 0x0020837BU, 0x0200007FU,
    0x10000313U, 0x900000B7U, 0x40008093U, 0x0020837BU, 0x0200007FU, 0x00000313U, 0x00030007U, 0x02801067U,
    0x00830087U, 0x02801067U, 0x42000257U, 0x09601067U, 0x20000413U, 0x00042207U, 0x02801067U, 0x00842287U,
    0x02801067U, 0x20000413U, 0x900011B7U, 0x80018193U, 0x022411FBU, 0x0200107FU, 0x30000413U, 0x900011B7U,
    0xC0018193U, 0x022411FBU, 0x0200107FU, 0x000040B7U, 0x00008067U, 0x00000013U, 0xC10E1073U, 0x00000073U,
};

#define KERNEL_ADD_LEN 54
static const uint32_t kernel_add[KERNEL_ADD_LEN] = {
    0x00100E13U, 0x00000293U, 0x0002807FU, 0x40000113U, 0x00000313U, 0x900000B7U, 0x0020837BU, 0x0200007FU,
    0x10000313U, 0x900000B7U, 0x40008093U, 0x0020837BU, 0x0200007FU, 0x00000313U, 0x00030007U, 0x02101067U,
    0x00830087U, 0x02101067U, 0x20000393U, 0x900010B7U, 0x80008093U, 0x002083FBU, 0x0200007FU, 0x30000393U,
    0x900010B7U, 0xC0008093U, 0x002083FBU, 0x0200007FU, 0x20000393U, 0x00038107U, 0x02101067U, 0x00838187U,
    0x02101067U, 0x00100257U, 0x04001067U, 0x40000413U, 0x00042207U, 0x02101067U, 0x00842287U, 0x02101067U,
    0x40000413U, 0x900011B7U, 0x022411FBU, 0x0200107FU, 0x50000413U, 0x900011B7U, 0x40018193U, 0x022411FBU,
    0x0200107FU, 0x00100093U, 0xC1009073U, 0x00000073U, 0xC10E1073U, 0x00000073U,
};

/* ── Helpers ───────────────────────────────────────────────────────────── */

static void seed_bank(uintptr_t dram_off, const uint16_t rows[][SAC_TRI_L_LANES], uint32_t first_row)
{
    uintptr_t base = SAC_TRI_L_DRAM_BASE + dram_off;
    for (uint32_t r = 0; r < SAC_TRI_L_BANK_ROWS; r++) {
        const uint16_t *row = rows[first_row + r];
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t val = ((uint32_t)row[2 * w + 1] << 16) | (uint32_t)row[2 * w];
            atlas_write32(base + r * SAC_TRI_L_ROW_BYTES + w * 4, val);
        }
    }
}

static int load_kernel(const uint32_t *words, uint32_t n, const char *label,
                       uint32_t word_offset, uint32_t tile)
{
    for (uint32_t i = 0; i < n; i++) {
        atlas_imem_write(word_offset + i, words[i]);
    }
    asm volatile ("fence" ::: "memory");
    int fail = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t got = atlas_imem_read(word_offset + i);
        if (got != words[i]) {
            printf("  IMEM MISMATCH[tile=%u %s word=%u]: got 0x%08x exp 0x%08x\n",
                   tile, label, word_offset + i, got, words[i]);
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

/* ── Saturn BF16 interlude (identical to single-tile) ──────────────────── */

static void saturn_interlude_row(uintptr_t in_addr, uintptr_t out_addr, uint16_t scale_bits)
{
    uint32_t scale_u32 = ((uint32_t)scale_bits) << 16;
    float scale_f;
    memcpy(&scale_f, &scale_u32, sizeof(float));

    size_t vl = __riscv_vsetvl_e16m1(SAC_TRI_L_LANES);
    vuint16m1_t in_u16 = __riscv_vle16_v_u16m1((const uint16_t *)in_addr, vl);
    vuint32m2_t in_u32 = __riscv_vzext_vf2_u32m2(in_u16, vl);
    vuint32m2_t in_bits = __riscv_vsll_vx_u32m2(in_u32, 16, vl);
    vfloat32m2_t in_f32 = __riscv_vreinterpret_v_u32m2_f32m2(in_bits);
    vfloat32m2_t p_f32  = __riscv_vfmul_vf_f32m2(in_f32, scale_f, vl);
    vuint32m2_t p_bits  = __riscv_vreinterpret_v_f32m2_u32m2(p_f32);
    vuint32m2_t p_hi    = __riscv_vsrl_vx_u32m2(p_bits, 16, vl);
    vuint16m1_t p_u16   = __riscv_vnsrl_wx_u16m1(p_hi, 0, vl);
    __riscv_vse16_v_u16m1((uint16_t *)out_addr, p_u16, vl);
}

static void run_interlude(uintptr_t in_b0, uintptr_t in_b1,
                          uintptr_t out_b0, uintptr_t out_b1,
                          const uint16_t scale[SAC_TRI_L_TILE_ROWS])
{
    const uintptr_t in_bases[2]  = {in_b0,  in_b1};
    const uintptr_t out_bases[2] = {out_b0, out_b1};
    for (uint32_t bank = 0; bank < 2; bank++) {
        for (uint32_t r = 0; r < SAC_TRI_L_BANK_ROWS; r++) {
            uintptr_t in_addr  = in_bases[bank]  + r * SAC_TRI_L_ROW_BYTES;
            uintptr_t out_addr = out_bases[bank] + r * SAC_TRI_L_ROW_BYTES;
            saturn_interlude_row(in_addr, out_addr, scale[r]);
        }
    }
}

/* ── Verification ──────────────────────────────────────────────────────── */

static int verify_bank(uintptr_t dram_off, const uint16_t expected[][SAC_TRI_L_LANES],
                       uint32_t first_row, uint32_t tile, const char *label)
{
    uintptr_t base = SAC_TRI_L_DRAM_BASE + dram_off;
    int mism = 0;
    for (uint32_t r = 0; r < SAC_TRI_L_BANK_ROWS; r++) {
        const uint16_t *exp = expected[first_row + r];
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t got = atlas_read32(base + r * SAC_TRI_L_ROW_BYTES + w * 4);
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

/* Returns 0 on clean (match or data-only Z'' mismatch), non-zero on
 * unrecoverable failure (IMEM readback, non-ECALL halt, dbg0 wrong,
 * timeout). On data-only mismatch `*tile_mism_out` is populated. */
static int run_one_tile(uint32_t tile, int *tile_mism_out)
{
    *tile_mism_out = 0;

    /* Seed A, B for phase 1. */
    seed_bank(SAC_TRI_L_OFF_A_B0, SAC_TRI_L_A[tile], 0);
    seed_bank(SAC_TRI_L_OFF_A_B1, SAC_TRI_L_A[tile], SAC_TRI_L_BANK_ROWS);
    seed_bank(SAC_TRI_L_OFF_B_B0, SAC_TRI_L_B[tile], 0);
    seed_bank(SAC_TRI_L_OFF_B_B1, SAC_TRI_L_B[tile], SAC_TRI_L_BANK_ROWS);
    asm volatile ("fence" ::: "memory");

    /* Phase 1: Atlas mul. */
    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    asm volatile ("fence" ::: "memory");

    if (load_kernel(kernel_mul, KERNEL_MUL_LEN, "mul", 0, tile)) {
        printf("FAIL(tile=%u): kernel_mul IMEM readback\n", tile);
        return 1;
    }

    /* Per-tile CSR deltas: sample before phase 1 kickoff, after
     * phase 2+3 halt. Deltas cover all Atlas work for this tile. */
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

    /* Saturn interlude 1: Y -> Y', write back to A slot. */
    asm volatile ("fence" ::: "memory");
    run_interlude(SAC_TRI_L_DRAM_BASE + SAC_TRI_L_OFF_Y_B0,
                  SAC_TRI_L_DRAM_BASE + SAC_TRI_L_OFF_Y_B1,
                  SAC_TRI_L_DRAM_BASE + SAC_TRI_L_OFF_A_B0,
                  SAC_TRI_L_DRAM_BASE + SAC_TRI_L_OFF_A_B1,
                  SAC_TRI_L_SCALE[tile]);
    asm volatile ("fence" ::: "memory");

    /* Phase 2+3: one Atlas run via JAL dispatch across IMEM halves. */
    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    asm volatile ("fence" ::: "memory");

    if (load_kernel(kernel_red_jal, KERNEL_RED_JAL_LEN, "red_jal", 0, tile)) {
        printf("FAIL(tile=%u): kernel_red_jal IMEM readback\n", tile);
        return 1;
    }
    if (load_kernel(kernel_add, KERNEL_ADD_LEN, "add", SAC_TRI_L_IMEM_UPPER_WORD, tile)) {
        printf("FAIL(tile=%u): kernel_add IMEM readback\n", tile);
        return 1;
    }
    asm volatile ("fence" ::: "memory");

    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t status23 = 0;
    if (wait_halted(&status23)) {
        printf("FAIL(tile=%u): phase 2+3 did not halt (status=0x%08x)\n", tile, status23);
        return 1;
    }
    uint32_t reason23 = atlas_halt_reason();
    uint32_t dbg0_23  = atlas_dbg0();
    if (reason23 != ATLAS_HALT_ECALL || dbg0_23 != 1) {
        uint32_t ill = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_ILLEGAL_PC);
        printf("FAIL(tile=%u): phase 2+3 halt_reason=%u dbg0=%u illegal_pc=0x%08x\n",
               tile, reason23, dbg0_23, ill);
        if (reason23 == ATLAS_HALT_ILLEGAL) {
            if (ill < SAC_TRI_L_IMEM_UPPER_WORD) {
                printf("       illegal-halt in lower half -> phase 2 (kernel_red_jal)\n");
            } else {
                printf("       illegal-halt in upper half -> phase 3 (kernel_add)\n");
            }
        }
        return 1;
    }

    uint32_t cycle_after = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t inst_after  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);

    /* Saturn interlude 2: Z -> Z'' in a fresh DRAM slot. */
    asm volatile ("fence" ::: "memory");
    run_interlude(SAC_TRI_L_DRAM_BASE + SAC_TRI_L_OFF_Z_B0,
                  SAC_TRI_L_DRAM_BASE + SAC_TRI_L_OFF_Z_B1,
                  SAC_TRI_L_DRAM_BASE + SAC_TRI_L_OFF_Z2_B0,
                  SAC_TRI_L_DRAM_BASE + SAC_TRI_L_OFF_Z2_B1,
                  SAC_TRI_L_POST_SCALE[tile]);
    asm volatile ("fence" ::: "memory");

    int mism = 0;
    mism += verify_bank(SAC_TRI_L_OFF_Z2_B0, SAC_TRI_L_EXPECT_Z2[tile], 0,                        tile, "Z'' bank0");
    mism += verify_bank(SAC_TRI_L_OFF_Z2_B1, SAC_TRI_L_EXPECT_Z2[tile], SAC_TRI_L_BANK_ROWS,      tile, "Z'' bank1");

    printf("  tile %2u: cycles=%u insts=%u lane_mismatches=%d\n",
           tile, cycle_after - cycle_before, inst_after - inst_before, mism);

    *tile_mism_out = mism;
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("Saturn <-> Atlas tri-phase composition LARGE (N_TILES=%u, GOLDEN=%u)\n",
           (unsigned)SAC_TRI_L_N_TILES, (unsigned)SAC_TRI_L_GOLDEN_N_TILES);

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);
    asm volatile ("fence" ::: "memory");

    int tiles_with_mism = 0;
    int total_lane_mism = 0;

    for (uint32_t tile = 0; tile < (uint32_t)SAC_TRI_L_N_TILES; tile++) {
        int tile_mism = 0;
        int rc = run_one_tile(tile, &tile_mism);
        if (rc) {
            return 1;       /* unrecoverable */
        }
        if (tile_mism) {
            tiles_with_mism++;
            total_lane_mism += tile_mism;
        }
    }

    uint32_t total_cycles = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t total_insts  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    printf("End-of-test CSR totals: CYCLE=%u INSTCNT=%u\n", total_cycles, total_insts);

    if (tiles_with_mism) {
        printf("FAIL: %d/%u tiles had data mismatches (%d lane-level total)\n",
               tiles_with_mism, (unsigned)SAC_TRI_L_N_TILES, total_lane_mism);
        return 1;
    }

    printf("PASS: saturn_atlas_tri_phase_large (N_TILES=%u)\n",
           (unsigned)SAC_TRI_L_N_TILES);
    return 0;
}

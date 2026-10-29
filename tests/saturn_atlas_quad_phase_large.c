#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"
#include "saturn_atlas_quad_phase_large_golden.h"

/*
 * Saturn <-> Atlas quad-phase pipeline (3 Atlas phases + 2 Saturn interludes).
 *
 * Per tile:
 *
 *   Phase A1 (Atlas, JAL chain across IMEM halves):
 *     lower (smolvla_matmul_mxu1_to_high) :  Y1   = A1_A_fp8 @ A1_B_fp8 (MXU1/IPT)
 *     upper (smolvla_reduction_sum_at_2000): SUM1 = row_reduce_broadcast(A1_X)
 *
 *   Interlude S1 (Saturn RVV BF16):
 *     Y1' = trunc_bf16(Y1 * SCALE1)   [per-row broadcast scale]
 *
 *   Phase A2 (Atlas, single kernel):
 *     smolvla_matmul_mxu0:  Y2 = A2_A_fp8 @ A2_B_fp8 (MXU0/SA)
 *
 *   Interlude S2 (Saturn RVV BF16):
 *     Y2' = trunc_bf16(Y2 * SCALE2)
 *
 *   Phase A3 (Atlas, JAL chain):
 *     lower (smolvla_elementwise_add_to_high):    Z3 = A3_A_add + A3_B_add (BF16)
 *     upper (smolvla_matmul_mxu1_at_2000):        Y3 = A3_A_mm @ A3_B_mm (MXU1/IPT)
 *
 * Per-tile counts: 3 ECALL boundaries, 2 in-IMEM JALR transitions, 2 Saturn
 * intervals. N_TILES = SAC_QUAD_GOLDEN_N_TILES (=8) by default.
 *
 * Both MXUs are exercised: MXU1 (IPT, single-round aligned tree) in A1 + A3,
 * MXU0 (SA, per-PE rounding) in A2.  VPU + DMA in supporting roles every
 * phase.  Saturn RVV BF16 row-broadcast scale between phases.
 *
 * Target config: EE290SimConfig.
 */

#ifndef SAC_QUAD_N_TILES
#define SAC_QUAD_N_TILES SAC_QUAD_GOLDEN_N_TILES
#endif

_Static_assert(SAC_QUAD_N_TILES >= 1,
               "SAC_QUAD_N_TILES must be >= 1");
_Static_assert(SAC_QUAD_N_TILES <= SAC_QUAD_GOLDEN_N_TILES,
               "SAC_QUAD_N_TILES must not exceed SAC_QUAD_GOLDEN_N_TILES");

#define SAC_QUAD_DRAM_BASE       ((uint64_t)ATLAS_DRAM_BASE)

/* ── DRAM layout (per tile) ──────────────────────────────────────────────
 *
 * Lower-half kernels use stock smolvla_* offsets (0x0000-0x17FF).
 * Upper-half kernels in JAL chains use offsets shifted by +0x2000 so they
 * don't trample the lower-half kernel's I/O (kernels are JAL-chained, no
 * host intervention between).
 *
 * Saturn interludes write to a fresh region (0x4000+) so their outputs
 * remain readable for verification across the next phase's DRAM re-stage.
 */

/* A1 lower (matmul_mxu1_to_high, native offsets) */
#define SAC_QUAD_OFF_A1_A_FP8    0x0000U  /* 1024 B FP8 (32 rows × 32 bytes) */
#define SAC_QUAD_OFF_A1_B_FP8    0x0400U  /* 1024 B FP8 (B^T) */
#define SAC_QUAD_OFF_Y1_B0       0x0800U  /* 1024 B BF16 (Y1 cols 0-15) */
#define SAC_QUAD_OFF_Y1_B1       0x0C00U  /* 1024 B BF16 (Y1 cols 16-31) */

/* A1 upper (reduction_sum_at_2000, +0x2000 offsets) */
#define SAC_QUAD_OFF_A1_X_B0     0x2000U  /* 1024 B BF16 (X bank0) */
#define SAC_QUAD_OFF_A1_X_B1     0x2400U  /* 1024 B BF16 (X bank1) */
#define SAC_QUAD_OFF_SUM1_B0     0x2800U  /* 1024 B BF16 (SUM1 bank0) */
#define SAC_QUAD_OFF_SUM1_B1     0x2C00U  /* 1024 B BF16 (SUM1 bank1) */

/* A2 (matmul_mxu0, native offsets — overwrites A1 lower's slots) */
#define SAC_QUAD_OFF_A2_A_FP8    0x0000U
#define SAC_QUAD_OFF_A2_B_FP8    0x0400U
#define SAC_QUAD_OFF_Y2_B0       0x0800U
#define SAC_QUAD_OFF_Y2_B1       0x0C00U

/* A3 lower (elementwise_add_to_high, native offsets — overwrites A2's slots) */
#define SAC_QUAD_OFF_A3_A_ADD_B0 0x0000U
#define SAC_QUAD_OFF_A3_A_ADD_B1 0x0400U
#define SAC_QUAD_OFF_A3_B_ADD_B0 0x0800U
#define SAC_QUAD_OFF_A3_B_ADD_B1 0x0C00U
#define SAC_QUAD_OFF_Z3_B0       0x1000U
#define SAC_QUAD_OFF_Z3_B1       0x1400U

/* A3 upper (matmul_mxu1_at_2000, +0x2000 offsets — overwrites A1 upper's slots) */
#define SAC_QUAD_OFF_A3_A_MM_FP8 0x2000U
#define SAC_QUAD_OFF_A3_B_MM_FP8 0x2400U
#define SAC_QUAD_OFF_Y3_B0       0x2800U
#define SAC_QUAD_OFF_Y3_B1       0x2C00U

/* Saturn output slots (preserved across Atlas phases) */
#define SAC_QUAD_OFF_Y1P_B0      0x4000U
#define SAC_QUAD_OFF_Y1P_B1      0x4400U
#define SAC_QUAD_OFF_Y2P_B0      0x4800U
#define SAC_QUAD_OFF_Y2P_B1      0x4C00U

#define SAC_QUAD_BANK_ROWS       32U
#define SAC_QUAD_BF16_ROW_BYTES  32U      /* 16 lanes × 2 bytes */
#define SAC_QUAD_FP8_ROW_BYTES   32U      /* 32 lanes × 1 byte  */
#define SAC_QUAD_FP8_BANK_BYTES  1024U    /* 32 rows × 32 bytes */

#define SAC_QUAD_IMEM_UPPER_WORD 0x4000U  /* IMEM byte 0x10000 = word 0x4000 */

#define POLL_ITERS               200000000U

/* ── Embedded Atlas kernel words ─────────────────────────────────────────
 * Assembled from:
 *   generators/atlas-npu/baremetal/assembly/{name}.S
 * via assembler.py. The two _to_high lower-half kernels end with a
 * word-indexed JALR to byte 0x10000 (word 0x4000) so the upper-half
 * kernel runs without halting. The upper-half kernels and the single
 * mxu0 kernel end with the standard pass-path:
 *   ADDI x1,x0,1 ; CSRRW x0,0xC10,x1 ; ECALL  (dbg0=1, halt_reason=ECALL).
 */

#define KERNEL_MM1_TO_HIGH_LEN 48
static const uint32_t kernel_mm1_to_high[KERNEL_MM1_TO_HIGH_LEN] = {
    0x00100E13U, 0x00000293U, 0x0002807FU, 0x40000113U, 0x00000313U, 0x900000B7U, 0x40008093U, 0x0020837BU,
    0x0200007FU, 0x00000313U, 0x00030207U, 0x02801067U, 0x10000393U, 0x900000B7U, 0x002083FBU, 0x0200007FU,
    0x10000393U, 0x00038007U, 0x02801067U, 0x000082EBU, 0x05001067U, 0x0200A077U, 0x02801067U, 0x16000077U,
    0x07801067U, 0x12000177U, 0x02801067U, 0x20000413U, 0x00042107U, 0x02801067U, 0x30000493U, 0x0004A187U,
    0x02801067U, 0x20000413U, 0x900011B7U, 0x80018193U, 0x022411FBU, 0x0200107FU, 0x30000493U, 0x900011B7U,
    0xC0018193U, 0x022491FBU, 0x0200107FU, 0x000040B7U, 0x00008067U, 0x00000013U, 0xC10E1073U, 0x00000073U,
};

#define KERNEL_REDSUM_AT_2000_LEN 40
static const uint32_t kernel_redsum_at_2000[KERNEL_REDSUM_AT_2000_LEN] = {
    0x00100E13U, 0x00000293U, 0x0002807FU, 0x40000113U, 0x00000313U, 0x900020B7U, 0x0020837BU, 0x0200007FU,
    0x10000313U, 0x900020B7U, 0x40008093U, 0x0020837BU, 0x0200007FU, 0x00000313U, 0x00030007U, 0x02801067U,
    0x00830087U, 0x02801067U, 0x42000257U, 0x09601067U, 0x20000413U, 0x00042207U, 0x02801067U, 0x00842287U,
    0x02801067U, 0x20000413U, 0x900031B7U, 0x80018193U, 0x022411FBU, 0x0200107FU, 0x30000413U, 0x900031B7U,
    0xC0018193U, 0x022411FBU, 0x0200107FU, 0x00100093U, 0xC1009073U, 0x00000073U, 0xC10E1073U, 0x00000073U,
};

#define KERNEL_MM0_LEN 48
static const uint32_t kernel_mm0[KERNEL_MM0_LEN] = {
    0x00100E13U, 0x00000293U, 0x0002807FU, 0x40000113U, 0x00000313U, 0x900000B7U, 0x40008093U, 0x0020837BU,
    0x0200007FU, 0x00000313U, 0x00030207U, 0x02101067U, 0x10000393U, 0x900000B7U, 0x002083FBU, 0x0200007FU,
    0x10000393U, 0x00038007U, 0x02101067U, 0x000082EBU, 0x04001067U, 0x0000A077U, 0x01F01067U, 0x14000077U,
    0x05F01067U, 0x10000177U, 0x01F01067U, 0x20000413U, 0x00042107U, 0x02101067U, 0x30000493U, 0x0004A187U,
    0x02101067U, 0x20000413U, 0x900011B7U, 0x80018193U, 0x022411FBU, 0x0200107FU, 0x30000493U, 0x900011B7U,
    0xC0018193U, 0x022491FBU, 0x0200107FU, 0x00100093U, 0xC1009073U, 0x00000073U, 0xC10E1073U, 0x00000073U,
};

#define KERNEL_ADD_TO_HIGH_LEN 54
static const uint32_t kernel_add_to_high[KERNEL_ADD_TO_HIGH_LEN] = {
    0x00100E13U, 0x00000293U, 0x0002807FU, 0x40000113U, 0x00000313U, 0x900000B7U, 0x0020837BU, 0x0200007FU,
    0x10000313U, 0x900000B7U, 0x40008093U, 0x0020837BU, 0x0200007FU, 0x00000313U, 0x00030007U, 0x02801067U,
    0x00830087U, 0x02801067U, 0x20000393U, 0x900010B7U, 0x80008093U, 0x002083FBU, 0x0200007FU, 0x30000393U,
    0x900010B7U, 0xC0008093U, 0x002083FBU, 0x0200007FU, 0x20000393U, 0x00038107U, 0x02801067U, 0x00838187U,
    0x02801067U, 0x00100257U, 0x09601067U, 0x40000413U, 0x00042207U, 0x02801067U, 0x00842287U, 0x02801067U,
    0x40000413U, 0x900011B7U, 0x022411FBU, 0x0200107FU, 0x50000413U, 0x900011B7U, 0x40018193U, 0x022411FBU,
    0x0200107FU, 0x000040B7U, 0x00008067U, 0x00000013U, 0xC10E1073U, 0x00000073U,
};

#define KERNEL_MM1_AT_2000_LEN 48
static const uint32_t kernel_mm1_at_2000[KERNEL_MM1_AT_2000_LEN] = {
    0x00100E13U, 0x00000293U, 0x0002807FU, 0x40000113U, 0x00000313U, 0x900020B7U, 0x40008093U, 0x0020837BU,
    0x0200007FU, 0x00000313U, 0x00030207U, 0x02801067U, 0x10000393U, 0x900020B7U, 0x002083FBU, 0x0200007FU,
    0x10000393U, 0x00038007U, 0x02801067U, 0x000082EBU, 0x05001067U, 0x0200A077U, 0x02801067U, 0x16000077U,
    0x07801067U, 0x12000177U, 0x02801067U, 0x20000413U, 0x00042107U, 0x02801067U, 0x30000493U, 0x0004A187U,
    0x02801067U, 0x20000413U, 0x900031B7U, 0x80018193U, 0x022411FBU, 0x0200107FU, 0x30000493U, 0x900031B7U,
    0xC0018193U, 0x022491FBU, 0x0200107FU, 0x00100093U, 0xC1009073U, 0x00000073U, 0xC10E1073U, 0x00000073U,
};

/* ── Helpers ────────────────────────────────────────────────────────────── */

static void seed_bf16_bank(uintptr_t dram_off,
                           const uint16_t rows[][SAC_QUAD_LANES],
                           uint32_t first_row)
{
    uintptr_t base = SAC_QUAD_DRAM_BASE + dram_off;
    for (uint32_t r = 0; r < SAC_QUAD_BANK_ROWS; r++) {
        const uint16_t *row = rows[first_row + r];
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t val = ((uint32_t)row[2 * w + 1] << 16) | (uint32_t)row[2 * w];
            atlas_write32(base + r * SAC_QUAD_BF16_ROW_BYTES + w * 4, val);
        }
    }
}

/* FP8 tile is 32 rows × 32 bytes = 1024 bytes contiguous. */
static void seed_fp8_tile(uintptr_t dram_off,
                          const uint8_t tile[SAC_QUAD_TILE_ROWS][SAC_QUAD_TILE_COLS])
{
    uintptr_t base = SAC_QUAD_DRAM_BASE + dram_off;
    for (uint32_t r = 0; r < SAC_QUAD_TILE_ROWS; r++) {
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t val = ((uint32_t)tile[r][4 * w + 0])       |
                           ((uint32_t)tile[r][4 * w + 1] << 8)  |
                           ((uint32_t)tile[r][4 * w + 2] << 16) |
                           ((uint32_t)tile[r][4 * w + 3] << 24);
            atlas_write32(base + r * SAC_QUAD_FP8_ROW_BYTES + w * 4, val);
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

/* ── Saturn BF16 row-broadcast scale (matches tri_phase_large semantics) ── */

static void saturn_interlude_row(uintptr_t in_addr, uintptr_t out_addr, uint16_t scale_bits)
{
    uint32_t scale_u32 = ((uint32_t)scale_bits) << 16;
    float scale_f;
    memcpy(&scale_f, &scale_u32, sizeof(float));

    size_t vl = __riscv_vsetvl_e16m1(SAC_QUAD_LANES);
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
                          const uint16_t scale[SAC_QUAD_TILE_ROWS])
{
    const uintptr_t in_bases[2]  = {in_b0,  in_b1};
    const uintptr_t out_bases[2] = {out_b0, out_b1};
    for (uint32_t bank = 0; bank < 2; bank++) {
        for (uint32_t r = 0; r < SAC_QUAD_BANK_ROWS; r++) {
            uintptr_t in_addr  = in_bases[bank]  + r * SAC_QUAD_BF16_ROW_BYTES;
            uintptr_t out_addr = out_bases[bank] + r * SAC_QUAD_BF16_ROW_BYTES;
            saturn_interlude_row(in_addr, out_addr, scale[r]);
        }
    }
}

/* ── Verification ──────────────────────────────────────────────────────── */

static int verify_bank(uintptr_t dram_off,
                       const uint16_t expected[][SAC_QUAD_LANES],
                       uint32_t first_row,
                       uint32_t tile,
                       const char *label)
{
    uintptr_t base = SAC_QUAD_DRAM_BASE + dram_off;
    int mism = 0;
    for (uint32_t r = 0; r < SAC_QUAD_BANK_ROWS; r++) {
        const uint16_t *exp = expected[first_row + r];
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t got = atlas_read32(base + r * SAC_QUAD_BF16_ROW_BYTES + w * 4);
            uint16_t got_lo = (uint16_t)(got & 0xFFFF);
            uint16_t got_hi = (uint16_t)(got >> 16);
            uint16_t exp_lo = exp[2 * w];
            uint16_t exp_hi = exp[2 * w + 1];
            if (got_lo != exp_lo) {
                if (mism < 4) {
                    printf("  mismatch: tile=%u %s row=%u lane=%u got=0x%04x exp=0x%04x\n",
                           tile, label, r, 2 * w, got_lo, exp_lo);
                }
                mism++;
            }
            if (got_hi != exp_hi) {
                if (mism < 4) {
                    printf("  mismatch: tile=%u %s row=%u lane=%u got=0x%04x exp=0x%04x\n",
                           tile, label, r, 2 * w + 1, got_hi, exp_hi);
                }
                mism++;
            }
        }
    }
    return mism;
}

/* ── Per-tile execution ────────────────────────────────────────────────── */

static int run_phase(uint32_t tile,
                     const char *phase_label,
                     const uint32_t *lower_words, uint32_t lower_len, const char *lower_label,
                     const uint32_t *upper_words, uint32_t upper_len, const char *upper_label,
                     uint32_t *cycles_out, uint32_t *insts_out)
{
    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    asm volatile ("fence" ::: "memory");

    if (load_kernel(lower_words, lower_len, lower_label, 0, tile)) {
        printf("FAIL(tile=%u %s): %s IMEM readback\n", tile, phase_label, lower_label);
        return 1;
    }
    if (upper_words && load_kernel(upper_words, upper_len, upper_label,
                                   SAC_QUAD_IMEM_UPPER_WORD, tile)) {
        printf("FAIL(tile=%u %s): %s IMEM readback\n", tile, phase_label, upper_label);
        return 1;
    }
    asm volatile ("fence" ::: "memory");

    uint32_t cycle_before = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t inst_before  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);

    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t status = 0;
    if (wait_halted(&status)) {
        printf("FAIL(tile=%u %s): did not halt (status=0x%08x)\n", tile, phase_label, status);
        return 1;
    }
    uint32_t reason = atlas_halt_reason();
    uint32_t dbg0   = atlas_dbg0();
    if (reason != ATLAS_HALT_ECALL || dbg0 != 1) {
        uint32_t ill = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_ILLEGAL_PC);
        printf("FAIL(tile=%u %s): halt_reason=%u dbg0=%u illegal_pc=0x%08x\n",
               tile, phase_label, reason, dbg0, ill);
        if (reason == ATLAS_HALT_ILLEGAL && upper_words) {
            if (ill < SAC_QUAD_IMEM_UPPER_WORD) {
                printf("       illegal-halt in lower half (%s)\n", lower_label);
            } else {
                printf("       illegal-halt in upper half (%s)\n", upper_label);
            }
        }
        return 1;
    }

    *cycles_out = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE) - cycle_before;
    *insts_out  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT) - inst_before;
    return 0;
}

static int run_one_tile(uint32_t tile, int *tile_mism_out)
{
    *tile_mism_out = 0;

    /* ───── Phase A1: stage inputs (FP8 A,B for matmul + BF16 X for reduce) ───── */
    seed_fp8_tile (SAC_QUAD_OFF_A1_A_FP8, SAC_QUAD_A1_A_FP8[tile]);
    seed_fp8_tile (SAC_QUAD_OFF_A1_B_FP8, SAC_QUAD_A1_B_FP8[tile]);
    seed_bf16_bank(SAC_QUAD_OFF_A1_X_B0,  SAC_QUAD_A1_X_BF16[tile], 0);
    seed_bf16_bank(SAC_QUAD_OFF_A1_X_B1,  SAC_QUAD_A1_X_BF16[tile], SAC_QUAD_BANK_ROWS);
    asm volatile ("fence" ::: "memory");

    uint32_t a1_cyc = 0, a1_ins = 0;
    if (run_phase(tile, "A1",
                  kernel_mm1_to_high,    KERNEL_MM1_TO_HIGH_LEN,    "mm1_to_high",
                  kernel_redsum_at_2000, KERNEL_REDSUM_AT_2000_LEN, "redsum_at_2000",
                  &a1_cyc, &a1_ins)) {
        return 1;
    }

    int mism = 0;
    asm volatile ("fence" ::: "memory");
    mism += verify_bank(SAC_QUAD_OFF_Y1_B0,   SAC_QUAD_EXPECT_Y1[tile],   0,                    tile, "Y1 b0");
    mism += verify_bank(SAC_QUAD_OFF_Y1_B1,   SAC_QUAD_EXPECT_Y1[tile],   SAC_QUAD_BANK_ROWS,   tile, "Y1 b1");
    mism += verify_bank(SAC_QUAD_OFF_SUM1_B0, SAC_QUAD_EXPECT_SUM1[tile], 0,                    tile, "SUM1 b0");
    mism += verify_bank(SAC_QUAD_OFF_SUM1_B1, SAC_QUAD_EXPECT_SUM1[tile], SAC_QUAD_BANK_ROWS,   tile, "SUM1 b1");

    /* ───── Saturn S1: Y1 -> Y1' at 0x4000 ───── */
    asm volatile ("fence" ::: "memory");
    run_interlude(SAC_QUAD_DRAM_BASE + SAC_QUAD_OFF_Y1_B0,
                  SAC_QUAD_DRAM_BASE + SAC_QUAD_OFF_Y1_B1,
                  SAC_QUAD_DRAM_BASE + SAC_QUAD_OFF_Y1P_B0,
                  SAC_QUAD_DRAM_BASE + SAC_QUAD_OFF_Y1P_B1,
                  SAC_QUAD_SCALE1[tile]);
    asm volatile ("fence" ::: "memory");
    mism += verify_bank(SAC_QUAD_OFF_Y1P_B0, SAC_QUAD_EXPECT_Y1P[tile], 0,                  tile, "Y1' b0");
    mism += verify_bank(SAC_QUAD_OFF_Y1P_B1, SAC_QUAD_EXPECT_Y1P[tile], SAC_QUAD_BANK_ROWS, tile, "Y1' b1");

    /* ───── Phase A2: stage inputs (FP8 A,B for MXU0 matmul) ───── */
    seed_fp8_tile(SAC_QUAD_OFF_A2_A_FP8, SAC_QUAD_A2_A_FP8[tile]);
    seed_fp8_tile(SAC_QUAD_OFF_A2_B_FP8, SAC_QUAD_A2_B_FP8[tile]);
    asm volatile ("fence" ::: "memory");

    uint32_t a2_cyc = 0, a2_ins = 0;
    if (run_phase(tile, "A2",
                  kernel_mm0, KERNEL_MM0_LEN, "mm0",
                  NULL, 0, NULL,
                  &a2_cyc, &a2_ins)) {
        return 1;
    }

    asm volatile ("fence" ::: "memory");
    mism += verify_bank(SAC_QUAD_OFF_Y2_B0, SAC_QUAD_EXPECT_Y2[tile], 0,                  tile, "Y2 b0");
    mism += verify_bank(SAC_QUAD_OFF_Y2_B1, SAC_QUAD_EXPECT_Y2[tile], SAC_QUAD_BANK_ROWS, tile, "Y2 b1");

    /* ───── Saturn S2: Y2 -> Y2' at 0x4800 ───── */
    asm volatile ("fence" ::: "memory");
    run_interlude(SAC_QUAD_DRAM_BASE + SAC_QUAD_OFF_Y2_B0,
                  SAC_QUAD_DRAM_BASE + SAC_QUAD_OFF_Y2_B1,
                  SAC_QUAD_DRAM_BASE + SAC_QUAD_OFF_Y2P_B0,
                  SAC_QUAD_DRAM_BASE + SAC_QUAD_OFF_Y2P_B1,
                  SAC_QUAD_SCALE2[tile]);
    asm volatile ("fence" ::: "memory");
    mism += verify_bank(SAC_QUAD_OFF_Y2P_B0, SAC_QUAD_EXPECT_Y2P[tile], 0,                  tile, "Y2' b0");
    mism += verify_bank(SAC_QUAD_OFF_Y2P_B1, SAC_QUAD_EXPECT_Y2P[tile], SAC_QUAD_BANK_ROWS, tile, "Y2' b1");

    /* ───── Phase A3: stage inputs (BF16 add A,B + FP8 matmul A,B) ───── */
    seed_bf16_bank(SAC_QUAD_OFF_A3_A_ADD_B0, SAC_QUAD_A3_A_ADD_BF16[tile], 0);
    seed_bf16_bank(SAC_QUAD_OFF_A3_A_ADD_B1, SAC_QUAD_A3_A_ADD_BF16[tile], SAC_QUAD_BANK_ROWS);
    seed_bf16_bank(SAC_QUAD_OFF_A3_B_ADD_B0, SAC_QUAD_A3_B_ADD_BF16[tile], 0);
    seed_bf16_bank(SAC_QUAD_OFF_A3_B_ADD_B1, SAC_QUAD_A3_B_ADD_BF16[tile], SAC_QUAD_BANK_ROWS);
    seed_fp8_tile (SAC_QUAD_OFF_A3_A_MM_FP8, SAC_QUAD_A3_A_MM_FP8[tile]);
    seed_fp8_tile (SAC_QUAD_OFF_A3_B_MM_FP8, SAC_QUAD_A3_B_MM_FP8[tile]);
    asm volatile ("fence" ::: "memory");

    uint32_t a3_cyc = 0, a3_ins = 0;
    if (run_phase(tile, "A3",
                  kernel_add_to_high, KERNEL_ADD_TO_HIGH_LEN, "add_to_high",
                  kernel_mm1_at_2000, KERNEL_MM1_AT_2000_LEN, "mm1_at_2000",
                  &a3_cyc, &a3_ins)) {
        return 1;
    }

    asm volatile ("fence" ::: "memory");
    mism += verify_bank(SAC_QUAD_OFF_Z3_B0, SAC_QUAD_EXPECT_Z3[tile], 0,                  tile, "Z3 b0");
    mism += verify_bank(SAC_QUAD_OFF_Z3_B1, SAC_QUAD_EXPECT_Z3[tile], SAC_QUAD_BANK_ROWS, tile, "Z3 b1");
    mism += verify_bank(SAC_QUAD_OFF_Y3_B0, SAC_QUAD_EXPECT_Y3[tile], 0,                  tile, "Y3 b0");
    mism += verify_bank(SAC_QUAD_OFF_Y3_B1, SAC_QUAD_EXPECT_Y3[tile], SAC_QUAD_BANK_ROWS, tile, "Y3 b1");

    printf("  tile %2u: A1 cyc=%u ins=%u | A2 cyc=%u ins=%u | A3 cyc=%u ins=%u | mism=%d\n",
           tile, a1_cyc, a1_ins, a2_cyc, a2_ins, a3_cyc, a3_ins, mism);

    *tile_mism_out = mism;
    return 0;
}

/* ── Main ──────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("Saturn <-> Atlas quad-phase pipeline LARGE (N_TILES=%u, GOLDEN=%u)\n",
           (unsigned)SAC_QUAD_N_TILES, (unsigned)SAC_QUAD_GOLDEN_N_TILES);

    /* Pin frm = RNE for any Saturn-side BF16 narrow that depends on rounding. */
    asm volatile ("csrwi frm, 0" ::: "memory");

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);
    asm volatile ("fence" ::: "memory");

    int tiles_with_mism = 0;
    int total_lane_mism = 0;

    for (uint32_t tile = 0; tile < (uint32_t)SAC_QUAD_N_TILES; tile++) {
        int tile_mism = 0;
        int rc = run_one_tile(tile, &tile_mism);
        if (rc) {
            return 1;
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
               tiles_with_mism, (unsigned)SAC_QUAD_N_TILES, total_lane_mism);
        return 1;
    }

    printf("PASS: saturn_atlas_quad_phase_large (N_TILES=%u)\n",
           (unsigned)SAC_QUAD_N_TILES);
    return 0;
}

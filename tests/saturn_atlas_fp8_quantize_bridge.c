#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "atlas_mmio.h"
#include "saturn_rvv_mx.h"
#include "saturn_atlas_fp8_quantize_bridge_golden.h"

/*
 * Saturn -> Atlas FP8 quantize bridge test.
 *
 * One-way handshake: Saturn quantizes a BF16 activation tile to E4M3
 * bytes via native vfncvtbf16.f.f.w (altfmt=E4M3), and Atlas's MXU1
 * matmul kernel consumes those exact bytes alongside a host-preloaded
 * FP8 weight tile and a BF16 bias.
 *
 *   stage 0  host:   seed A_bf16 (Saturn src), B_fp8 (host pre-quantized),
 *                    bias_bf16 in DRAM
 *   stage 1  Saturn: native BF16->E4M3 quantize of A_bf16 -> A_fp8
 *
 *   ── checkpoint 1 (Saturn quantizer alignment) ──
 *     compare Saturn-produced A_fp8 bytes against bf16_tile_to_e4m3_bytes
 *     golden (= Atlas's own VMATPUSH.ACC.BF16 + SELI 0x7F + VMATPOP.FP8
 *     roundtrip). Failure here = real Saturn↔Atlas FP8-format finding
 *     for normal-finite inputs.
 *
 *   stage 2  Atlas:  smolvla_fused_matmul_bias_v2_mxu1 (FP8 A + FP8 B
 *                    + BF16 bias -> BF16 out)
 *
 *   ── checkpoint 2 (end-to-end correctness) ──
 *     compare Atlas BF16 output against IPTLinearRTLFunction +
 *     run_binary_rows("add", ...) chain through software_models.
 *     Catches matmul/bias-add bugs separately from encoding bugs.
 *
 * Atlas operates on a perfect-software assumption: no Inf/NaN/subnormal
 * inputs. The generator further restricts the Saturn-quantized A_bf16
 * tile to |x| in [0.25, 2.0) because Atlas's bf16_tile_to_e4m3_bytes
 * flushes near-zero values to +/-0 while Saturn emits E4M3 subnormals
 * for the same inputs (real cross-engine encoding divergence outside
 * the well-defined domain). B_fp8 is host-prequantized through the
 * same Atlas helper so its goldens stay byte-aligned.
 *
 * Target config: EE290SimConfig.
 */

#define SAC_BR_DRAM_BASE     ((uint64_t)ATLAS_DRAM_BASE)

/* DRAM layout (matches smolvla_fused_matmul_bias_v2_mxu1.S kernel + a
 * Saturn input slot at 0x2000): */
#define SAC_BR_OFF_A_FP8     0x0000U   /* 1024 B,  Saturn writes (= Atlas A) */
#define SAC_BR_OFF_B_FP8     0x0400U   /* 1024 B,  host seeds   (= Atlas B) */
#define SAC_BR_OFF_BIAS      0x0800U   /* 2048 B,  host seeds   (Atlas bias) */
#define SAC_BR_OFF_OUT       0x1000U   /* 2048 B,  Atlas writes (verified) */
#define SAC_BR_OFF_A_BF16    0x2000U   /* 2048 B,  host seeds   (Saturn src) */

#define SAC_BR_BF16_ROW_BYTES    64U   /* 32 BF16 lanes */
#define SAC_BR_FP8_ROW_BYTES     32U   /* 32 E4M3 bytes */
#define SAC_BR_BANK_ROW_BYTES    32U   /* 16 BF16 lanes (bank-split row) */

#define POLL_ITERS               200000000U

/* ── Atlas kernel words ───────────────────────────────────────────────
 * Assembled from generators/atlas-npu/baremetal/assembly/
 * smolvla_fused_matmul_bias_v2_mxu1.S via:
 *   python3 generators/atlas-npu/baremetal/assembler.py \
 *     generators/atlas-npu/baremetal/assembly/smolvla_fused_matmul_bias_v2_mxu1.S \
 *     --out-hex /tmp/v2_mxu1.hex
 * Pass marker: CSRRW x0, 0xC10, 1; ECALL (dbg0=1, halt_reason=ECALL).
 */
#define KERNEL_LEN 62
static const uint32_t kernel_words[KERNEL_LEN] = {
    0x00100E13U, 0x00000293U, 0x0002807FU, 0x40000113U, 0x00000313U, 0x900000B7U, 0x0020837BU, 0x0200007FU,
    0x00000313U, 0x00030007U, 0x02101067U, 0x10000393U, 0x900000B7U, 0x40008093U, 0x002083FBU, 0x0200007FU,
    0x10000393U, 0x00038207U, 0x02101067U, 0x20000493U, 0x900010B7U, 0x80008093U, 0x002084FBU, 0x0200007FU,
    0x30000493U, 0x900010B7U, 0xC0008093U, 0x002084FBU, 0x0200007FU, 0x20000493U, 0x00048307U, 0x02101067U,
    0x00848387U, 0x02101067U, 0x000082EBU, 0x04001067U, 0x0200A077U, 0x01F01067U, 0x16000077U, 0x02201067U,
    0x12000577U, 0x01F01067U, 0x00314657U, 0x04001067U, 0x40000413U, 0x00042607U, 0x02101067U, 0x00842687U,
    0x02101067U, 0x40000413U, 0x900011B7U, 0x022411FBU, 0x0200107FU, 0x50000413U, 0x900011B7U, 0x40018193U,
    0x022411FBU, 0x0200107FU, 0x00100093U, 0xC1009073U, 0x00000073U, 0xC10E1073U, 0x00000073U,
};

/* ── DRAM seed helpers ───────────────────────────────────────────────── */

static void seed_a_bf16(void)
{
    uintptr_t base = SAC_BR_DRAM_BASE + SAC_BR_OFF_A_BF16;
    for (uint32_t r = 0; r < SAC_BR_TILE_ROWS; r++) {
        for (uint32_t w = 0; w < 16; w++) {
            uint32_t v = ((uint32_t)SAC_BR_A_BF16[r][2 * w + 1] << 16) |
                         (uint32_t)SAC_BR_A_BF16[r][2 * w];
            atlas_write32(base + r * SAC_BR_BF16_ROW_BYTES + w * 4, v);
        }
    }
}

static void seed_b_fp8(void)
{
    uintptr_t base = SAC_BR_DRAM_BASE + SAC_BR_OFF_B_FP8;
    for (uint32_t r = 0; r < SAC_BR_TILE_ROWS; r++) {
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t v = ((uint32_t)SAC_BR_B_FP8[r][4 * w + 3] << 24) |
                         ((uint32_t)SAC_BR_B_FP8[r][4 * w + 2] << 16) |
                         ((uint32_t)SAC_BR_B_FP8[r][4 * w + 1] <<  8) |
                         (uint32_t)SAC_BR_B_FP8[r][4 * w];
            atlas_write32(base + r * SAC_BR_FP8_ROW_BYTES + w * 4, v);
        }
    }
}

static void seed_bias(void)
{
    /* Bank-split: SAC_BR_BIAS[64][16] BF16 (bank0 rows 0..31, bank1 rows 0..31). */
    uintptr_t base = SAC_BR_DRAM_BASE + SAC_BR_OFF_BIAS;
    for (uint32_t r = 0; r < SAC_BR_TOTAL_BANK_ROWS; r++) {
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t v = ((uint32_t)SAC_BR_BIAS[r][2 * w + 1] << 16) |
                         (uint32_t)SAC_BR_BIAS[r][2 * w];
            atlas_write32(base + r * SAC_BR_BANK_ROW_BYTES + w * 4, v);
        }
    }
}

/* ── Atlas kernel staging / control ─────────────────────────────────── */

static int load_kernel(const uint32_t *words, uint32_t n, const char *label)
{
    for (uint32_t i = 0; i < n; i++) {
        atlas_imem_write(i, words[i]);
    }
    asm volatile ("fence" ::: "memory");
    int fail = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t got = atlas_imem_read(i);
        if (got != words[i]) {
            printf("  IMEM MISMATCH[%s word %u]: got 0x%08x exp 0x%08x\n",
                   label, i, got, words[i]);
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

/* ── Saturn native BF16 -> E4M3 quantize ───────────────────────────────
 *
 * Per row (32 BF16 lanes -> 32 E4M3 bytes):
 *   1. vsetvli  SEW=e16 LMUL=m2 alt=0   (load 32 BF16 lanes; 2x op width)
 *   2. vle16.v  v0..v1                  (BF16 input)
 *   3. vsetvli  SEW=e8  LMUL=m1 alt=0   (E4M3 op SEW; alt=0 selects E4M3
 *                                        per Saturn vec-mx-unary line 85)
 *   4. VFNCVTBF16_F_F_W                 (narrow BF16 -> E4M3, RNE)
 *   5. vsetvli  SEW=e8  LMUL=m1 alt=0   (E4M3 store SEW; redundant but
 *                                        explicit per Saturn benchmark
 *                                        store pattern)
 *   6. vse8.v   v24                     (32 E4M3 bytes out)
 *
 * Naming note: Saturn's `VFNCVTBF16_F_F_W` macro is the generic narrow —
 * the actual format produced is selected by SEW+altfmt at the time of
 * issue. At SEW=e8 alt=0 it produces E4M3.
 */
static void saturn_quantize_row(uintptr_t in_bf16_addr, uintptr_t out_fp8_addr)
{
    size_t vl;

    VSETVLI_ALTFMT(vl, SAC_BR_TILE_COLS, SEW_E16, LMUL_M2, 0);
    asm volatile ("vle16.v v0, (%0)" : : "r"(in_bf16_addr)
                  : "v0", "v1", "memory");

    VSETVLI_ALTFMT_X0(vl, SEW_E8, LMUL_M1, 0);
    VFNCVTBF16_F_F_W(V24, V0);

    VSETVLI_ALTFMT_X0(vl, SEW_E8, LMUL_M1, 0);
    asm volatile ("vse8.v v24, (%0)" : : "r"(out_fp8_addr)
                  : "v24", "memory");
}

static void run_saturn_quantize(void)
{
    /* Pin frm=RNE before native BF16->E4M3 narrow; the golden was
     * generated via bf16_tile_to_e4m3_bytes which assumes RNE narrow.
     * Don't depend on ambient reset state. */
    SATURN_SET_FRM_RNE();

    for (uint32_t r = 0; r < SAC_BR_TILE_ROWS; r++) {
        uintptr_t in  = SAC_BR_DRAM_BASE + SAC_BR_OFF_A_BF16 + r * SAC_BR_BF16_ROW_BYTES;
        uintptr_t out = SAC_BR_DRAM_BASE + SAC_BR_OFF_A_FP8  + r * SAC_BR_FP8_ROW_BYTES;
        saturn_quantize_row(in, out);
    }
}

/* ── Verification ─────────────────────────────────────────────────────── */

static int verify_a_fp8(void)
{
    uintptr_t base = SAC_BR_DRAM_BASE + SAC_BR_OFF_A_FP8;
    int mism = 0;
    for (uint32_t r = 0; r < SAC_BR_TILE_ROWS; r++) {
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t got = atlas_read32(base + r * SAC_BR_FP8_ROW_BYTES + w * 4);
            uint32_t exp = ((uint32_t)SAC_BR_EXPECT_A_FP8[r][4 * w + 3] << 24) |
                           ((uint32_t)SAC_BR_EXPECT_A_FP8[r][4 * w + 2] << 16) |
                           ((uint32_t)SAC_BR_EXPECT_A_FP8[r][4 * w + 1] <<  8) |
                           (uint32_t)SAC_BR_EXPECT_A_FP8[r][4 * w];
            if (got != exp) {
                if (mism < 8) {
                    printf("  CKPT1 row %u word %u: got 0x%08x exp 0x%08x\n",
                           r, w, got, exp);
                }
                mism++;
            }
        }
    }
    return mism;
}

static int verify_out(void)
{
    uintptr_t base = SAC_BR_DRAM_BASE + SAC_BR_OFF_OUT;
    int mism = 0;
    for (uint32_t r = 0; r < SAC_BR_TOTAL_BANK_ROWS; r++) {
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t got = atlas_read32(base + r * SAC_BR_BANK_ROW_BYTES + w * 4);
            uint32_t exp = ((uint32_t)SAC_BR_EXPECT_OUT[r][2 * w + 1] << 16) |
                           (uint32_t)SAC_BR_EXPECT_OUT[r][2 * w];
            if (got != exp) {
                if (mism < 8) {
                    printf("  CKPT2 bank-row %u word %u: got 0x%08x exp 0x%08x\n",
                           r, w, got, exp);
                }
                mism++;
            }
        }
    }
    return mism;
}

/* ── Main ─────────────────────────────────────────────────────────────── */

int main(void)
{
    printf("Saturn -> Atlas FP8 quantize bridge "
           "(saturn vfncvtbf16 -> Atlas mxu1 matmul + bias)\n");

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);
    asm volatile ("fence" ::: "memory");

    printf("Seeding DRAM (A_bf16 @0x2000, B_fp8 @0x0400, bias @0x0800) ...\n");
    seed_a_bf16();
    seed_b_fp8();
    seed_bias();
    asm volatile ("fence" ::: "memory");

    /* ── Stage 1: Saturn native BF16 -> E4M3 quantize ───────────────── */
    printf("Saturn: native BF16 -> E4M3 quantize of A_bf16 -> A_fp8 (32 rows) ...\n");
    run_saturn_quantize();
    asm volatile ("fence" ::: "memory");

    /* ── Checkpoint 1: Saturn quantizer alignment ───────────────────── */
    printf("Checkpoint 1: Saturn A_fp8 vs bf16_tile_to_e4m3_bytes ...\n");
    int mism1 = verify_a_fp8();
    if (mism1) {
        printf("FAIL: %d byte mismatches at checkpoint 1 "
               "(Saturn quantizer encoding alignment)\n", mism1);
        return 1;
    }
    printf("  CKPT1 PASS — Saturn E4M3 bytes match Atlas roundtrip golden\n");

    /* ── Stage 2: Atlas MXU1 matmul + bias-add ─────────────────────── */
    printf("Atlas: staging smolvla_fused_matmul_bias_v2_mxu1 (%u words) ...\n",
           KERNEL_LEN);
    if (load_kernel(kernel_words, KERNEL_LEN, "v2_mxu1")) {
        printf("FAIL: kernel IMEM readback\n");
        return 1;
    }

    printf("Atlas: starting ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t status = 0;
    if (wait_halted(&status)) {
        printf("FAIL: Atlas did not halt (status=0x%08x)\n", status);
        return 1;
    }
    uint32_t reason = atlas_halt_reason();
    uint32_t dbg0   = atlas_dbg0();
    uint32_t cycles = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    printf("  Atlas done: status=0x%08x halt_reason=%u dbg0=%u cycles=%u\n",
           status, reason, dbg0, cycles);
    if (reason != ATLAS_HALT_ECALL || dbg0 != 1) {
        printf("FAIL: Atlas post-conditions (expected halt_reason=ECALL, dbg0=1)\n");
        return 1;
    }

    /* ── Checkpoint 2: end-to-end output ─────────────────────────────── */
    asm volatile ("fence" ::: "memory");
    printf("Checkpoint 2: Atlas BF16 output vs IPT + bias-add golden ...\n");
    int mism2 = verify_out();
    if (mism2) {
        printf("FAIL: %d byte mismatches at checkpoint 2 (end-to-end)\n", mism2);
        return 1;
    }

    printf("PASS: saturn_atlas_fp8_quantize_bridge "
           "(saturn vfncvtbf16 -> Atlas mxu1 matmul + bias)\n");
    return 0;
}

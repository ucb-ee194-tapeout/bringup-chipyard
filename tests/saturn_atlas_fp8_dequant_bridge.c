#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "atlas_mmio.h"
#include "saturn_rvv_mx.h"
#include "saturn_atlas_fp8_dequant_bridge_golden.h"

/*
 * Atlas -> Saturn FP8 dequantize bridge test.
 *
 * Closes the bidirectional FP8 handoff story (the partner test
 * saturn_atlas_fp8_quantize_bridge.c covers the Saturn -> Atlas direction).
 * "Handoff" rather than "byte-equivalence": each side is verified against
 * its own correct golden, but Atlas's scaled VFP8UNPACK output and Saturn's
 * no-scale vfwcvtbf16 decode produce different bits by construction (the
 * E8M0 scale 0x7E only applies to the Atlas path).
 *
 *   stage 0  host:    seed A_fp8, B_fp8, scale byte in DRAM
 *   stage 1  Atlas:   run mxu1_quant_relu_roundtrip kernel:
 *                       MXU1 matmul (A_fp8 @ B_fp8 -> BF16 raw at 0x1000)
 *                       VPU ReLU on BF16 raw (in MRF)
 *                       VFP8PACK pack to FP8 (with E8M0 scale 0x7E) at 0x1800
 *                       VFP8UNPACK roundtrip back to BF16 at 0x2000 (unused)
 *
 *   ---- checkpoint 1 (Atlas FP8 production) ----
 *     compare Atlas's packed FP8 bytes at 0x1800 against the Python
 *     golden via IPTLinearRTLFunction + VPU ReLU + run_fp8_pack_rows.
 *
 *   stage 2  Saturn:  load FP8 bytes from 0x1800 via vle8.v, widen via
 *                     native vfwcvtbf16.f.f.v (E4M3 -> BF16, no scale),
 *                     store BF16 to 0x3000.
 *
 *   ---- checkpoint 2 (Saturn FP8 decode) ----
 *     compare Saturn's decoded BF16 at 0x3000 against the Python golden
 *     via e4m3_bytes_to_float (canonical E4M3 -> FP32 LUT) cast to BF16
 *     (RNE narrow). Saturn's vfwcvtbf16.f.f.v applies no scale; this
 *     golden applies no scale. They should match.
 *
 * Atlas's own VFP8UNPACK output at 0x2000 applies the E8M0 scale (0x7E =
 * 2^-1), so it is intentionally not compared against Saturn's no-scale
 * decode. This cooperative test verifies Atlas's packed FP8 bytes at
 * 0x1800 and Saturn's no-scale decode at 0x3000; the existing
 * mxu1_quant_relu_roundtrip baremetal test covers Atlas's scaled unpack
 * output.
 *
 * Target config: EE290SimConfig.
 */

#define SAC_FQ_DRAM_BASE     ((uint64_t)ATLAS_DRAM_BASE)

/* DRAM offsets, matching mxu1_quant_relu_roundtrip.S kernel layout. */
#define SAC_FQ_OFF_A_FP8     0x0000U   /* 1024 B,  host seeds  */
#define SAC_FQ_OFF_B_FP8     0x0400U   /* 1024 B,  host seeds  */
#define SAC_FQ_OFF_SCALE     0x0800U   /* 32 B,    host seeds  (only byte 0 used) */
#define SAC_FQ_OFF_RAW_BF16  0x1000U   /* 2048 B,  Atlas writes (not verified) */
#define SAC_FQ_OFF_PACK_FP8  0x1800U   /* 1024 B,  Atlas writes -> ckpt 1 */
#define SAC_FQ_OFF_UNPACK    0x2000U   /* 2048 B,  Atlas writes (not verified) */
#define SAC_FQ_OFF_SATURN    0x3000U   /* 2048 B,  Saturn writes -> ckpt 2 */

#define SAC_FQ_FP8_ROW_BYTES  32U       /* 32 E4M3 lanes per row */
#define SAC_FQ_BF16_ROW_BYTES 64U       /* 32 BF16 lanes per row */

#define POLL_ITERS            200000000U

/* ── Atlas kernel words ───────────────────────────────────────────────
 * Assembled from generators/atlas-npu/baremetal/assembly/
 * mxu1_quant_relu_roundtrip.S via:
 *   python3 generators/atlas-npu/baremetal/assembler.py \
 *     generators/atlas-npu/baremetal/assembly/mxu1_quant_relu_roundtrip.S \
 *     --out-hex /tmp/qrr.hex
 * Pass marker: CSRRW x0, 0xC10, 1; ECALL (dbg0=1, halt_reason=ECALL).
 */
#define KERNEL_LEN 78
static const uint32_t kernel_words[KERNEL_LEN] = {
    0x00100E13U, 0x00000293U, 0x0002807FU, 0x40000613U, 0x000016B7U, 0x80068693U, 0x02000713U, 0x00000313U,
    0x90000537U, 0x00C5037BU, 0x0200007FU, 0x10000313U, 0x90000537U, 0x40050513U, 0x00C5137BU, 0x0200107FU,
    0x20000313U, 0x90001537U, 0x80050513U, 0x00E5237BU, 0x0200207FU, 0x000013B7U, 0x80038393U, 0x0003E003U,
    0x00201067U, 0x00000313U, 0x00030007U, 0x02101067U, 0x10000313U, 0x00030207U, 0x02101067U, 0x02008077U,
    0x01F01067U, 0x16000077U, 0x02201067U, 0x12000477U, 0x01F01067U, 0x90010657U, 0x04001067U, 0x88600857U,
    0x04001067U, 0x8A800A57U, 0x04101067U, 0x40000313U, 0x00032407U, 0x02101067U, 0x50000313U, 0x00032487U,
    0x02101067U, 0x90001537U, 0x40000313U, 0x02D3357BU, 0x0200307FU, 0x60000313U, 0x00032807U, 0x02101067U,
    0x90002537U, 0x80050513U, 0x60000313U, 0x02C3457BU, 0x0200407FU, 0x00001337U, 0x80030313U, 0x00032A07U,
    0x02101067U, 0x00001337U, 0x90030313U, 0x00032A87U, 0x02101067U, 0x90002537U, 0x00001337U, 0x80030313U,
    0x02D3557BU, 0x0200507FU, 0x00100093U, 0xC1009073U, 0x00000073U, 0xC10E1073U, 0x00000073U,
};

/* ── DRAM seed helpers ───────────────────────────────────────────────── */

static void seed_fp8_tile(uintptr_t off, const uint8_t bytes[][SAC_FQ_TILE_COLS])
{
    uintptr_t base = SAC_FQ_DRAM_BASE + off;
    for (uint32_t r = 0; r < SAC_FQ_TILE_ROWS; r++) {
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t v = ((uint32_t)bytes[r][4 * w + 3] << 24) |
                         ((uint32_t)bytes[r][4 * w + 2] << 16) |
                         ((uint32_t)bytes[r][4 * w + 1] <<  8) |
                         (uint32_t)bytes[r][4 * w];
            atlas_write32(base + r * SAC_FQ_FP8_ROW_BYTES + w * 4, v);
        }
    }
}

static void seed_scale_byte(uint8_t scale_e8m0)
{
    /* Kernel reads only byte 0 of the scale region; write a 32B beat with
     * scale in lane 0 and zeros elsewhere. */
    uintptr_t base = SAC_FQ_DRAM_BASE + SAC_FQ_OFF_SCALE;
    atlas_write32(base, (uint32_t)scale_e8m0);
    for (uint32_t w = 1; w < 8; w++) {
        atlas_write32(base + w * 4, 0);
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

/* ── Saturn native FP8 (E4M3) -> BF16 widen ─────────────────────────────
 *
 * Per row (32 E4M3 bytes -> 32 BF16 lanes):
 *   1. vsetvli  SEW=e8  LMUL=m1 alt=0   (32 E4M3 lanes; alt=0 = E4M3)
 *   2. vle8.v   v0                       (load 32 bytes)
 *   3. vsetvli  SEW=e8  LMUL=m1 alt=0   (op SEW; alt=0 = E4M3 source)
 *   4. VFWCVTBF16_F_F_V                  (widen E4M3 -> BF16, output 2x)
 *   5. vsetvli  SEW=e16 LMUL=m2 alt=0   (BF16 store SEW, 2x lmul)
 *   6. vse16.v  v8                       (store 32 BF16 = 64 bytes)
 *
 * The reference Saturn benchmark vec-mx-unary e4m3_widen row uses ealt=0,
 * confirming alt=0 selects E4M3 input at SEW=8 op.
 */
static void saturn_dequant_row(uintptr_t in_fp8_addr, uintptr_t out_bf16_addr)
{
    size_t vl;

    VSETVLI_ALTFMT(vl, SAC_FQ_TILE_COLS, SEW_E8, LMUL_M1, 0);
    asm volatile ("vle8.v v0, (%0)" : : "r"(in_fp8_addr)
                  : "v0", "memory");

    VSETVLI_ALTFMT_X0(vl, SEW_E8, LMUL_M1, 0);
    VFWCVTBF16_F_F_V(V8, V0);

    VSETVLI_ALTFMT_X0(vl, SEW_E16, LMUL_M2, 0);
    asm volatile ("vse16.v v8, (%0)" : : "r"(out_bf16_addr)
                  : "v8", "v9", "memory");
}

static void run_saturn_dequant(void)
{
    /* No frm pin needed: E4M3 -> BF16 widen is exact (BF16 has more
     * mantissa precision than E4M3 for normals); rounding mode is
     * irrelevant. */
    for (uint32_t r = 0; r < SAC_FQ_TILE_ROWS; r++) {
        uintptr_t in  = SAC_FQ_DRAM_BASE + SAC_FQ_OFF_PACK_FP8
                        + r * SAC_FQ_FP8_ROW_BYTES;
        uintptr_t out = SAC_FQ_DRAM_BASE + SAC_FQ_OFF_SATURN
                        + r * SAC_FQ_BF16_ROW_BYTES;
        saturn_dequant_row(in, out);
    }
}

/* ── Verification ─────────────────────────────────────────────────────── */

static int verify_pack_fp8(void)
{
    uintptr_t base = SAC_FQ_DRAM_BASE + SAC_FQ_OFF_PACK_FP8;
    int mism = 0;
    for (uint32_t r = 0; r < SAC_FQ_TILE_ROWS; r++) {
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t got = atlas_read32(base + r * SAC_FQ_FP8_ROW_BYTES + w * 4);
            uint32_t exp = ((uint32_t)SAC_FQ_EXPECT_PACK_FP8[r][4 * w + 3] << 24) |
                           ((uint32_t)SAC_FQ_EXPECT_PACK_FP8[r][4 * w + 2] << 16) |
                           ((uint32_t)SAC_FQ_EXPECT_PACK_FP8[r][4 * w + 1] <<  8) |
                           (uint32_t)SAC_FQ_EXPECT_PACK_FP8[r][4 * w];
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

static int verify_saturn_bf16(void)
{
    uintptr_t base = SAC_FQ_DRAM_BASE + SAC_FQ_OFF_SATURN;
    int mism = 0;
    for (uint32_t r = 0; r < SAC_FQ_TILE_ROWS; r++) {
        for (uint32_t w = 0; w < 16; w++) {
            uint32_t got = atlas_read32(base + r * SAC_FQ_BF16_ROW_BYTES + w * 4);
            uint32_t exp = ((uint32_t)SAC_FQ_EXPECT_SATURN_BF16[r][2 * w + 1] << 16) |
                           (uint32_t)SAC_FQ_EXPECT_SATURN_BF16[r][2 * w];
            if (got != exp) {
                if (mism < 8) {
                    printf("  CKPT2 row %u word %u: got 0x%08x exp 0x%08x\n",
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
    printf("Atlas -> Saturn FP8 dequantize bridge "
           "(Atlas mxu1+pack -> Saturn vfwcvtbf16)\n");

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);
    asm volatile ("fence" ::: "memory");

    printf("Seeding DRAM (A_fp8 @0x0000, B_fp8 @0x0400, scale @0x0800) ...\n");
    seed_fp8_tile(SAC_FQ_OFF_A_FP8, SAC_FQ_A_FP8);
    seed_fp8_tile(SAC_FQ_OFF_B_FP8, SAC_FQ_B_FP8);
    seed_scale_byte(SAC_FQ_SCALE_BYTE);
    asm volatile ("fence" ::: "memory");

    /* ── Stage 1: Atlas mxu1_quant_relu_roundtrip ─────────────────── */
    printf("Atlas: staging mxu1_quant_relu_roundtrip (%u words) ...\n",
           KERNEL_LEN);
    if (load_kernel(kernel_words, KERNEL_LEN, "qrr")) {
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

    /* ── Checkpoint 1: Atlas FP8 production matches IPT+ReLU+pack golden ── */
    asm volatile ("fence" ::: "memory");
    printf("Checkpoint 1: Atlas packed FP8 vs IPT+ReLU+VFP8PACK golden ...\n");
    int mism1 = verify_pack_fp8();
    if (mism1) {
        printf("FAIL: %d byte mismatches at checkpoint 1 (Atlas FP8 production)\n", mism1);
        return 1;
    }
    printf("  CKPT1 PASS -- Atlas FP8 bytes match IPT+ReLU+pack golden\n");

    /* ── Stage 2: Saturn FP8 -> BF16 widen ───────────────────────────── */
    printf("Saturn: native E4M3 -> BF16 widen of packed FP8 (32 rows) ...\n");
    run_saturn_dequant();
    asm volatile ("fence" ::: "memory");

    /* ── Checkpoint 2: Saturn-decoded BF16 matches e4m3_bytes_to_float ── */
    printf("Checkpoint 2: Saturn BF16 vs e4m3_bytes_to_float + BF16 cast golden ...\n");
    int mism2 = verify_saturn_bf16();
    if (mism2) {
        printf("FAIL: %d word mismatches at checkpoint 2 (Saturn FP8 decode)\n", mism2);
        return 1;
    }

    printf("PASS: saturn_atlas_fp8_dequant_bridge "
           "(Atlas mxu1+pack -> Saturn vfwcvtbf16)\n");
    return 0;
}

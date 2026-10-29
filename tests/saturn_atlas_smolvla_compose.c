#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"
#include "saturn_atlas_smolvla_compose_golden.h"

/*
 * Saturn <-> Atlas smolvla composition test.
 *
 * Two Atlas BF16 kernels run back-to-back with a Saturn-driven BF16 row
 * scale in between:
 *
 *   Phase 1 (Atlas):    Y  = A * B                 (smolvla_elementwise_mul)
 *   Interlude (Saturn): Y' = trunc_bf16(Y * scale)  (RVV, row-broadcast)
 *   Phase 2 (Atlas):    S  = row_reduce(Y')         (smolvla_reduction_sum)
 *
 * Validates: concurrent Shuttle+Saturn / Atlas access to DRAM at
 * SAC_DRAM_BASE, BF16 handoff across the phase boundary without
 * native Zvfbf16, and back-to-back IMEM swap while halted.
 *
 * Target config: EE290SimConfig. Golden produced by
 * generators/atlas-npu/baremetal/generators/gen_saturn_atlas_smolvla_compose.py.
 */

#define SAC_DRAM_BASE    ((uint64_t)ATLAS_DRAM_BASE)

/* DRAM layout (byte offsets from SAC_DRAM_BASE), matching the stock
 * smolvla kernels' expectations (see the kernel .S files for details):
 *   Phase 1 reads  A at 0x0000/0x0400, B at 0x0800/0x0C00
 *   Phase 1 writes Y at 0x1000/0x1400
 *   Saturn reads   Y at 0x1000/0x1400, writes Y' to 0x0000/0x0400
 *   Phase 2 reads  X at 0x0000/0x0400 (= Y'), writes S to 0x0800/0x0C00
 */
#define SAC_OFF_A_B0     0x0000U
#define SAC_OFF_A_B1     0x0400U
#define SAC_OFF_B_B0     0x0800U
#define SAC_OFF_B_B1     0x0C00U
#define SAC_OFF_Y_B0     0x1000U
#define SAC_OFF_Y_B1     0x1400U
#define SAC_OFF_S_B0     0x0800U
#define SAC_OFF_S_B1     0x0C00U

#define SAC_BANK_ROWS    32U    /* rows per bank */
#define SAC_ROW_BYTES    32U    /* 16 BF16 lanes = 32 bytes */
#define SAC_BANK_BYTES   (SAC_BANK_ROWS * SAC_ROW_BYTES)   /* 1024 */

#define POLL_ITERS       200000000U

/* ── Assembled smolvla kernels ─────────────────────────────────────────────
 *
 * Regenerate with:
 *   python3 generators/atlas-npu/baremetal/assembler.py \
 *     generators/atlas-npu/baremetal/assembly/smolvla_elementwise_mul.S \
 *     --out-hex /tmp/mul.hex
 *   (and likewise for smolvla_reduction_sum.S).
 *
 * Both end with CSRRW x0, 0xC10, 1 + ECALL on the pass path (dbg0 = 1).
 */

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

/* ── DRAM helpers ─────────────────────────────────────────────────────── */

static void seed_bank(uintptr_t dram_off, const uint16_t rows[][SAC_LANES], uint32_t first_row)
{
    /* Write SAC_BANK_ROWS rows of 16 BF16 lanes each (1 KiB per bank).
     * Each row is 32 bytes = 8 u32 words. */
    uintptr_t base = SAC_DRAM_BASE + dram_off;
    for (uint32_t r = 0; r < SAC_BANK_ROWS; r++) {
        const uint16_t *row = rows[first_row + r];
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t val = ((uint32_t)row[2 * w + 1] << 16) | (uint32_t)row[2 * w];
            atlas_write32(base + r * SAC_ROW_BYTES + w * 4, val);
        }
    }
}

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

/* ── Saturn BF16 interlude via RVV ──────────────────────────────────────
 *
 * Per-row (16 BF16 lanes):
 *   1. vle16.v         load u16 bf16 lanes from DRAM
 *   2. vzext.vf2       widen to u32
 *   3. vsll.vi 16      shift into FP32 high-half bit pattern
 *   4. vfmul.vf        FP32 multiply by scalar scale
 *   5. vsrl.vi 16      shift FP32 result to low half
 *   6. vnsrl.wx (0)    narrow to u16
 *   7. vse16.v         store back to DRAM
 *
 * Semantics match bf16_upper_half_of_fp32_bits (software_models/vpu/bf16_utils.py)
 * — raw upper-16 slice of the FP32 product, no second RNE pass. The golden
 * generator uses the same policy for this stage (and only this stage).
 */
static void saturn_interlude_row(uintptr_t y_addr, uintptr_t yprime_addr, uint16_t scale_bits)
{
    uint32_t scale_u32 = ((uint32_t)scale_bits) << 16;
    float scale_f;
    memcpy(&scale_f, &scale_u32, sizeof(float));

    size_t vl = __riscv_vsetvl_e16m1(SAC_LANES);
    vuint16m1_t y_u16 = __riscv_vle16_v_u16m1((const uint16_t *)y_addr, vl);
    vuint32m2_t y_u32 = __riscv_vzext_vf2_u32m2(y_u16, vl);
    vuint32m2_t y_bits = __riscv_vsll_vx_u32m2(y_u32, 16, vl);
    vfloat32m2_t y_f32 = __riscv_vreinterpret_v_u32m2_f32m2(y_bits);
    vfloat32m2_t p_f32 = __riscv_vfmul_vf_f32m2(y_f32, scale_f, vl);
    vuint32m2_t p_bits = __riscv_vreinterpret_v_f32m2_u32m2(p_f32);
    vuint32m2_t p_hi   = __riscv_vsrl_vx_u32m2(p_bits, 16, vl);
    vuint16m1_t p_u16  = __riscv_vnsrl_wx_u16m1(p_hi, 0, vl);
    __riscv_vse16_v_u16m1((uint16_t *)yprime_addr, p_u16, vl);
}

static void run_interlude(void)
{
    /* Flat layout: bank0 rows [0..31] at DRAM[OFF_Y_B0..], bank1 rows [0..31]
     * at DRAM[OFF_Y_B1..]. Each row is 32 bytes. Destination layout is the
     * mirror at OFF_A_B0/OFF_A_B1 (overwriting the phase-1 A input slot). */
    for (uint32_t bank = 0; bank < 2; bank++) {
        uintptr_t y_base  = SAC_DRAM_BASE + (bank ? SAC_OFF_Y_B1 : SAC_OFF_Y_B0);
        uintptr_t yp_base = SAC_DRAM_BASE + (bank ? SAC_OFF_A_B1 : SAC_OFF_A_B0);
        for (uint32_t r = 0; r < SAC_BANK_ROWS; r++) {
            uintptr_t y_addr  = y_base  + r * SAC_ROW_BYTES;
            uintptr_t yp_addr = yp_base + r * SAC_ROW_BYTES;
            saturn_interlude_row(y_addr, yp_addr, SAC_SCALE[r]);
        }
    }
}

/* ── Verification ─────────────────────────────────────────────────────── */

static int verify_bank(uintptr_t dram_off, const uint16_t expected[][SAC_LANES],
                       uint32_t first_row, const char *label)
{
    uintptr_t base = SAC_DRAM_BASE + dram_off;
    int mism = 0;
    for (uint32_t r = 0; r < SAC_BANK_ROWS; r++) {
        const uint16_t *exp = expected[first_row + r];
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t got = atlas_read32(base + r * SAC_ROW_BYTES + w * 4);
            uint32_t exp_w = ((uint32_t)exp[2 * w + 1] << 16) | (uint32_t)exp[2 * w];
            if (got != exp_w) {
                if (mism < 4) {
                    printf("  [%s] DRAM[0x%08lx] mismatch: got 0x%08x exp 0x%08x\n",
                           label, (unsigned long)(base + r * SAC_ROW_BYTES + w * 4),
                           got, exp_w);
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
    printf("Saturn <-> Atlas smolvla composition (mul -> saturn scale -> reduce)\n");

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);
    asm volatile ("fence" ::: "memory");

    printf("Seeding DRAM A,B at 0x%lx ...\n", (unsigned long)SAC_DRAM_BASE);
    seed_bank(SAC_OFF_A_B0, SAC_A, 0);
    seed_bank(SAC_OFF_A_B1, SAC_A, SAC_BANK_ROWS);
    seed_bank(SAC_OFF_B_B0, SAC_B, 0);
    seed_bank(SAC_OFF_B_B1, SAC_B, SAC_BANK_ROWS);
    asm volatile ("fence" ::: "memory");

    /* ── Phase 1: Atlas mul ─────────────────────────────────────────── */
    printf("Phase 1: staging smolvla_elementwise_mul (%u words) ...\n", KERNEL_MUL_LEN);
    if (load_kernel(kernel_mul, KERNEL_MUL_LEN, "mul")) {
        printf("FAIL: kernel_mul IMEM readback\n");
        return 1;
    }

    printf("Phase 1: starting Atlas ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t status1 = 0;
    if (wait_halted(&status1)) {
        printf("FAIL: phase 1 did not halt (status=0x%08x)\n", status1);
        return 1;
    }
    uint32_t reason1 = atlas_halt_reason();
    uint32_t dbg0_1  = atlas_dbg0();
    printf("  phase 1 done: status=0x%08x halt_reason=%u dbg0=%u\n",
           status1, reason1, dbg0_1);
    if (reason1 != ATLAS_HALT_ECALL || dbg0_1 != 1) {
        printf("FAIL: phase 1 post-conditions (expected halt_reason=ECALL, dbg0=1)\n");
        return 1;
    }

    /* ── Saturn interlude ──────────────────────────────────────────── */
    asm volatile ("fence" ::: "memory");
    printf("Interlude: Saturn RVV BF16 row-scale (u16 -> f32 -> trunc) ...\n");
    run_interlude();
    asm volatile ("fence" ::: "memory");

    /* ── Phase 2: Atlas row-reduce ─────────────────────────────────── */
    atlas_stop();   /* defensive — already halted, but clears any lingering state */
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    asm volatile ("fence" ::: "memory");

    printf("Phase 2: staging smolvla_reduction_sum (%u words) ...\n", KERNEL_RED_LEN);
    if (load_kernel(kernel_red, KERNEL_RED_LEN, "red")) {
        printf("FAIL: kernel_red IMEM readback\n");
        return 1;
    }

    printf("Phase 2: starting Atlas ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t status2 = 0;
    if (wait_halted(&status2)) {
        printf("FAIL: phase 2 did not halt (status=0x%08x)\n", status2);
        return 1;
    }
    uint32_t reason2 = atlas_halt_reason();
    uint32_t dbg0_2  = atlas_dbg0();
    uint32_t cycles  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t instcnt = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    printf("  phase 2 done: status=0x%08x halt_reason=%u dbg0=%u\n",
           status2, reason2, dbg0_2);
    printf("  cycles=%u instcnt=%u\n", cycles, instcnt);
    if (reason2 != ATLAS_HALT_ECALL || dbg0_2 != 1) {
        printf("FAIL: phase 2 post-conditions (expected halt_reason=ECALL, dbg0=1)\n");
        return 1;
    }

    /* ── Verify final row-sum output ───────────────────────────────── */
    asm volatile ("fence" ::: "memory");
    int mism = 0;
    mism += verify_bank(SAC_OFF_S_B0, SAC_EXPECT_S, 0,               "S bank0");
    mism += verify_bank(SAC_OFF_S_B1, SAC_EXPECT_S, SAC_BANK_ROWS,   "S bank1");

    if (mism) {
        printf("FAIL: %d DRAM mismatches in final row sums\n", mism);
        return 1;
    }

    printf("PASS: saturn_atlas_smolvla_compose (mul -> saturn scale -> reduce)\n");
    return 0;
}

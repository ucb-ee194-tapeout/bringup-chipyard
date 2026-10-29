#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "atlas_mmio.h"
#include "saturn_rvv_mx.h"
#include "saturn_atlas_smolvla_compose_native_bf16_golden.h"

/*
 * Saturn <-> Atlas smolvla composition test — native BF16 interlude.
 *
 * Native-BF16 twin of saturn_atlas_smolvla_compose.c. Same Atlas kernels
 * (smolvla_elementwise_mul + smolvla_reduction_sum), same DRAM layout.
 * Only the Saturn interlude is rewritten:
 *
 *   Phase 1 (Atlas):    Y  = A * B                 (smolvla_elementwise_mul)
 *   Interlude (Saturn): Y' = bf16_rne(Y * scale)    (RVV native BF16 mul)
 *   Phase 2 (Atlas):    S  = row_reduce(Y')         (smolvla_reduction_sum)
 *
 * Replaces the 9-step bit-hack interlude (u16 -> zext -> sll16 -> reinterpret
 * f32 -> vfmul -> srl16 -> vnsrl) with 4 native ops via Saturn's MX path:
 *   vsetvli altfmt=1, SEW=e16, LMUL=m1
 *   vle16.v
 *   vfmul.vv  (vs1 = broadcast BF16 scalar)
 *   vse16.v
 *
 * The .insn-style helpers in saturn_rvv_mx.h emit raw encodings that don't
 * require GCC to know about zfbfmin/zvfbfa — same approach Saturn benchmarks
 * use. The VCS simv (EE290SimConfig + VectorParams.mxParams) executes them.
 *
 * Semantic difference vs the truncation twin: BF16 narrow uses RNE, not
 * upper-16 bit truncation. Goldens regenerate via
 *   gen_saturn_atlas_smolvla_compose.py --interlude-mode=native_bf16_rne
 * which routes the interlude golden through bf16_mul (== f32_to_bf16_bits_rne
 * after FP32 mul).
 *
 * Target config: EE290SimConfig.
 */

#define SAC_DRAM_BASE    ((uint64_t)ATLAS_DRAM_BASE)

#define SAC_OFF_A_B0     0x0000U
#define SAC_OFF_A_B1     0x0400U
#define SAC_OFF_B_B0     0x0800U
#define SAC_OFF_B_B1     0x0C00U
#define SAC_OFF_Y_B0     0x1000U
#define SAC_OFF_Y_B1     0x1400U
#define SAC_OFF_S_B0     0x0800U
#define SAC_OFF_S_B1     0x0C00U

#define SAC_BANK_ROWS    32U
#define SAC_ROW_BYTES    32U
#define SAC_BANK_BYTES   (SAC_BANK_ROWS * SAC_ROW_BYTES)

#define POLL_ITERS       200000000U

/* Atlas kernel words — same as the truncation twin (regenerate via the
 * sp26 assembler if the .S kernels change). */

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

/* ── Saturn native-BF16 interlude via RVV ──────────────────────────────
 *
 * Per row (16 BF16 lanes):
 *   1. vsetvli   altfmt=1 SEW=e16 LMUL=m1   (BF16 mode)
 *   2. vle16.v                              (load 16 BF16 lanes)
 *   3. vmv.v.x                              (broadcast BF16 scalar from GPR)
 *   4. vfmul.vv                             (BF16 lane-wise mul, RNE)
 *   5. vse16.v                              (store 16 BF16 lanes)
 *
 * Compare to the bit-hack twin (saturn_atlas_smolvla_compose.c:128-158):
 * 9 ops including zero-extend, two shifts, two reinterprets, FP32 mul,
 * and a narrow. Native is shorter and matches Saturn's vec-mx-scale
 * benchmark pattern (generators/saturn/benchmarks/vec-mx-scale/main.c).
 *
 * Rounding: vfmul.vv at altfmt=1 + SEW=e16 produces a BF16 result with RNE
 * (the FP32-mul-then-RNE-narrow path inside Saturn's MX FMA pipeline).
 * That matches f32_to_bf16_bits_rne, which is what the native-mode golden
 * generator uses (gen_saturn_atlas_smolvla_compose.py --interlude-mode
 * native_bf16_rne, which routes through bf16_mul).
 */
static void saturn_interlude_row_native(uintptr_t y_addr,
                                        uintptr_t yprime_addr,
                                        uint16_t scale_bits)
{
    /* BF16 scalar bits in lower 16 of a GPR; vmv.v.x broadcasts at SEW=e16. */
    uintptr_t scale_xreg = (uintptr_t)scale_bits;

    size_t vl;
    VSETVLI_ALTFMT(vl, SAC_LANES, SEW_E16, LMUL_M1, 1);

    asm volatile (
        "vle16.v   v0, (%[in])\n"
        "vmv.v.x   v1, %[scale]\n"
        "vfmul.vv  v2, v0, v1\n"
        "vse16.v   v2, (%[out])\n"
        :
        : [in]"r"(y_addr), [scale]"r"(scale_xreg), [out]"r"(yprime_addr)
        : "v0", "v1", "v2", "memory"
    );
}

static void run_interlude(void)
{
    /* Pin frm=RNE before native BF16 narrow ops; the golden was generated
     * via bf16_mul (= f32_to_bf16_bits_rne after FP32 mul). Don't depend
     * on ambient reset state. */
    SATURN_SET_FRM_RNE();

    for (uint32_t bank = 0; bank < 2; bank++) {
        uintptr_t y_base  = SAC_DRAM_BASE + (bank ? SAC_OFF_Y_B1 : SAC_OFF_Y_B0);
        uintptr_t yp_base = SAC_DRAM_BASE + (bank ? SAC_OFF_A_B1 : SAC_OFF_A_B0);
        for (uint32_t r = 0; r < SAC_BANK_ROWS; r++) {
            uintptr_t y_addr  = y_base  + r * SAC_ROW_BYTES;
            uintptr_t yp_addr = yp_base + r * SAC_ROW_BYTES;
            saturn_interlude_row_native(y_addr, yp_addr, SAC_SCALE[r]);
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
    printf("Saturn <-> Atlas smolvla composition (mul -> saturn NATIVE BF16 scale -> reduce)\n");

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

    /* ── Saturn interlude (NATIVE BF16) ────────────────────────────── */
    asm volatile ("fence" ::: "memory");
    printf("Interlude: Saturn RVV native BF16 row-scale (vfmul.vv altfmt=1) ...\n");
    run_interlude();
    asm volatile ("fence" ::: "memory");

    /* ── Phase 2: Atlas row-reduce ─────────────────────────────────── */
    atlas_stop();
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
    mism += verify_bank(SAC_OFF_S_B0, SAC_EXPECT_S, 0,             "S bank0");
    mism += verify_bank(SAC_OFF_S_B1, SAC_EXPECT_S, SAC_BANK_ROWS, "S bank1");

    if (mism) {
        printf("FAIL: %d DRAM mismatches in final row sums (native BF16)\n", mism);
        return 1;
    }

    printf("PASS: saturn_atlas_smolvla_compose_native_bf16 (mul -> saturn native BF16 scale -> reduce)\n");
    return 0;
}

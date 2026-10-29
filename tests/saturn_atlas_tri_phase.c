#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"
#include "saturn_atlas_tri_phase_golden.h"

/*
 * Saturn <-> Atlas tri-phase composition.
 *
 * Combines ECALL handshake + in-IMEM JALR cross-half dispatch + Saturn
 * cooperative RVV compute in a single cooperative test:
 *
 *   Phase 1 (Atlas):      Y   = A * B                  (smolvla_elementwise_mul)
 *   Interlude 1 (Saturn): Y'  = trunc_bf16(Y * scale)  (RVV, row-broadcast)
 *                         + stage kernel_red_jal @ IMEM lower half
 *                         + stage kernel_add     @ IMEM upper half
 *   Phase 2 (Atlas):      S   = row_reduce(Y')         (sac_reduction_sum_to_high)
 *                         JALR to IMEM upper half, NO halt
 *   Phase 3 (Atlas):      Z   = Y' + S                 (smolvla_elementwise_add)
 *                         ECALL with dbg0 = 1          (single halt for both kernels)
 *   Interlude 2 (Saturn): Z'' = trunc_bf16(Z * post_scale) (RVV, row-broadcast)
 *
 * Host sees exactly two Atlas halts: one after phase 1, one after phase 3.
 * Phase 2 does NOT halt — it JALs into phase 3.
 *
 * Target config: EE290SimConfig. Golden produced by
 * generators/atlas-npu/baremetal/generators/gen_saturn_atlas_tri_phase.py.
 */

#define SAC_DRAM_BASE    ((uint64_t)ATLAS_DRAM_BASE)

/* DRAM layout, same shape as compose test's phase 1+2. Phase 3 reuses
 * the phase-1 A slot (still-valid bytes after phase-2 DMA.LOAD) as its
 * A input and the phase-2 S slot as its B input, writing Z to the
 * stale phase-1 Y slot. Saturn's post-scale output lands in a fresh
 * slot so verification reads a bank untouched by any Atlas kernel.
 *
 *   Phase 1 reads  A at 0x0000/0x0400, B at 0x0800/0x0C00
 *   Phase 1 writes Y at 0x1000/0x1400
 *   Saturn reads   Y at 0x1000/0x1400, writes Y' to 0x0000/0x0400
 *   Phase 2 reads  X at 0x0000/0x0400 (= Y'), writes S to 0x0800/0x0C00
 *   Phase 3 reads  A at 0x0000/0x0400 (= Y' bytes, DMA.LOAD leaves DRAM unchanged),
 *                  B at 0x0800/0x0C00 (= S), writes Z at 0x1000/0x1400
 *   Saturn reads   Z at 0x1000/0x1400, writes Z'' to 0x1800/0x1C00
 */
#define SAC_TRI_OFF_A_B0     0x0000U
#define SAC_TRI_OFF_A_B1     0x0400U
#define SAC_TRI_OFF_B_B0     0x0800U
#define SAC_TRI_OFF_B_B1     0x0C00U
#define SAC_TRI_OFF_Y_B0     0x1000U
#define SAC_TRI_OFF_Y_B1     0x1400U
#define SAC_TRI_OFF_S_B0     0x0800U
#define SAC_TRI_OFF_S_B1     0x0C00U
#define SAC_TRI_OFF_Z_B0     0x1000U
#define SAC_TRI_OFF_Z_B1     0x1400U
#define SAC_TRI_OFF_Z2_B0    0x1800U
#define SAC_TRI_OFF_Z2_B1    0x1C00U

#define SAC_TRI_BANK_ROWS    32U
#define SAC_TRI_ROW_BYTES    32U    /* 16 BF16 lanes = 32 bytes */

/* IMEM staging: phase-2 kernel goes at word 0 (byte 0); phase-3 kernel
 * at word 0x4000 (byte 0x10000) — the upper-half boundary the JAL in
 * sac_reduction_sum_to_high.S is hardcoded to target. */
#define SAC_TRI_IMEM_UPPER_WORD   0x4000U

#define POLL_ITERS       200000000U

/* ── Assembled smolvla kernels ─────────────────────────────────────────────
 *
 * Regenerate with:
 *   python3 generators/atlas-npu/baremetal/assembler.py \
 *     generators/atlas-npu/baremetal/assembly/smolvla_elementwise_mul.S \
 *     --out-hex /tmp/mul.hex
 *   (and likewise for sac_reduction_sum_to_high.S and smolvla_elementwise_add.S).
 *
 * kernel_mul       — stock smolvla_elementwise_mul, ECALL pass block → dbg0=1.
 * kernel_red_jal   — sac_reduction_sum_to_high: row-reduce, then
 *                    LI x1, 0x4000 ; JALR x0, x1, 0  (hop to upper IMEM, no halt).
 *                    dbg0 is NOT written here; phase 3 writes it.
 * kernel_add       — stock smolvla_elementwise_add, ECALL pass block → dbg0=1.
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

/* NB: word 36 is a NOP (0x00000013 = ADDI x0,x0,0) occupying the
 * delay slot after JALR. Without it the fail-path CSRRW+ECALL at words
 * 37-38 would execute in the delay slot and halt Atlas before the JAL
 * redirect to kernel_add takes effect. */
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

/* ── DRAM helpers ─────────────────────────────────────────────────────── */

static void seed_bank(uintptr_t dram_off, const uint16_t rows[][SAC_TRI_LANES], uint32_t first_row)
{
    uintptr_t base = SAC_DRAM_BASE + dram_off;
    for (uint32_t r = 0; r < SAC_TRI_BANK_ROWS; r++) {
        const uint16_t *row = rows[first_row + r];
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t val = ((uint32_t)row[2 * w + 1] << 16) | (uint32_t)row[2 * w];
            atlas_write32(base + r * SAC_TRI_ROW_BYTES + w * 4, val);
        }
    }
}

static int load_kernel(const uint32_t *words, uint32_t n, const char *label, uint32_t word_offset)
{
    for (uint32_t i = 0; i < n; i++) {
        atlas_imem_write(word_offset + i, words[i]);
    }
    asm volatile ("fence" ::: "memory");
    int fail = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t got = atlas_imem_read(word_offset + i);
        if (got != words[i]) {
            printf("  IMEM MISMATCH[%s word %u]: got 0x%08x exp 0x%08x\n",
                   label, word_offset + i, got, words[i]);
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
 * Identical to saturn_atlas_smolvla_compose.c's interlude: u16 -> FP32
 * (shift-left-16, no round), FP32 multiply, upper-16 truncation. Used
 * by both interlude 1 (row-scale on Y) and interlude 2 (post-scale on Z).
 */
static void saturn_interlude_row(uintptr_t in_addr, uintptr_t out_addr, uint16_t scale_bits)
{
    uint32_t scale_u32 = ((uint32_t)scale_bits) << 16;
    float scale_f;
    memcpy(&scale_f, &scale_u32, sizeof(float));

    size_t vl = __riscv_vsetvl_e16m1(SAC_TRI_LANES);
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

/* Apply per-row scale across both banks, reading from in_b0/in_b1 and
 * writing to out_b0/out_b1. Each bank has SAC_TRI_BANK_ROWS rows of 16
 * BF16 lanes; scale is TILE_ROWS long, shared across banks. */
static void run_interlude(uintptr_t in_b0, uintptr_t in_b1,
                          uintptr_t out_b0, uintptr_t out_b1,
                          const uint16_t scale[SAC_TRI_TILE_ROWS])
{
    const uintptr_t in_bases[2]  = {in_b0,  in_b1};
    const uintptr_t out_bases[2] = {out_b0, out_b1};
    for (uint32_t bank = 0; bank < 2; bank++) {
        for (uint32_t r = 0; r < SAC_TRI_BANK_ROWS; r++) {
            uintptr_t in_addr  = in_bases[bank]  + r * SAC_TRI_ROW_BYTES;
            uintptr_t out_addr = out_bases[bank] + r * SAC_TRI_ROW_BYTES;
            saturn_interlude_row(in_addr, out_addr, scale[r]);
        }
    }
}

/* ── Verification ─────────────────────────────────────────────────────── */

static int verify_bank(uintptr_t dram_off, const uint16_t expected[][SAC_TRI_LANES],
                       uint32_t first_row, const char *label)
{
    uintptr_t base = SAC_DRAM_BASE + dram_off;
    int mism = 0;
    for (uint32_t r = 0; r < SAC_TRI_BANK_ROWS; r++) {
        const uint16_t *exp = expected[first_row + r];
        for (uint32_t w = 0; w < 8; w++) {
            uint32_t got = atlas_read32(base + r * SAC_TRI_ROW_BYTES + w * 4);
            uint32_t exp_w = ((uint32_t)exp[2 * w + 1] << 16) | (uint32_t)exp[2 * w];
            if (got != exp_w) {
                if (mism < 4) {
                    printf("  [%s] DRAM[0x%08lx] mismatch: got 0x%08x exp 0x%08x\n",
                           label, (unsigned long)(base + r * SAC_TRI_ROW_BYTES + w * 4),
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
    printf("Saturn <-> Atlas tri-phase composition "
           "(mul -> saturn scale -> reduce|JAL|add -> saturn scale)\n");

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);
    asm volatile ("fence" ::: "memory");

    printf("Seeding DRAM A,B at 0x%lx ...\n", (unsigned long)SAC_DRAM_BASE);
    seed_bank(SAC_TRI_OFF_A_B0, SAC_TRI_A, 0);
    seed_bank(SAC_TRI_OFF_A_B1, SAC_TRI_A, SAC_TRI_BANK_ROWS);
    seed_bank(SAC_TRI_OFF_B_B0, SAC_TRI_B, 0);
    seed_bank(SAC_TRI_OFF_B_B1, SAC_TRI_B, SAC_TRI_BANK_ROWS);
    asm volatile ("fence" ::: "memory");

    /* ── Phase 1: Atlas mul ─────────────────────────────────────────── */
    printf("Phase 1: staging smolvla_elementwise_mul (%u words) ...\n", KERNEL_MUL_LEN);
    if (load_kernel(kernel_mul, KERNEL_MUL_LEN, "mul", 0)) {
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

    asm volatile ("fence" ::: "memory");
    int y_mism = 0;
    y_mism += verify_bank(SAC_TRI_OFF_Y_B0, SAC_TRI_EXPECT_Y, 0,                  "Y bank0");
    y_mism += verify_bank(SAC_TRI_OFF_Y_B1, SAC_TRI_EXPECT_Y, SAC_TRI_BANK_ROWS,  "Y bank1");
    if (y_mism) { printf("FAIL: %d DRAM mismatches in Y (phase 1 output)\n", y_mism); return 1; }
    printf("  Y verified (phase 1 OK).\n");

    /* ── Saturn interlude 1: RVV scale + dual-kernel IMEM staging ─── */
    asm volatile ("fence" ::: "memory");
    printf("Interlude 1: Saturn RVV BF16 row-scale on Y -> Y' ...\n");
    run_interlude(SAC_DRAM_BASE + SAC_TRI_OFF_Y_B0, SAC_DRAM_BASE + SAC_TRI_OFF_Y_B1,
                  SAC_DRAM_BASE + SAC_TRI_OFF_A_B0, SAC_DRAM_BASE + SAC_TRI_OFF_A_B1,
                  SAC_TRI_SCALE);
    asm volatile ("fence" ::: "memory");

    int yp_mism = 0;
    yp_mism += verify_bank(SAC_TRI_OFF_A_B0, SAC_TRI_EXPECT_YP, 0,                  "Y' bank0");
    yp_mism += verify_bank(SAC_TRI_OFF_A_B1, SAC_TRI_EXPECT_YP, SAC_TRI_BANK_ROWS,  "Y' bank1");
    if (yp_mism) { printf("FAIL: %d DRAM mismatches in Y' (interlude 1 output)\n", yp_mism); return 1; }
    printf("  Y' verified (interlude 1 OK).\n");

    /* ── Phase 2+3: one Atlas run, one halt, two kernels chained via JAL ── */
    atlas_stop();   /* defensive */
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    asm volatile ("fence" ::: "memory");

    printf("Interlude 1: staging kernel_red_jal (%u words) @ IMEM lower half ...\n",
           KERNEL_RED_JAL_LEN);
    if (load_kernel(kernel_red_jal, KERNEL_RED_JAL_LEN, "red_jal", 0)) {
        printf("FAIL: kernel_red_jal IMEM readback\n");
        return 1;
    }
    printf("Interlude 1: staging kernel_add (%u words) @ IMEM upper half (word 0x%x) ...\n",
           KERNEL_ADD_LEN, SAC_TRI_IMEM_UPPER_WORD);
    if (load_kernel(kernel_add, KERNEL_ADD_LEN, "add", SAC_TRI_IMEM_UPPER_WORD)) {
        printf("FAIL: kernel_add IMEM readback\n");
        return 1;
    }
    asm volatile ("fence" ::: "memory");

    printf("Phase 2+3: starting Atlas (red_jal -> JALR 0x4000 -> add -> ECALL) ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t status23 = 0;
    if (wait_halted(&status23)) {
        printf("FAIL: phase 2+3 did not halt (status=0x%08x)\n", status23);
        return 1;
    }
    uint32_t reason23 = atlas_halt_reason();
    uint32_t dbg0_23  = atlas_dbg0();
    uint32_t ill_pc   = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_ILLEGAL_PC);
    uint32_t cycles   = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t instcnt  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    printf("  phase 2+3 done: status=0x%08x halt_reason=%u dbg0=%u illegal_pc=0x%08x\n",
           status23, reason23, dbg0_23, ill_pc);
    printf("  cycles=%u instcnt=%u\n", cycles, instcnt);
    if (reason23 != ATLAS_HALT_ECALL || dbg0_23 != 1) {
        /* illegal_pc is a WORD index. Lower half = [0, 0x4000),
         * upper half = [0x4000, 0x8000). Helps localize failures. */
        if (reason23 == ATLAS_HALT_ILLEGAL) {
            if (ill_pc < SAC_TRI_IMEM_UPPER_WORD) {
                printf("       illegal-halt in lower half -> phase 2 (kernel_red_jal)\n");
            } else {
                printf("       illegal-halt in upper half -> phase 3 (kernel_add)\n");
            }
        }
        printf("FAIL: phase 2+3 post-conditions (expected halt_reason=ECALL, dbg0=1)\n");
        return 1;
    }

    /* ── Verify phase 2+3 intermediates (S was written by red_jal, Z by add) ── */
    asm volatile ("fence" ::: "memory");
    int s_mism = 0;
    s_mism += verify_bank(SAC_TRI_OFF_S_B0, SAC_TRI_EXPECT_S, 0,                  "S bank0");
    s_mism += verify_bank(SAC_TRI_OFF_S_B1, SAC_TRI_EXPECT_S, SAC_TRI_BANK_ROWS,  "S bank1");
    if (s_mism) { printf("FAIL: %d DRAM mismatches in S (phase 2 output)\n", s_mism); return 1; }
    printf("  S verified (phase 2 OK).\n");

    int z_mism = 0;
    z_mism += verify_bank(SAC_TRI_OFF_Z_B0, SAC_TRI_EXPECT_Z, 0,                  "Z bank0");
    z_mism += verify_bank(SAC_TRI_OFF_Z_B1, SAC_TRI_EXPECT_Z, SAC_TRI_BANK_ROWS,  "Z bank1");
    if (z_mism) { printf("FAIL: %d DRAM mismatches in Z (phase 3 output)\n", z_mism); return 1; }
    printf("  Z verified (phase 3 OK).\n");

    /* ── Saturn interlude 2: RVV post-scale on Z -> Z'' ─────────────── */
    asm volatile ("fence" ::: "memory");
    printf("Interlude 2: Saturn RVV BF16 row-scale on Z -> Z'' ...\n");
    run_interlude(SAC_DRAM_BASE + SAC_TRI_OFF_Z_B0,  SAC_DRAM_BASE + SAC_TRI_OFF_Z_B1,
                  SAC_DRAM_BASE + SAC_TRI_OFF_Z2_B0, SAC_DRAM_BASE + SAC_TRI_OFF_Z2_B1,
                  SAC_TRI_POST_SCALE);
    asm volatile ("fence" ::: "memory");

    /* ── Verify Z'' ─────────────────────────────────────────────────── */
    int mism = 0;
    mism += verify_bank(SAC_TRI_OFF_Z2_B0, SAC_TRI_EXPECT_Z2, 0,                   "Z'' bank0");
    mism += verify_bank(SAC_TRI_OFF_Z2_B1, SAC_TRI_EXPECT_Z2, SAC_TRI_BANK_ROWS,   "Z'' bank1");

    if (mism) {
        printf("FAIL: %d DRAM mismatches in final Z''\n", mism);
        return 1;
    }

    printf("PASS: saturn_atlas_tri_phase "
           "(mul -> saturn scale -> reduce|JAL|add -> saturn scale)\n");
    return 0;
}

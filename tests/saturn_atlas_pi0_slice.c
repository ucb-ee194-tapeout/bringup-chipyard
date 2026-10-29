#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"

/*
 * pi0-reflective cooperative workload.
 *
 * Atlas Phase 0 seeds Q and K tiles in VMEM (32 int32 each). The host
 * (Saturn on Shuttle) then applies a Q15 fixed-point pairwise rotation
 * — the integer analogue of RoPE — to both tiles and writes them back.
 * Atlas Phase 1 reads the rotated tiles, produces Q'⊕K' and a running
 * XOR checksum. The host compares against a scalar golden reference
 * that replays the full seed → rotate → XOR pipeline.
 *
 * Single self-dispatching IMEM kernel:
 *   kernel:  generators/atlas-npu/baremetal/assembly/saturn_atlas_pi0.S
 *
 * VMEM layout (byte offsets from ATLAS_VMEM_BASE):
 *    0..127   Q tile   (32 × int32, Q[i] = (i+1)*113)
 *  128..255   K tile   (32 × int32, K[i] = (i+1)*97)
 *  256..383   result   (32 × int32, Q'[i] ⊕ K'[i])
 *      384    checksum (1 × int32, XOR-reduce of result)
 *
 * Target config: EE290SimConfig.
 */

#define N_TILE          32
#define N_PAIRS         (N_TILE / 2)   /* = 16 */
#define Q_BYTE_BASE     0U
#define K_BYTE_BASE     (N_TILE * 4)
#define RES_BYTE_BASE   (2 * N_TILE * 4)
#define CHK_BYTE_BASE   (3 * N_TILE * 4)

#define PHASE0_MARKER   1U
#define PHASE1_MARKER   2U
#define POLL_ITERS      200000000U

/* Atlas kernel assembled from saturn_atlas_pi0.S — 61 words. */
#define ATLAS_PROGRAM_LEN 56
static const uint32_t atlas_program[ATLAS_PROGRAM_LEN] = {
    0xC11020F3U, 0x00008463U, 0x00000013U, 0x0380006FU, 0x00000013U, 0x07100113U, 0x00000193U, 0x00000213U,
    0x02000293U, 0x00000313U, 0x00230333U, 0x0061A023U, 0x00101067U, 0x00418193U, 0x00120213U, 0xFE524BE3U,
    0x00000013U, 0x06100113U, 0x08000193U, 0x00000213U, 0x00000313U, 0x00230333U, 0x0061A023U, 0x00101067U,
    0x00418193U, 0x00120213U, 0xFE524BE3U, 0x00000013U, 0x00100093U, 0xC1009073U, 0x00000073U, 0x00000193U,
    0x08000393U, 0x10000413U, 0x00000213U, 0x02000293U, 0x00000493U, 0x0001A503U, 0x00201067U, 0x0003A583U,
    0x00201067U, 0x00B54633U, 0x00C4C4B3U, 0x00C42023U, 0x00101067U, 0x00418193U, 0x00438393U, 0x00440413U,
    0x00120213U, 0xFE5244E3U, 0x00000013U, 0x00942023U, 0x00101067U, 0x00200093U, 0xC1009073U, 0x00000073U,
};

/*
 * Q15 cos/sin tables for angles θ_i = i·π/32, i ∈ [0, 16).
 * Computed as round(32767 * trig(θ_i)) in the generator script; values
 * are referenced bit-for-bit from both Saturn's transform and the host
 * golden so results collide exactly.
 */
static const int32_t rope_cos_q15[N_PAIRS] = {
    32767, 32609, 32137, 31356, 30273, 28898, 27245, 25329,
    23170, 20787, 18204, 15446, 12539,  9512,  6393,  3212,
};
static const int32_t rope_sin_q15[N_PAIRS] = {
        0,  3212,  6393,  9512, 12539, 15446, 18204, 20787,
    23170, 25329, 27245, 28898, 30273, 31356, 32137, 32609,
};

static int stage_imem(void)
{
    for (uint32_t i = 0; i < ATLAS_PROGRAM_LEN; i++) {
        atlas_imem_write(i, atlas_program[i]);
    }
    asm volatile ("fence" ::: "memory");

    int mism = 0;
    for (uint32_t i = 0; i < ATLAS_PROGRAM_LEN; i++) {
        uint32_t got = atlas_imem_read(i);
        if (got != atlas_program[i]) {
            printf("IMEM MISMATCH[%u]: got 0x%08x exp 0x%08x\n",
                   i, got, atlas_program[i]);
            mism++;
        }
    }
    return mism;
}

static int wait_halted_with_dbg0(uint32_t expected_dbg0, const char *label)
{
    uint32_t i;
    uint32_t status = 0;

    for (i = 0; i < POLL_ITERS; i++) {
        status = atlas_status();
        if (status & ATLAS_STATUS_HALTED_BIT) break;
    }

    uint32_t dbg0   = atlas_dbg0();
    uint32_t reason = atlas_halt_reason();
    printf("  [%s] status=0x%08x halt_reason=%u dbg0=%u\n",
           label, status, reason, dbg0);

    if (!(status & ATLAS_STATUS_HALTED_BIT)) {
        printf("FAIL [%s]: Atlas did not halt (timeout)\n", label);
        return 1;
    }
    if (reason != ATLAS_HALT_ECALL) {
        printf("FAIL [%s]: halt_reason = %u, expected %u (ECALL)\n",
               label, reason, ATLAS_HALT_ECALL);
        return 1;
    }
    if (dbg0 != expected_dbg0) {
        printf("FAIL [%s]: dbg0 = %u, expected %u\n",
               label, dbg0, expected_dbg0);
        return 1;
    }
    return 0;
}

/*
 * Saturn RoPE pairwise rotation (Q15 fixed-point).
 *
 * For each feature pair (q[2i], q[2i+1]):
 *   q'[2i]   = (q[2i]·c[i] − q[2i+1]·s[i]) >> 15
 *   q'[2i+1] = (q[2i]·s[i] + q[2i+1]·c[i]) >> 15
 *
 * RVV stride-8 loads pick even/odd lanes into parallel registers; the
 * rotation math is then four stride-1 vector multiplies + an add/sub +
 * arithmetic shift right, followed by stride-8 scatter. Operating on a
 * local int32[] buffer keeps SBUS traffic as plain scalar reads/writes
 * at the boundary — RVV runs entirely against the dcache-backed stack.
 */
static void saturn_rope_rotate_rvv(int32_t *buf)
{
    size_t remaining = N_PAIRS;
    const int32_t *cp = rope_cos_q15;
    const int32_t *sp = rope_sin_q15;
    int32_t *bep = buf;          /* even-index base */
    int32_t *bop = buf + 1;      /* odd-index base  */

    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);

        vint32m1_t vqe = __riscv_vlse32_v_i32m1(bep, 8, vl);
        vint32m1_t vqo = __riscv_vlse32_v_i32m1(bop, 8, vl);
        vint32m1_t vc  = __riscv_vle32_v_i32m1(cp, vl);
        vint32m1_t vs  = __riscv_vle32_v_i32m1(sp, vl);

        vint32m1_t vqe_c = __riscv_vmul_vv_i32m1(vqe, vc, vl);
        vint32m1_t vqo_s = __riscv_vmul_vv_i32m1(vqo, vs, vl);
        vint32m1_t vqe_s = __riscv_vmul_vv_i32m1(vqe, vs, vl);
        vint32m1_t vqo_c = __riscv_vmul_vv_i32m1(vqo, vc, vl);

        vint32m1_t vqe_new = __riscv_vsra_vx_i32m1(
            __riscv_vsub_vv_i32m1(vqe_c, vqo_s, vl), 15, vl);
        vint32m1_t vqo_new = __riscv_vsra_vx_i32m1(
            __riscv_vadd_vv_i32m1(vqe_s, vqo_c, vl), 15, vl);

        __riscv_vsse32_v_i32m1(bep, 8, vqe_new, vl);
        __riscv_vsse32_v_i32m1(bop, 8, vqo_new, vl);

        remaining -= vl;
        cp  += vl;
        sp  += vl;
        bep += vl * 2;
        bop += vl * 2;
    }
}

/*
 * Scalar reference rotation — bit-identical semantics to the RVV version.
 * Used by golden replay so saturn output and golden match exactly.
 */
static void rope_rotate_scalar(int32_t *buf)
{
    for (size_t p = 0; p < N_PAIRS; p++) {
        int32_t qe = buf[2*p];
        int32_t qo = buf[2*p + 1];
        int32_t c  = rope_cos_q15[p];
        int32_t s  = rope_sin_q15[p];
        buf[2*p]     = (int32_t)(((int64_t)qe * c - (int64_t)qo * s) >> 15);
        buf[2*p + 1] = (int32_t)(((int64_t)qe * s + (int64_t)qo * c) >> 15);
    }
}

static int read_tile(int32_t *dst, uint32_t vmem_byte_base)
{
    uintptr_t addr = ATLAS_VMEM_BASE + vmem_byte_base;
    size_t remaining = N_TILE;
    int32_t *bp = dst;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vint32m1_t v = __riscv_vle32_v_i32m1((int32_t *)addr, vl);
        __riscv_vse32_v_i32m1(bp, v, vl);
        remaining -= vl;
        addr     += vl * 4;
        bp       += vl;
    }
    return 0;
}

static void write_tile(const int32_t *src, uint32_t vmem_byte_base)
{
    uintptr_t addr = ATLAS_VMEM_BASE + vmem_byte_base;
    size_t remaining = N_TILE;
    const int32_t *bp = src;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vint32m1_t v = __riscv_vle32_v_i32m1(bp, vl);
        __riscv_vse32_v_i32m1((int32_t *)addr, v, vl);
        remaining -= vl;
        addr     += vl * 4;
        bp       += vl;
    }
    asm volatile ("fence" ::: "memory");
}

int main(void)
{
    printf("Saturn <-> Atlas pi0-reflective slice\n");

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);
    for (uint32_t i = 0; i < 4 * N_TILE; i++) {
        atlas_write32(ATLAS_VMEM_BASE + i * 4, 0);
    }
    asm volatile ("fence" ::: "memory");

    printf("Staging kernel (%u words) ...\n", ATLAS_PROGRAM_LEN);
    if (stage_imem()) {
        printf("FAIL: IMEM readback mismatch\n");
        return 1;
    }

    /* ── Phase 0: Atlas seeds Q and K ──────────────────────────────────── */
    printf("Starting Atlas (phase 0: seed Q and K) ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    if (wait_halted_with_dbg0(PHASE0_MARKER, "phase 0")) return 1;

    int32_t q_buf[N_TILE];
    int32_t k_buf[N_TILE];
    read_tile(q_buf, Q_BYTE_BASE);
    read_tile(k_buf, K_BYTE_BASE);

    int fail = 0;
    for (int32_t i = 0; i < N_TILE; i++) {
        int32_t exp_q = (i + 1) * 113;
        int32_t exp_k = (i + 1) * 97;
        if (q_buf[i] != exp_q) {
            printf("  Q SEED MISMATCH[%d]: got %d exp %d\n", i, q_buf[i], exp_q);
            fail++;
        }
        if (k_buf[i] != exp_k) {
            printf("  K SEED MISMATCH[%d]: got %d exp %d\n", i, k_buf[i], exp_k);
            fail++;
        }
    }
    if (fail) {
        printf("FAIL: phase-0 seed verification\n");
        return 1;
    }
    printf("PASS: phase 0 seed verified\n");

    /* ── Saturn RoPE between phases ────────────────────────────────────── */
    printf("Saturn RVV RoPE rotation (Q15 fixed-point) ...\n");
    saturn_rope_rotate_rvv(q_buf);
    saturn_rope_rotate_rvv(k_buf);
    write_tile(q_buf, Q_BYTE_BASE);
    write_tile(k_buf, K_BYTE_BASE);

    /* Scalar readback sanity check. */
    for (int32_t i = 0; i < N_TILE; i++) {
        int32_t got_q = (int32_t)atlas_read32(ATLAS_VMEM_BASE + Q_BYTE_BASE + i * 4);
        int32_t got_k = (int32_t)atlas_read32(ATLAS_VMEM_BASE + K_BYTE_BASE + i * 4);
        if (got_q != q_buf[i] || got_k != k_buf[i]) {
            printf("  ROPE WRITEBACK MISMATCH[%d]: vmem_q=%d buf_q=%d vmem_k=%d buf_k=%d\n",
                   i, got_q, q_buf[i], got_k, k_buf[i]);
            fail++;
        }
    }
    if (fail) {
        printf("FAIL: RoPE writeback did not land\n");
        return 1;
    }
    printf("PASS: RoPE writeback coherent\n");

    /* ── Phase 1: Atlas consumes rotated Q', K' ────────────────────────── */
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_set_dbg1(1);
    asm volatile ("fence" ::: "memory");

    printf("Restarting Atlas (phase 1: XOR + checksum) ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    if (wait_halted_with_dbg0(PHASE1_MARKER, "phase 1")) return 1;

    uint32_t cycles  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t instcnt = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    printf("  cycles=%u instcnt=%u\n", cycles, instcnt);

    /* ── Golden reference replay ───────────────────────────────────────── */
    int32_t golden_q[N_TILE];
    int32_t golden_k[N_TILE];
    for (int32_t i = 0; i < N_TILE; i++) {
        golden_q[i] = (i + 1) * 113;
        golden_k[i] = (i + 1) * 97;
    }
    rope_rotate_scalar(golden_q);
    rope_rotate_scalar(golden_k);

    uint32_t golden_checksum = 0;
    for (int32_t i = 0; i < N_TILE; i++) {
        uint32_t want = (uint32_t)golden_q[i] ^ (uint32_t)golden_k[i];
        uint32_t got  = atlas_read32(ATLAS_VMEM_BASE + RES_BYTE_BASE + i * 4);
        if (got != want) {
            printf("  RESULT MISMATCH[%d]: got 0x%08x exp 0x%08x (Q'=%d K'=%d)\n",
                   i, got, want, golden_q[i], golden_k[i]);
            fail++;
        }
        golden_checksum ^= want;
    }
    uint32_t got_checksum = atlas_read32(ATLAS_VMEM_BASE + CHK_BYTE_BASE);
    if (got_checksum != golden_checksum) {
        printf("  CHECKSUM MISMATCH: got 0x%08x exp 0x%08x\n",
               got_checksum, golden_checksum);
        fail++;
    }
    if (fail) {
        printf("FAIL: phase-1 result did not match golden (%d errors)\n", fail);
        return 1;
    }

    printf("  golden_checksum=0x%08x\n", golden_checksum);
    printf("PASS: pi0-reflective slice\n");
    return 0;
}

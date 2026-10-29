#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"

/*
 * Saturn ↔ Atlas ECALL pause / host-work / resume handshake.
 *
 * Kernel: generators/atlas-npu/baremetal/assembly/saturn_atlas_two_phase.S
 * (assembled via assembler.py; array below is the verbatim output.)
 *
 * Protocol:
 *   1. Stage the kernel into IMEM.
 *   2. Clear dbg0/dbg1; execControl = 1.
 *   3. Atlas runs phase 0 (selected by dbg1 == 0): seeds VMEM[i*4] = 0xA0000000 | i
 *      for i in 0..15, writes dbg0 = 1, ECALLs.
 *   4. Host polls status.halted, verifies halt_reason == ECALL and dbg0 == 1.
 *   5. Host reads VMEM via Saturn RVV, verifies seed, adds 0x10000 to each word,
 *      writes back.
 *   6. Host clears dbg0, sets dbg1 = 1, writes execControl = 1.
 *   7. Atlas runs phase 1: reads VMEM, verifies == 0xA0010000 | i, writes dbg0 = 2,
 *      ECALLs.
 *   8. Host polls status.halted, verifies halt_reason == ECALL and dbg0 == 2.
 *
 * Target config: EE290SimConfig.
 */

#define N_WORDS             16
#define SEED_BASE           0xA0000000U
#define TRANSFORM_DELTA     0x00010000U
#define EXPECTED_BASE       (SEED_BASE + TRANSFORM_DELTA)   /* 0xA0010000 */

#define PHASE0_MARKER       1U
#define PHASE1_MARKER       2U

#define POLL_ITERS          200000000U

#define ATLAS_PROGRAM_LEN   38
static const uint32_t atlas_program[ATLAS_PROGRAM_LEN] = {
    0xC11020F3U, 0x00008463U, 0x00000013U, 0x0200006FU, 0x00000013U, 0xA0000137U, 0x00000193U, 0x00000213U,
    0x01000293U, 0x00416333U, 0x0061A023U, 0x00101067U, 0x00418193U, 0x00120213U, 0xFE524BE3U, 0x00000013U,
    0x00100093U, 0xC1009073U, 0x00000073U, 0xA0010137U, 0x00000193U, 0x00000213U, 0x01000293U, 0x0001A303U,
    0x00201067U, 0x004163B3U, 0x00731963U, 0x00000013U, 0x00418193U, 0x00120213U, 0xFE5249E3U, 0x00000013U,
    0x00200093U, 0xC1009073U, 0x00000073U, 0x06420413U, 0xC1041073U, 0x00000073U,
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
    uint32_t dbg0 = 0;
    uint32_t status = 0;

    for (i = 0; i < POLL_ITERS; i++) {
        status = atlas_status();
        if (status & ATLAS_STATUS_HALTED_BIT) break;
    }

    dbg0 = atlas_dbg0();
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
 * Verify the phase-0 seed pattern, then transform each word in-place with
 * Saturn RVV: VMEM[i*4] = seed + TRANSFORM_DELTA.
 */
static int saturn_transform_vmem(void)
{
    uintptr_t vmem = ATLAS_VMEM_BASE;

    /* Verify phase-0 seed, buffered through a local array so we can use a
     * pure stride-1 vector load from VMEM.                                  */
    uint32_t buf[N_WORDS];

    size_t remaining = N_WORDS;
    uintptr_t vp = vmem;
    uint32_t *bp = buf;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vuint32m1_t v = __riscv_vle32_v_u32m1((uint32_t *)vp, vl);
        __riscv_vse32_v_u32m1(bp, v, vl);
        remaining -= vl;
        vp       += vl * 4;
        bp       += vl;
    }

    int fail = 0;
    for (uint32_t i = 0; i < N_WORDS; i++) {
        uint32_t exp = SEED_BASE | i;
        if (buf[i] != exp) {
            printf("  SEED MISMATCH[%u]: got 0x%08x exp 0x%08x\n",
                   i, buf[i], exp);
            fail++;
        }
    }
    if (fail) return fail;

    /* Transform in place: add TRANSFORM_DELTA via RVV vadd.vx. */
    remaining = N_WORDS;
    vp = vmem;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vuint32m1_t v = __riscv_vle32_v_u32m1((uint32_t *)vp, vl);
        v = __riscv_vadd_vx_u32m1(v, TRANSFORM_DELTA, vl);
        __riscv_vse32_v_u32m1((uint32_t *)vp, v, vl);
        remaining -= vl;
        vp       += vl * 4;
    }
    asm volatile ("fence" ::: "memory");

    /* Scalar readback to confirm transform landed before we hand back. */
    for (uint32_t i = 0; i < N_WORDS; i++) {
        uint32_t got = atlas_read32(vmem + i * 4);
        uint32_t exp = EXPECTED_BASE | i;
        if (got != exp) {
            printf("  TRANSFORM MISMATCH[%u]: got 0x%08x exp 0x%08x\n",
                   i, got, exp);
            fail++;
        }
    }
    return fail;
}

int main(void)
{
    printf("Saturn <-> Atlas ECALL pause/resume\n");

    /* Make sure Atlas is stopped and counters are zeroed before we begin. */
    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);
    asm volatile ("fence" ::: "memory");

    printf("Staging kernel (%u words) ...\n", ATLAS_PROGRAM_LEN);
    if (stage_imem()) {
        printf("FAIL: IMEM readback mismatch\n");
        return 1;
    }

    /* ── Phase 0 ───────────────────────────────────────────────────────── */
    printf("Starting Atlas (phase 0) ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    if (wait_halted_with_dbg0(PHASE0_MARKER, "phase 0")) {
        return 1;
    }
    printf("PASS: phase 0 (seed done)\n");

    /* ── Host (Saturn) work ───────────────────────────────────────────── */
    printf("Saturn RVV transform on VMEM ...\n");
    if (saturn_transform_vmem()) {
        printf("FAIL: Saturn-side verify/transform\n");
        return 1;
    }
    printf("PASS: Saturn RVV transform\n");

    /* ── Phase 1 ───────────────────────────────────────────────────────── */
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_set_dbg1(1);
    asm volatile ("fence" ::: "memory");

    printf("Restarting Atlas (phase 1) ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    if (wait_halted_with_dbg0(PHASE1_MARKER, "phase 1")) {
        return 1;
    }

    uint32_t cycles  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t instcnt = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    printf("  cycles=%u instcnt=%u\n", cycles, instcnt);

    printf("PASS: Saturn <-> Atlas ECALL pause/resume\n");
    return 0;
}

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "atlas_mmio.h"

/*
 * Software-buffered dual-kernel IMEM load, Option B (restart).
 *
 * Atlas runs kernel A until it ECALLs. Once A has halted, the host stages
 * kernel B into IMEM[0..) (overwriting the original trampoline), then
 * writes execControl=1 again: PC resets to 0 and Atlas runs B from scratch.
 *
 * Unlike the end-JAL and lower-first variants, this one does NOT exercise
 * concurrent write/execute (B is staged while Atlas is halted). It
 * validates the restart-dispatch mechanism instead — a useful alternative
 * when A's work cannot be timed to a JAL hand-off.
 *
 * Initial layout:
 *   IMEM[0             .. 4*LEN_TRAMP)  = trampoline (JALR to 0x10000)
 *   IMEM[0x10000       .. +4*LEN_A)     = kernel A (ends with ECALL, dbg0=1)
 *
 * Target config: EE290SimConfig.
 */

#define LEN_TRAMP 4
static const uint32_t kernel_trampoline[LEN_TRAMP] = {
    0x000040b7, 0x00008067, 0x00000013, 0x00000013,
};

#define LEN_A 15
static const uint32_t kernel_a[LEN_A] = {
    0x00001537, 0x38850513, 0x00000593, 0x00158593,
    0xfea5cfe3, 0x00000013, 0x00000013, 0xaaaab137,
    0xaaa10113, 0x00000193, 0x0021a023, 0x00201067,
    0x00100093, 0xc1009073, 0x00000073,
};

#define LEN_B 15
static const uint32_t kernel_b[LEN_B] = {
    0x00000193, 0x0001a203, 0x00201067, 0xaaaab2b7,
    0xaaa28293, 0x00521663, 0x00000013, 0x00000013,
    0x00200093, 0xc1009073, 0x00000073, 0x000010b7,
    0xbad08093, 0xc1009073, 0x00000073,
};

#define A_WORD_OFFSET       (ATLAS_IMEM_HALF_BYTES / 4)   /* = 16384 */
#define POLL_ITERS          200000000U

static int stage_kernel(uint32_t word_base, const uint32_t *words, uint32_t n,
                        const char *label)
{
    for (uint32_t i = 0; i < n; i++) {
        atlas_imem_write(word_base + i, words[i]);
    }
    asm volatile ("fence" ::: "memory");

    int mism = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t got = atlas_imem_read(word_base + i);
        if (got != words[i]) {
            printf("IMEM MISMATCH[%s word %u]: got 0x%08x exp 0x%08x\n",
                   label, word_base + i, got, words[i]);
            mism++;
        }
    }
    return mism;
}

static int wait_halted(uint32_t *status_out, uint32_t *dbg0_out)
{
    uint32_t i;
    uint32_t status = 0;
    for (i = 0; i < POLL_ITERS; i++) {
        status = atlas_status();
        if (status & ATLAS_STATUS_HALTED_BIT) break;
    }
    *status_out = status;
    *dbg0_out = atlas_dbg0();
    return (status & ATLAS_STATUS_HALTED_BIT) ? 0 : 1;
}

int main(void)
{
    printf("Buffered dual-kernel IMEM load (optB restart)\n");

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);
    atlas_write32(ATLAS_VMEM_BASE, 0);
    asm volatile ("fence" ::: "memory");

    printf("Staging trampoline at IMEM[0..%u) + A at IMEM[0x10000..) ...\n",
           LEN_TRAMP);
    if (stage_kernel(0, kernel_trampoline, LEN_TRAMP, "trampoline") ||
        stage_kernel(A_WORD_OFFSET, kernel_a, LEN_A, "A")) {
        return 1;
    }

    /* ── Phase A ───────────────────────────────────────────────────────── */
    printf("Starting Atlas (trampoline → A → ECALL) ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t status_a = 0, dbg0_a = 0;
    if (wait_halted(&status_a, &dbg0_a)) {
        printf("FAIL: A did not halt\n");
        return 1;
    }
    uint32_t reason_a = atlas_halt_reason();
    uint32_t marker_a = atlas_read32(ATLAS_VMEM_BASE);
    printf("  [A] status=0x%08x halt_reason=%u dbg0=%u marker=0x%08x\n",
           status_a, reason_a, dbg0_a, marker_a);
    if (reason_a != ATLAS_HALT_ECALL || dbg0_a != 1 || marker_a != 0xAAAAAAAAU) {
        printf("FAIL: A post-conditions not met\n");
        return 1;
    }

    /* ── Stage B and restart ──────────────────────────────────────────── */
    printf("Staging kernel B at IMEM[0..) (overwriting trampoline) ...\n");
    if (stage_kernel(0, kernel_b, LEN_B, "B")) {
        return 1;
    }

    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    asm volatile ("fence" ::: "memory");

    printf("Restarting Atlas (PC resets to 0 → B) ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    uint32_t status_b = 0, dbg0_b = 0;
    if (wait_halted(&status_b, &dbg0_b)) {
        printf("FAIL: B did not halt\n");
        return 1;
    }
    uint32_t reason_b = atlas_halt_reason();
    uint32_t cycles   = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t instcnt  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    printf("  [B] status=0x%08x halt_reason=%u dbg0=%u\n",
           status_b, reason_b, dbg0_b);
    printf("  cycles=%u instcnt=%u\n", cycles, instcnt);

    if (reason_b != ATLAS_HALT_ECALL || dbg0_b != 2) {
        printf("FAIL: B post-conditions not met\n");
        return 1;
    }

    printf("PASS: buffered dual-kernel (optB restart)\n");
    return 0;
}

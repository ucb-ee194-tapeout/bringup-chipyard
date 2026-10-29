#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "atlas_mmio.h"

/*
 * Software-buffered dual-kernel IMEM load, lower-first
 * (reversed halves of the end-JAL variant).
 *
 * Initial layout:
 *   IMEM[0             .. 4*LEN_TRAMP)  = trampoline (JALR to 0x10000)
 *   IMEM[0x10000       .. +4*LEN_A)     = kernel A (ends with JALR to 0)
 *
 * Atlas starts at PC=0, runs the trampoline, lands in A. While A's ~20k
 * busy loop is running, the host overwrites IMEM[0..0x10000) with
 * kernel B. A's terminal JALR targets byte 0 — which, by the time Atlas
 * fetches it, now holds B's first instruction.
 *
 * This flips the failure mode relative to the end-JAL variant: if B
 * staging races badly, Atlas lands on the still-present trampoline and
 * loops forever.
 *
 * Target config: EE290SimConfig.
 */

#define LEN_TRAMP 4
static const uint32_t kernel_trampoline[LEN_TRAMP] = {
    0x000040b7, 0x00008067, 0x00000013, 0x00000013,
};

#define LEN_A 16
static const uint32_t kernel_a[LEN_A] = {
    /* LI x10, 500000  — busy-loop iterations (bumped from 5000) so the
     * host has time to stage B concurrently before A's terminal JALR. */
    0x0007a537, 0x12050513, 0x00000593, 0x00158593,
    0xfea5cfe3, 0x00000013, 0x00000013, 0xaaaab137,
    0xaaa10113, 0x00000193, 0x0021a023, 0x00201067,
    0x00000093, 0x00008067, 0x00000013, 0x00000013,
};

#define LEN_B 15
static const uint32_t kernel_b[LEN_B] = {
    0x00000193, 0x0001a203, 0x00201067, 0xaaaab2b7,
    0xaaa28293, 0x00521663, 0x00000013, 0x00000013,
    0x00200093, 0xc1009073, 0x00000073, 0x000010b7,
    0xbad08093, 0xc1009073, 0x00000073,
};

#define A_WORD_OFFSET       (ATLAS_IMEM_HALF_BYTES / 4)   /* = 16384 */
#define EXPECTED_DBG0       2U
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

int main(void)
{
    printf("Buffered dual-kernel IMEM load (lower-first)\n");

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

    printf("Starting Atlas (trampoline → A busy loop, B staged concurrently) ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    /* Concurrent B staging at IMEM[0..), overwriting the trampoline.
     * NOTE: do NOT printf between atlas_start() and stage_kernel(B) —
     * baremetal UART output eats thousands of target cycles, costing the
     * race. */
    if (stage_kernel(0, kernel_b, LEN_B, "B")) {
        return 1;
    }

    uint32_t i;
    uint32_t status = 0;
    for (i = 0; i < POLL_ITERS; i++) {
        status = atlas_status();
        if (status & ATLAS_STATUS_HALTED_BIT) break;
    }

    uint32_t dbg0     = atlas_dbg0();
    uint32_t reason   = atlas_halt_reason();
    uint32_t cycles   = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t instcnt  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    uint32_t illegal  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_ILLEGAL_PC);
    uint32_t marker   = atlas_read32(ATLAS_VMEM_BASE);

    printf("  status=0x%08x halt_reason=%u dbg0=%u marker=0x%08x\n",
           status, reason, dbg0, marker);
    printf("  cycles=%u instcnt=%u illegal_pc=0x%08x\n", cycles, instcnt, illegal);

    /* Post-mortem: read back IMEM to see what Atlas actually fetched. */
    if (reason == ATLAS_HALT_ILLEGAL) {
        uint32_t pc_word_idx = illegal;
        uint32_t fetched = atlas_imem_read(pc_word_idx);
        printf("  IMEM[pc_word=%u] = 0x%08x (pc_byte=0x%08x)\n",
               pc_word_idx, fetched, pc_word_idx * 4);
        printf("  IMEM[0..7] =");
        for (uint32_t i = 0; i < 8; i++) {
            printf(" %08x", atlas_imem_read(i));
        }
        printf("\n  IMEM[16384..16391] =");
        for (uint32_t i = 16384; i < 16392; i++) {
            printf(" %08x", atlas_imem_read(i));
        }
        printf("\n");
    }

    if (!(status & ATLAS_STATUS_HALTED_BIT)) {
        printf("FAIL: Atlas did not halt — B staging probably lost the race\n");
        return 1;
    }
    if (reason != ATLAS_HALT_ECALL) {
        printf("FAIL: halt_reason = %u, expected %u (ECALL)\n", reason, ATLAS_HALT_ECALL);
        return 1;
    }
    if (marker != 0xAAAAAAAAU) {
        printf("FAIL: VMEM[0] marker = 0x%08x, expected 0xAAAAAAAA\n", marker);
        return 1;
    }
    if (dbg0 != EXPECTED_DBG0) {
        printf("FAIL: dbg0 = %u, expected %u\n", dbg0, EXPECTED_DBG0);
        return 1;
    }

    printf("PASS: buffered dual-kernel (lower-first)\n");
    return 0;
}

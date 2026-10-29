#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "atlas_mmio.h"

/*
 * Software-buffered dual-kernel IMEM load, Option A (end-JAL).
 *
 * Atlas runs two kernels back-to-back with NO halt between them. The
 * transition is an in-IMEM absolute JALR from A → B. While Atlas is
 * executing kernel A's busy loop, the host (Shuttle+Saturn) concurrently
 * stages kernel B into the upper half of IMEM (byte 0x10000..). This
 * validates concurrent write/execute of Atlas's 1R1W SyncReadMem IMEM.
 *
 * Layout:
 *   IMEM[0            .. 4*LEN_A) = kernel A  (ends with JALR to 0x10000)
 *   IMEM[0x10000      .. +4*LEN_B) = kernel B (staged while A runs)
 *
 * Kernels: generators/atlas-npu/baremetal/assembly/sab_a_to_high.S
 *          generators/atlas-npu/baremetal/assembly/sab_b.S
 *
 * Target config: EE290SimConfig.
 */

#define LEN_A 16
static const uint32_t kernel_a[LEN_A] = {
    /* LI x10, 500000  — busy-loop iterations (bumped from 5000 to give
     * host MMIO time to commit stage_kernel(B) writes before A's terminal
     * JALR. Encoded as LUI x10, 0x7A + ADDI x10, x10, 0x120; LUI's lo12
     * uses a positive imm here. */
    0x0007a537, 0x12050513, 0x00000593, 0x00158593,
    0xfea5cfe3, 0x00000013, 0x00000013, 0xaaaab137,
    0xaaa10113, 0x00000193, 0x0021a023, 0x00201067,
    0x000040b7, 0x00008067, 0x00000013, 0x00000013,
};

#define LEN_B 15
static const uint32_t kernel_b[LEN_B] = {
    0x00000193, 0x0001a203, 0x00201067, 0xaaaab2b7,
    0xaaa28293, 0x00521663, 0x00000013, 0x00000013,
    0x00200093, 0xc1009073, 0x00000073, 0x000010b7,
    0xbad08093, 0xc1009073, 0x00000073,
};

#define B_WORD_OFFSET       (ATLAS_IMEM_HALF_BYTES / 4)   /* = 0x4000 = 16384 */
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
    printf("Buffered dual-kernel IMEM load (optA end-JAL)\n");

    atlas_stop();
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0, 0);
    atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, 0);
    atlas_write32(ATLAS_VMEM_BASE, 0);                 /* clear marker */
    asm volatile ("fence" ::: "memory");

    printf("Staging kernel A at IMEM[0..%u) ...\n", LEN_A);
    if (stage_kernel(0, kernel_a, LEN_A, "A")) {
        printf("FAIL: A staging\n");
        return 1;
    }

    printf("Starting Atlas (A runs busy loop, B staged concurrently) ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    /* Concurrent B staging — the whole point of this test.
     * Host has from atlas_start() to A's terminal JALR to push B into
     * IMEM[0x10000..]. NOTE: do NOT add a printf between atlas_start()
     * and stage_kernel(B) — baremetal UART output is synchronous and
     * easily costs thousands of target cycles, which A burns through
     * before the host's MMIO writes for B reach IMEM. */
    if (stage_kernel(B_WORD_OFFSET, kernel_b, LEN_B, "B")) {
        printf("FAIL: B staging\n");
        return 1;
    }

    /* Wait for B to ECALL. */
    uint32_t i;
    uint32_t status = 0;
    for (i = 0; i < POLL_ITERS; i++) {
        status = atlas_status();
        if (status & ATLAS_STATUS_HALTED_BIT) break;
    }

    uint32_t dbg0    = atlas_dbg0();
    uint32_t reason  = atlas_halt_reason();
    uint32_t cycles  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t instcnt = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    uint32_t marker  = atlas_read32(ATLAS_VMEM_BASE);

    printf("  status=0x%08x halt_reason=%u dbg0=%u marker=0x%08x\n",
           status, reason, dbg0, marker);
    printf("  cycles=%u instcnt=%u\n", cycles, instcnt);

    if (!(status & ATLAS_STATUS_HALTED_BIT)) {
        printf("FAIL: Atlas did not halt (timeout) — B staging probably lost the race\n");
        return 1;
    }
    if (reason != ATLAS_HALT_ECALL) {
        printf("FAIL: halt_reason = %u, expected %u (ECALL)\n", reason, ATLAS_HALT_ECALL);
        return 1;
    }
    if (marker != 0xAAAAAAAAU) {
        printf("FAIL: VMEM[0] marker = 0x%08x, expected 0xAAAAAAAA (A never ran)\n", marker);
        return 1;
    }
    if (dbg0 != EXPECTED_DBG0) {
        printf("FAIL: dbg0 = %u, expected %u (B never completed)\n", dbg0, EXPECTED_DBG0);
        return 1;
    }

    printf("PASS: buffered dual-kernel (optA end-JAL)\n");
    return 0;
}

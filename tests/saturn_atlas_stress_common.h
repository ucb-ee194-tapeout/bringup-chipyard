#ifndef SATURN_ATLAS_STRESS_COMMON_H
#define SATURN_ATLAS_STRESS_COMMON_H

/*
 * Shared scaffolding for the Saturn↔Atlas memory stress test suite
 * (T1-T5).  Lifts IMEM staging and halt-poll helpers out of
 * saturn_atlas_pi0_slice.c so each new cooperative driver doesn't
 * re-implement them.
 *
 * Default VMEM geometry: 1.5 MiB, 6 banks of 256 KiB each.
 * Bank N starts at ATLAS_VMEM_BASE + N * SAS_BANK_BYTES.
 */

#include <stdio.h>
#include <stdint.h>

#include "atlas_mmio.h"

#define SAS_BANK_BYTES        ATLAS_VMEM_BANK_BYTES
#define SAS_BANK_BASE(n)      (ATLAS_VMEM_BASE + (uintptr_t)(n) * SAS_BANK_BYTES)
#define SAS_DRAM_BASE         ((uint64_t)ATLAS_DRAM_BASE)  /* FPGA DRAM window */

#define SAS_POLL_ITERS        200000000U

static inline int sas_stage_imem_from_array(const uint32_t *prog,
                                            uint32_t len)
{
    for (uint32_t i = 0; i < len; i++) {
        atlas_imem_write(i, prog[i]);
    }
    asm volatile ("fence" ::: "memory");

    int mism = 0;
    for (uint32_t i = 0; i < len; i++) {
        uint32_t got = atlas_imem_read(i);
        if (got != prog[i]) {
            printf("IMEM MISMATCH[%u]: got 0x%08x exp 0x%08x\n",
                   i, got, prog[i]);
            mism++;
            if (mism > 4) {
                printf("  ... further IMEM mismatches suppressed\n");
                break;
            }
        }
    }
    return mism;
}

static inline int sas_wait_halted_with_dbg0(uint32_t expected_dbg0,
                                            const char *label)
{
    uint32_t i;
    uint32_t status = 0;

    for (i = 0; i < SAS_POLL_ITERS; i++) {
        status = atlas_status();
        if (status & ATLAS_STATUS_HALTED_BIT) break;
    }

    uint32_t dbg0   = atlas_dbg0();
    uint32_t dbg1   = atlas_dbg1();
    uint32_t reason = atlas_halt_reason();
    printf("  [%s] status=0x%08x halt_reason=%u dbg0=%u dbg1=%u\n",
           label, status, reason, dbg0, dbg1);

    if (!(status & ATLAS_STATUS_HALTED_BIT)) {
        printf("FAIL [%s]: Atlas did not halt (timeout) — last dbg0=%u dbg1=%u "
               "(see kernel for phase markers)\n", label, dbg0, dbg1);
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

#endif /* SATURN_ATLAS_STRESS_COMMON_H */

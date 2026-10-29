#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include <riscv_vector.h>

#include "atlas_mmio.h"

/*
 * VMEM direct-access smoke test.
 *
 * Validates that a Shuttle+Saturn host can reach the Atlas VMEM region
 * (ATLAS_VMEM_BASE = 0x20000000) over both scalar MMIO and RVV vector
 * stores/loads.
 *
 * Target config: EE290SimConfig (Shuttle + Saturn + WithAtlasTile()).
 */

#define N_SCALAR_WORDS 16
#define N_VECTOR_WORDS 64

static uint32_t src[N_VECTOR_WORDS];
static uint32_t dst[N_VECTOR_WORDS];

static int scalar_round_trip(void)
{
    const uintptr_t vmem = ATLAS_VMEM_BASE;
    int fail = 0;

    for (uint32_t i = 0; i < N_SCALAR_WORDS; i++) {
        atlas_write32(vmem + i * 4, 0xDEAD0000U | i);
    }
    asm volatile ("fence" ::: "memory");

    for (uint32_t i = 0; i < N_SCALAR_WORDS; i++) {
        uint32_t got = atlas_read32(vmem + i * 4);
        uint32_t exp = 0xDEAD0000U | i;
        if (got != exp) {
            printf("SCALAR MISMATCH[%u]: got 0x%08x exp 0x%08x\n", i, got, exp);
            fail++;
        }
    }
    return fail;
}

static int vector_round_trip(void)
{
    /* Offset into VMEM so we don't overlap the scalar-test region. */
    const uintptr_t vmem = ATLAS_VMEM_BASE + 0x1000;
    int fail = 0;

    for (uint32_t i = 0; i < N_VECTOR_WORDS; i++) {
        src[i] = 0xC0DE0000U | i;
        dst[i] = 0;
    }

    /* Stride-1 RVV store: src -> VMEM */
    size_t remaining = N_VECTOR_WORDS;
    uint32_t *sp = src;
    uintptr_t vp = vmem;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vuint32m1_t v = __riscv_vle32_v_u32m1(sp, vl);
        __riscv_vse32_v_u32m1((uint32_t *)vp, v, vl);
        remaining -= vl;
        sp       += vl;
        vp       += vl * 4;
    }
    asm volatile ("fence" ::: "memory");

    /* Stride-1 RVV load: VMEM -> dst */
    remaining = N_VECTOR_WORDS;
    uint32_t *dp = dst;
    vp = vmem;
    while (remaining > 0) {
        size_t vl = __riscv_vsetvl_e32m1(remaining);
        vuint32m1_t v = __riscv_vle32_v_u32m1((uint32_t *)vp, vl);
        __riscv_vse32_v_u32m1(dp, v, vl);
        remaining -= vl;
        dp       += vl;
        vp       += vl * 4;
    }

    for (uint32_t i = 0; i < N_VECTOR_WORDS; i++) {
        if (dst[i] != src[i]) {
            printf("VECTOR MISMATCH[%u]: got 0x%08x exp 0x%08x\n",
                   i, dst[i], src[i]);
            fail++;
        }
    }
    return fail;
}

int main(void)
{
    printf("VMEM direct-access smoke test\n");

    int scalar_fail = scalar_round_trip();
    if (scalar_fail) {
        printf("FAIL: scalar round-trip (%d mismatches)\n", scalar_fail);
        return 1;
    }
    printf("PASS: scalar round-trip (%u words)\n", (unsigned)N_SCALAR_WORDS);

    int vector_fail = vector_round_trip();
    if (vector_fail) {
        printf("FAIL: vector round-trip (%d mismatches)\n", vector_fail);
        return 1;
    }
    printf("PASS: vector round-trip (%u words)\n", (unsigned)N_VECTOR_WORDS);

    printf("PASS: VMEM direct-access smoke\n");
    return 0;
}

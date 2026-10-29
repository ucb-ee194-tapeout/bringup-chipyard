#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "atlas_mmio.h"

#define ATLAS_PROGRAM_LEN 103

static const uint32_t atlas_program[ATLAS_PROGRAM_LEN] = {
    0x00100E13,
    0x02A00F93,
    0x02A00F13,
    0x0DEF9063,
    0x00000013,
    0x00000013,
    0x001E0E13,
    0x01E00093,
    0x04600113,
    0x00208FB3,
    0x06400F13,
    0x0BEF9863,
    0x00000013,
    0x00000013,
    0x001E0E13,
    0x40110FB3,
    0x02800F13,
    0x0BEF9263,
    0x00000013,
    0x00000013,
    0x001E0E13,
    0x0FF00093,
    0x00F00113,
    0x0020FFB3,
    0x00F00F13,
    0x09EF9A63,
    0x00000013,
    0x00000013,
    0x001E0E13,
    0x0A000093,
    0x00500113,
    0x0020EFB3,
    0x0A500F13,
    0x09EF9263,
    0x00000013,
    0x00000013,
    0x001E0E13,
    0x0FF00093,
    0x00F00113,
    0x0020CFB3,
    0x0F000F13,
    0x07EF9A63,
    0x00000013,
    0x00000013,
    0x001E0E13,
    0x00100093,
    0x00400113,
    0x00209FB3,
    0x01000F13,
    0x07EF9263,
    0x00000013,
    0x00000013,
    0x001E0E13,
    0x10000093,
    0x00400113,
    0x0020DFB3,
    0x01000F13,
    0x05EF9A63,
    0x00000013,
    0x00000013,
    0x001E0E13,
    0xFFB00093,
    0x00A00113,
    0x0020AFB3,
    0x00100F13,
    0x05EF9263,
    0x00000013,
    0x00000013,
    0x001E0E13,
    0x12345FB7,
    0x12345F37,
    0x03EF9C63,
    0x00000013,
    0x00000013,
    0x001E0E13,
    0xABCDEFB7,
    0x123F8F93,
    0xABCDEF37,
    0x123F0F13,
    0x03EF9463,
    0x00000013,
    0x00000013,
    0x001E0E13,
    0x06400093,
    0xFE208F93,
    0x04600F13,
    0x01EF9D63,
    0x00000013,
    0x00000013,
    0x001E0E13,
    0x3E700013,
    0x00001863,
    0x00000013,
    0x00000013,
    0x00100093,
    0xC1009073,
    0x0000006F,
    0x00000013,
    0x00000013,
    0xC10E1073,
    0x0000006F,
    0x00000013,
    0x00000013
};

int main(void)
{
    uint32_t i;
    int fail = 0;

    printf("Writing program to ATLAS IMEM (%u words) ...\n", ATLAS_PROGRAM_LEN);
    for (i = 0; i < ATLAS_PROGRAM_LEN; i++) {
        atlas_imem_write(i, atlas_program[i]);
    }
    asm volatile ("fence" ::: "memory");

    printf("Reading back IMEM and verifying ...\n");
    for (i = 0; i < ATLAS_PROGRAM_LEN; i++) {
        uint32_t got = atlas_imem_read(i);
        if (got != atlas_program[i]) {
            printf("MISMATCH word[%u]: expected 0x%08x, got 0x%08x\n",
                   i, atlas_program[i], got);
            fail++;
        }
    }

    if (fail) {
        printf("FAIL: %d IMEM mismatches\n", fail);
        return 1;
    }

    printf("IMEM readback OK — all %u words match\n", ATLAS_PROGRAM_LEN);

    printf("Starting Atlas core ...\n");
    atlas_start();
    asm volatile ("fence" ::: "memory");

    printf("Waiting for Atlas core to execute ...\n");
    uint32_t dbg0 = 0;
    for (i = 0; i < 200000000U; i++) {
        dbg0 = atlas_dbg0();
        if (dbg0 != 0) break;
    }

    uint32_t cycles  = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_CYCLE);
    uint32_t instcnt = atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_INSTCNT);
    uint32_t status  = atlas_status();

    printf("  DBG0    = %u\n", dbg0);
    printf("  cycles  = %u\n", cycles);
    printf("  instcnt = %u\n", instcnt);
    printf("  status  = 0x%08x\n", status);

    if (dbg0 == 0) {
        printf("FAIL: Atlas core did not complete (timeout)\n");
        return 1;
    } else if (dbg0 == 1) {
        printf("PASS: all ALU tests passed\n");
        return 0;
    } else {
        printf("FAIL: ALU test %u failed\n", dbg0);
        return 1;
    }
}

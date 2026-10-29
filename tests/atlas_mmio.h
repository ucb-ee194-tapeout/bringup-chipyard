#ifndef CHIPYARD_TESTS_ATLAS_MMIO_H
#define CHIPYARD_TESTS_ATLAS_MMIO_H

// Host-side MMIO layout for the Atlas accelerator.
// Authoritative map:  generators/atlas-npu/src/main/scala/atlas/scalar/ScalarCore.scala (AtlasMemMap)
// Authoritative CSRs: generators/atlas-npu/src/main/scala/diplomatic/memory/CSRFile.scala

#include <stdint.h>

#define ATLAS_IMEM_BASE        0x00020000UL
#define ATLAS_IMEM_SIZE        0x00020000UL   /* 128 KiB */
#define ATLAS_IMEM_HALF_BYTES  (ATLAS_IMEM_SIZE / 2)

#define ATLAS_CSR_BASE         0x00040000UL   /* 4 KiB window; only first 0x20 B used */
#define ATLAS_CSR_WINDOW_SIZE  0x00001000UL

#define ATLAS_VMEM_BASE        0x20000000UL
#define ATLAS_VMEM_SIZE        0x00180000UL   /* 1.5 MiB */
#define ATLAS_VMEM_NUM_BANKS   6UL
#define ATLAS_VMEM_BANK_BYTES  (ATLAS_VMEM_SIZE / ATLAS_VMEM_NUM_BANKS)  /* 256 KiB */

/*
 * FPGA DRAM is limited to the lower 4 GiB; the Saturn<->Atlas host
 * tests and Atlas baremetal suite all access DRAM through this shared
 * window so that simulator and silicon agree on the addressing.
 */
#define ATLAS_DRAM_BASE        0x90000000UL

/* CSR offsets from ATLAS_CSR_BASE */
#define ATLAS_CSR_CYCLE        0x00U  /* RW */
#define ATLAS_CSR_INSTCNT      0x04U  /* RW */
#define ATLAS_CSR_STATUS       0x08U  /* RO: bit 0 = halted, bits [2:1] = halt_reason */
#define ATLAS_CSR_ILLEGAL_PC   0x0CU  /* RO */
#define ATLAS_CSR_DBG0         0x10U  /* RW; Atlas kernels access as CSR 0xC10 */
#define ATLAS_CSR_DBG1         0x14U  /* RW; Atlas kernels access as CSR 0xC11 */
#define ATLAS_CSR_EXEC_CONTROL 0x18U  /* RW: bit 0 = 1 → start; auto-clears on halt */

#define ATLAS_STATUS_HALTED_BIT        0x1U
#define ATLAS_STATUS_HALT_REASON_SHIFT 1
#define ATLAS_STATUS_HALT_REASON_MASK  0x3U

/* halt_reason values (bits [2:1] of STATUS) */
#define ATLAS_HALT_NONE    0U
#define ATLAS_HALT_ILLEGAL 1U
#define ATLAS_HALT_ECALL   2U
#define ATLAS_HALT_EBREAK  3U

static inline uint32_t atlas_read32(uintptr_t addr) {
  return *(volatile uint32_t *)addr;
}
static inline void atlas_write32(uintptr_t addr, uint32_t val) {
  *(volatile uint32_t *)addr = val;
}

static inline uint32_t atlas_status(void) {
  return atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_STATUS);
}
static inline int atlas_halted(void) {
  return (int)(atlas_status() & ATLAS_STATUS_HALTED_BIT);
}
static inline uint32_t atlas_halt_reason(void) {
  return (atlas_status() >> ATLAS_STATUS_HALT_REASON_SHIFT) & ATLAS_STATUS_HALT_REASON_MASK;
}
static inline void atlas_start(void) {
  atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_EXEC_CONTROL, 1);
}
static inline void atlas_stop(void) {
  atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_EXEC_CONTROL, 0);
}
static inline uint32_t atlas_dbg0(void) {
  return atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_DBG0);
}
static inline uint32_t atlas_dbg1(void) {
  return atlas_read32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1);
}
static inline void atlas_set_dbg1(uint32_t v) {
  atlas_write32(ATLAS_CSR_BASE + ATLAS_CSR_DBG1, v);
}

static inline void atlas_imem_write(uint32_t word_idx, uint32_t word) {
  atlas_write32(ATLAS_IMEM_BASE + ((uintptr_t)word_idx * 4), word);
}
static inline uint32_t atlas_imem_read(uint32_t word_idx) {
  return atlas_read32(ATLAS_IMEM_BASE + ((uintptr_t)word_idx * 4));
}

#endif /* CHIPYARD_TESTS_ATLAS_MMIO_H */

#include <stdio.h>
#include <riscv-pk/encoding.h>
#include "marchid.h"
#include <stdint.h>

int main(void) {
  uint64_t marchid = read_csr(marchid);
  volatile uint32_t *bottom_uncached = 0x40000000;
  volatile uint32_t *dram = 0x80000000;
  volatile uint32_t *top_uncached = 0x880000000;

  *bottom_uncached = 1111;
  *top_uncached = 2222;

  *bottom_uncached = 3333;
  *top_uncached = 4444;

  *bottom_uncached = 5555;
  *top_uncached = 6666;

  const char* march = get_march(marchid);
  // printf("Hello world from core 0, a %s\n", march);
  return 0;
}

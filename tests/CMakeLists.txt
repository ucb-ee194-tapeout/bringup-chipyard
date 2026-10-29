########################################################################################################################
# file:  CMakeLists.txt
#
# usage: 
#   Edit "VARIABLES"-section to suit project requirements.
#   Build instructions:
#     cmake -S ./ -B ./build/ -D CMAKE_BUILD_TYPE=Debug
#     cmake --build ./build/ --target all
#   Cleaning:
#     cmake --build ./build/ --target clean
########################################################################################################################
cmake_minimum_required(VERSION 3.12)

project(chipyard-tests LANGUAGES C CXX)


#################################
# RISCV Toolchain
#################################

set(CMAKE_SYSTEM_NAME         "Generic" CACHE STRING "")
set(CMAKE_SYSTEM_PROCESSOR    "riscv"   CACHE STRING "")

set(TOOLCHAIN_PREFIX    "riscv64-unknown-elf-")

set(CMAKE_AR            "${TOOLCHAIN_PREFIX}ar")
set(CMAKE_ASM_COMPILER  "${TOOLCHAIN_PREFIX}gcc")
set(CMAKE_C_COMPILER    "${TOOLCHAIN_PREFIX}gcc")
set(CMAKE_CXX_COMPILER  "${TOOLCHAIN_PREFIX}g++")
set(CMAKE_LINKER        "${TOOLCHAIN_PREFIX}ld")
set(CMAKE_OBJCOPY       "${TOOLCHAIN_PREFIX}objcopy")
set(CMAKE_OBJDUMP       "${TOOLCHAIN_PREFIX}objdump")
set(CMAKE_SIZE          "${TOOLCHAIN_PREFIX}size")

set(CMAKE_EXECUTABLE_SUFFIX            ".riscv")


#################################
# Flags
#################################

# CPU architecture
set(ARCH                "rv64imafd")
set(ABI                 "lp64d")
set(CMODEL              "medany")
set(ARCH_FLAGS          -march=${ARCH} -mabi=${ABI} -mcmodel=${CMODEL})

# Saturn+RVV architecture (opt-in via add_saturn_test).
# Matches generators/saturn/benchmarks/Makefile.
set(SATURN_ARCH         "rv64gcv_zfh_zvfh")
set(SATURN_ABI          "lp64d")
set(SATURN_ARCH_FLAGS   -march=${SATURN_ARCH} -mabi=${SATURN_ABI} -mcmodel=${CMODEL})

# spec
set(SPECS               "htif_nano.specs")
set(SPEC_FLAGS          -specs=${SPECS})

# linker script
set(LINKER_SCRIPT       "htif.ld")

add_compile_options(-std=gnu99)
add_compile_options(-O2 -Wall -Wextra)
add_compile_options(-fno-common -fno-builtin-printf)
add_compile_options(${ARCH_FLAGS})
add_compile_options(${SPEC_FLAGS})

add_link_options(-static)
add_link_options(${SPEC_FLAGS})
add_link_options(-T ${LINKER_SCRIPT})


#################################
# Saturn+RVV target helper
#################################

# Usage: add_saturn_test(<target> <source>...)
# Trailing -march/-mabi/-mcmodel override the directory-wide scalar flags
# (gcc honors the last -march on the command line).
function(add_saturn_test target_name)
    add_executable(${target_name} ${ARGN})
    target_compile_options(${target_name} PRIVATE ${SATURN_ARCH_FLAGS})
    target_link_options(${target_name} PRIVATE ${SATURN_ARCH_FLAGS})
endfunction()


#################################
# Build
#################################

add_executable(pwm pwm.c)
add_executable(blkdev blkdev.c)
add_executable(accum accum.c)
add_executable(charcount charcount.c)
add_executable(cpp-hello cpp-hello.cpp)
add_executable(nic-loopback nic-loopback.c)
add_executable(big-blkdev big-blkdev.c)
add_executable(pingd pingd.c)
add_executable(streaming-passthrough streaming-passthrough.c)
add_executable(streaming-fir streaming-fir.c)
add_executable(nvdla nvdla.c)
add_executable(spiflashread spiflashread.c)
add_executable(spiflashwrite spiflashwrite.c)
add_executable(fft fft.c)
add_executable(gcd gcd.c)
add_executable(hello hello.c)
add_executable(mt-hello mt-hello.c)
add_executable(symmetric symmetric.c)
add_executable(ctc-test ctc-test.c)
add_executable(multi-ctc-test multi-ctc-test.c)
add_executable(triple-chiplet triple-chiplet.c)
add_executable(atlas-alu-basic atlas_alu_basic.c)

# Saturn+Atlas cooperative tests (RVV-enabled)
add_saturn_test(saturn_atlas_vmem_smoke saturn_atlas_vmem_smoke.c)
add_saturn_test(saturn_atlas_pause_resume saturn_atlas_pause_resume.c)
add_saturn_test(saturn_atlas_pi0_slice saturn_atlas_pi0_slice.c)

# Buffered dual-kernel IMEM loads (scalar host, no RVV)
add_executable(saturn_atlas_buffered_optA_end_jal saturn_atlas_buffered_optA_end_jal.c)
add_executable(saturn_atlas_buffered_lower_first  saturn_atlas_buffered_lower_first.c)
add_executable(saturn_atlas_buffered_optB_restart saturn_atlas_buffered_optB_restart.c)

# Smolvla compose golden headers — regenerated at build time from the
# atlas-npu submodule so we don't check in large derivative artifacts.
set(SAC_GEN_DIR ${CMAKE_SOURCE_DIR}/../generators/atlas-npu/baremetal/generators)
set(SAC_GOLDEN_DIR ${CMAKE_BINARY_DIR}/golden_headers)
file(MAKE_DIRECTORY ${SAC_GOLDEN_DIR})

# Fail early at configure time if the submodule isn't populated.
foreach(gen
    gen_saturn_atlas_smolvla_compose.py
    gen_saturn_atlas_smolvla_compose_large.py
    gen_saturn_atlas_tri_phase.py
    gen_saturn_atlas_tri_phase_large.py
    gen_saturn_atlas_fp8_quantize_bridge.py
    gen_saturn_atlas_fp8_dequant_bridge.py
    gen_saturn_atlas_quad_phase_large.py)
    if(NOT EXISTS ${SAC_GEN_DIR}/${gen})
        message(FATAL_ERROR
            "atlas-npu submodule not initialized: ${SAC_GEN_DIR} is missing "
            "${gen}. Run: git submodule update --init generators/atlas-npu")
    endif()
endforeach()

# Dependency set for all generators. vpu_gen_utils lives alongside the
# generators; software_models/ holds VectorEngineModel + BF16 semantics
# helpers that govern the emitted bits. VectorEngineModel eagerly instantiates
# lane-box modules that dynamically load files under atlas-npu/dependencies/
# (notably fpex/model/bf16_exp_model.py via software_models/vpu/lane_boxes/exp.py),
# so edits there must also retrigger header regeneration. CONFIGURE_DEPENDS
# re-globs on each build so new helpers added later still trigger regen.
set(SAC_GEN_LOCAL_DEPS
    ${SAC_GEN_DIR}/gen_utils.py
    ${SAC_GEN_DIR}/vpu_gen_utils.py
    ${SAC_GEN_DIR}/mxu_fp8_utils.py)
file(GLOB_RECURSE SAC_GEN_SHARED_DEPS
    CONFIGURE_DEPENDS
    ${SAC_GEN_DIR}/software_models/*.py
    ${SAC_GEN_DIR}/../../dependencies/fpex/model/*.py)

add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_smolvla_compose_golden.h
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${SAC_GEN_DIR}
            python3 ${SAC_GEN_DIR}/gen_saturn_atlas_smolvla_compose.py
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_smolvla_compose_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_smolvla_compose.py
            ${SAC_GEN_LOCAL_DEPS}
            ${SAC_GEN_SHARED_DEPS}
    COMMENT "Generating saturn_atlas_smolvla_compose_golden.h"
)

# Native-BF16 twin: same generator, --interlude-mode=native_bf16_rne (BF16
# RNE narrow inside the FP32 mul, matching Saturn vfmul.vv altfmt=1 +
# vfncvtbf16.f.f.w semantics — different bytes from the truncate path).
add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_smolvla_compose_native_bf16_golden.h
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${SAC_GEN_DIR}
            python3 ${SAC_GEN_DIR}/gen_saturn_atlas_smolvla_compose.py
            --interlude-mode native_bf16_rne
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_smolvla_compose_native_bf16_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_smolvla_compose.py
            ${SAC_GEN_LOCAL_DEPS}
            ${SAC_GEN_SHARED_DEPS}
    COMMENT "Generating saturn_atlas_smolvla_compose_native_bf16_golden.h"
)

add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_smolvla_compose_large_golden.h
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${SAC_GEN_DIR}
            python3 ${SAC_GEN_DIR}/gen_saturn_atlas_smolvla_compose_large.py
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_smolvla_compose_large_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_smolvla_compose_large.py
            ${SAC_GEN_LOCAL_DEPS}
            ${SAC_GEN_SHARED_DEPS}
    COMMENT "Generating saturn_atlas_smolvla_compose_large_golden.h"
)

add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_tri_phase_golden.h
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${SAC_GEN_DIR}
            python3 ${SAC_GEN_DIR}/gen_saturn_atlas_tri_phase.py
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_tri_phase_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_tri_phase.py
            ${SAC_GEN_LOCAL_DEPS}
            ${SAC_GEN_SHARED_DEPS}
    COMMENT "Generating saturn_atlas_tri_phase_golden.h"
)

# Native-BF16 twin: same generator, --interlude-mode=native_bf16_rne. Both
# interludes (Y->Y' and Z->Z'') route through bf16_mul (RNE narrow); phases
# 1, 2, 3 stay Atlas-modeled via VectorEngineModel.
add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_tri_phase_native_bf16_golden.h
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${SAC_GEN_DIR}
            python3 ${SAC_GEN_DIR}/gen_saturn_atlas_tri_phase.py
            --interlude-mode native_bf16_rne
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_tri_phase_native_bf16_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_tri_phase.py
            ${SAC_GEN_LOCAL_DEPS}
            ${SAC_GEN_SHARED_DEPS}
    COMMENT "Generating saturn_atlas_tri_phase_native_bf16_golden.h"
)

add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_tri_phase_large_golden.h
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${SAC_GEN_DIR}
            python3 ${SAC_GEN_DIR}/gen_saturn_atlas_tri_phase_large.py
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_tri_phase_large_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_tri_phase_large.py
            ${SAC_GEN_LOCAL_DEPS}
            ${SAC_GEN_SHARED_DEPS}
    COMMENT "Generating saturn_atlas_tri_phase_large_golden.h"
)

# Saturn -> Atlas FP8 quantize bridge: Saturn quantizes A_bf16 -> A_fp8
# via native vfncvtbf16, then Atlas mxu1 consumes the bytes. Two
# checkpoints (Saturn quantizer alignment + end-to-end correctness).
add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_fp8_quantize_bridge_golden.h
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${SAC_GEN_DIR}
            python3 ${SAC_GEN_DIR}/gen_saturn_atlas_fp8_quantize_bridge.py
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_fp8_quantize_bridge_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_fp8_quantize_bridge.py
            ${SAC_GEN_LOCAL_DEPS}
            ${SAC_GEN_SHARED_DEPS}
    COMMENT "Generating saturn_atlas_fp8_quantize_bridge_golden.h"
)

# Atlas -> Saturn FP8 dequantize bridge: Atlas runs mxu1_quant_relu_roundtrip
# (matmul + ReLU + FP8 pack), Saturn decodes the FP8 bytes via native
# vfwcvtbf16. Two checkpoints (Atlas FP8 production + Saturn FP8 decode).
add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_fp8_dequant_bridge_golden.h
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${SAC_GEN_DIR}
            python3 ${SAC_GEN_DIR}/gen_saturn_atlas_fp8_dequant_bridge.py
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_fp8_dequant_bridge_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_fp8_dequant_bridge.py
            ${SAC_GEN_LOCAL_DEPS}
            ${SAC_GEN_SHARED_DEPS}
    COMMENT "Generating saturn_atlas_fp8_dequant_bridge_golden.h"
)

# Quad-phase pipeline (3 Atlas phases + 2 Saturn interludes per tile).
# A1 and A3 are JAL chains across IMEM halves; A2 is a single MXU0 matmul.
# Both MXUs covered: MXU1 (IPT) in A1/A3, MXU0 (SA) in A2.
add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_quad_phase_large_golden.h
    COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${SAC_GEN_DIR}
            python3 ${SAC_GEN_DIR}/gen_saturn_atlas_quad_phase_large.py
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_quad_phase_large_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_quad_phase_large.py
            ${SAC_GEN_LOCAL_DEPS}
            ${SAC_GEN_SHARED_DEPS}
    COMMENT "Generating saturn_atlas_quad_phase_large_golden.h"
)

# Smolvla composition: Atlas mul -> Saturn RVV BF16 scale -> Atlas row-reduce
add_saturn_test(saturn_atlas_smolvla_compose
    saturn_atlas_smolvla_compose.c
    ${SAC_GOLDEN_DIR}/saturn_atlas_smolvla_compose_golden.h)
target_include_directories(saturn_atlas_smolvla_compose PRIVATE ${SAC_GOLDEN_DIR})

# Smolvla composition — native-BF16 interlude twin. Saturn interlude uses
# Saturn's MX BF16 path (vfmul.vv altfmt=1 + RNE narrow) instead of the
# 9-step zext/sll/reinterpret/srl/nsrl bit-hack. Different bytes from the
# truncate twin (RNE vs upper-16 truncation); kept as a sibling so the
# truncation path stays as a regression anchor for Saturn FP32 subnormal
# behavior.
add_saturn_test(saturn_atlas_smolvla_compose_native_bf16
    saturn_atlas_smolvla_compose_native_bf16.c
    ${SAC_GOLDEN_DIR}/saturn_atlas_smolvla_compose_native_bf16_golden.h)
target_include_directories(saturn_atlas_smolvla_compose_native_bf16 PRIVATE ${SAC_GOLDEN_DIR})

# Smolvla composition — multi-tile variant. Default tile count (16) lives
# in the driver source via #ifndef SAC_L_N_TILES. Override from the CMake
# command line with -DSAC_L_N_TILES_OVERRIDE=N for dev runs (e.g. N=2 to
# cover both edge-case tiles in ~1.5 min of VCS).
add_saturn_test(saturn_atlas_smolvla_compose_large
    saturn_atlas_smolvla_compose_large.c
    ${SAC_GOLDEN_DIR}/saturn_atlas_smolvla_compose_large_golden.h)
target_include_directories(saturn_atlas_smolvla_compose_large PRIVATE ${SAC_GOLDEN_DIR})
if(DEFINED SAC_L_N_TILES_OVERRIDE)
    target_compile_definitions(saturn_atlas_smolvla_compose_large PRIVATE
        SAC_L_N_TILES=${SAC_L_N_TILES_OVERRIDE})
endif()

# Tri-phase cooperative test: Atlas mul (ECALL) -> Saturn RVV scale ->
# Atlas row-reduce (JALR to upper IMEM) -> Atlas add (ECALL) -> Saturn RVV
# post-scale. First cooperative test that combines ECALL handshake,
# buffered-IMEM JAL dispatch, and Saturn compute in one flow.
add_saturn_test(saturn_atlas_tri_phase
    saturn_atlas_tri_phase.c
    ${SAC_GOLDEN_DIR}/saturn_atlas_tri_phase_golden.h)
target_include_directories(saturn_atlas_tri_phase PRIVATE ${SAC_GOLDEN_DIR})

# Native-BF16 twin: same Atlas control flow (ECALL + JALR + ECALL), same
# kernels, but both Saturn interludes use the MX BF16 path (vfmul.vv
# altfmt=1) instead of the bit-hack. Different bytes from the truncate
# twin (RNE vs upper-16 truncation); kept as a sibling so the truncation
# path stays as the Saturn FP32 subnormal regression anchor.
add_saturn_test(saturn_atlas_tri_phase_native_bf16
    saturn_atlas_tri_phase_native_bf16.c
    ${SAC_GOLDEN_DIR}/saturn_atlas_tri_phase_native_bf16_golden.h)
target_include_directories(saturn_atlas_tri_phase_native_bf16 PRIVATE ${SAC_GOLDEN_DIR})

# FP8 quantize bridge: Saturn vfncvtbf16 (BF16->E4M3) -> Atlas mxu1
# matmul+bias. Two-checkpoint: Saturn quantizer encoding alignment
# (ckpt1) + end-to-end via IPT + VPU bias-add (ckpt2).
add_saturn_test(saturn_atlas_fp8_quantize_bridge
    saturn_atlas_fp8_quantize_bridge.c
    ${SAC_GOLDEN_DIR}/saturn_atlas_fp8_quantize_bridge_golden.h)
target_include_directories(saturn_atlas_fp8_quantize_bridge PRIVATE ${SAC_GOLDEN_DIR})

# FP8 dequantize bridge: Atlas mxu1+pack -> Saturn vfwcvtbf16. Closes
# the bidirectional FP8 byte-equivalence story. Two-checkpoint: Atlas
# FP8 production (ckpt1) + Saturn E4M3 decode (ckpt2).
add_saturn_test(saturn_atlas_fp8_dequant_bridge
    saturn_atlas_fp8_dequant_bridge.c
    ${SAC_GOLDEN_DIR}/saturn_atlas_fp8_dequant_bridge_golden.h)
target_include_directories(saturn_atlas_fp8_dequant_bridge PRIVATE ${SAC_GOLDEN_DIR})

# Tri-phase multi-tile variant. Default tile count (16) lives in the
# driver source via #ifndef SAC_TRI_L_N_TILES. Override at CMake level
# with -DSAC_TRI_L_N_TILES_OVERRIDE=N for dev runs (e.g. N=2 to cover
# both edge-case tiles only).
add_saturn_test(saturn_atlas_tri_phase_large
    saturn_atlas_tri_phase_large.c
    ${SAC_GOLDEN_DIR}/saturn_atlas_tri_phase_large_golden.h)
target_include_directories(saturn_atlas_tri_phase_large PRIVATE ${SAC_GOLDEN_DIR})
if(DEFINED SAC_TRI_L_N_TILES_OVERRIDE)
    target_compile_definitions(saturn_atlas_tri_phase_large PRIVATE
        SAC_TRI_L_N_TILES=${SAC_TRI_L_N_TILES_OVERRIDE})
endif()

# Quad-phase pipeline. 3 Atlas phases (2 JAL chains + 1 single MXU0 matmul)
# interleaved with 2 Saturn RVV BF16 interludes, all wrapped in an
# N_TILES=8 host loop. Default tile count lives in the driver source via
# #ifndef SAC_QUAD_N_TILES; override at CMake level with
# -DSAC_QUAD_N_TILES_OVERRIDE=N for dev runs.
add_saturn_test(saturn_atlas_quad_phase_large
    saturn_atlas_quad_phase_large.c
    ${SAC_GOLDEN_DIR}/saturn_atlas_quad_phase_large_golden.h)
target_include_directories(saturn_atlas_quad_phase_large PRIVATE ${SAC_GOLDEN_DIR})
if(DEFINED SAC_QUAD_N_TILES_OVERRIDE)
    target_compile_definitions(saturn_atlas_quad_phase_large PRIVATE
        SAC_QUAD_N_TILES=${SAC_QUAD_N_TILES_OVERRIDE})
endif()

#################################
# Saturn↔Atlas memory stress suite (T1-T5). Each test pairs a
# hand-written Atlas kernel under
# baremetal/assembly/saturn_atlas_cooperative/ with a host driver here
# in tests/. Two CMake-time generators feed each driver: gen_kernel_header.py
# assembles the .S into an embedded uint32_t array, and a per-test Python
# golden generator emits the DRAM input/expected header.
#################################

set(SAC_ASM_DIR
    ${CMAKE_SOURCE_DIR}/../generators/atlas-npu/baremetal/assembly/saturn_atlas_cooperative)
set(SAC_KERNEL_HEADER_DIR ${CMAKE_BINARY_DIR}/saturn_atlas_kernels)
file(MAKE_DIRECTORY ${SAC_KERNEL_HEADER_DIR})

set(SAC_KERNEL_GEN ${SAC_GEN_DIR}/gen_kernel_header.py)
set(SAC_ASSEMBLER ${SAC_GEN_DIR}/../assembler.py)

if(NOT EXISTS ${SAC_KERNEL_GEN})
    message(FATAL_ERROR
        "atlas-npu submodule missing ${SAC_KERNEL_GEN}. "
        "Run: git submodule update --init generators/atlas-npu")
endif()
if(NOT EXISTS ${SAC_ASSEMBLER})
    message(FATAL_ERROR
        "atlas-npu submodule missing ${SAC_ASSEMBLER}.")
endif()

# T2: 8-channel DMA echo + Saturn RVV bank-7 sweep
add_custom_command(
    OUTPUT ${SAC_KERNEL_HEADER_DIR}/kernel_sas_8dma_rvv_vmem.h
    COMMAND python3 ${SAC_KERNEL_GEN}
            ${SAC_ASM_DIR}/sas_8dma_rvv_vmem.S
            sas_8dma_rvv_vmem
            ${SAC_KERNEL_HEADER_DIR}/kernel_sas_8dma_rvv_vmem.h
    DEPENDS ${SAC_ASM_DIR}/sas_8dma_rvv_vmem.S
            ${SAC_KERNEL_GEN}
            ${SAC_ASSEMBLER}
    COMMENT "Assembling kernel_sas_8dma_rvv_vmem.h"
)

add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_8dma_rvv_vmem_golden.h
    COMMAND python3 ${SAC_GEN_DIR}/gen_saturn_atlas_8dma_rvv_vmem.py
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_8dma_rvv_vmem_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_8dma_rvv_vmem.py
    COMMENT "Generating saturn_atlas_8dma_rvv_vmem_golden.h"
)

add_saturn_test(saturn_atlas_8dma_rvv_vmem
    saturn_atlas_8dma_rvv_vmem.c
    ${SAC_KERNEL_HEADER_DIR}/kernel_sas_8dma_rvv_vmem.h
    ${SAC_GOLDEN_DIR}/saturn_atlas_8dma_rvv_vmem_golden.h)
target_include_directories(saturn_atlas_8dma_rvv_vmem PRIVATE
    ${SAC_KERNEL_HEADER_DIR}
    ${SAC_GOLDEN_DIR})

# Bisect variant: skip the Saturn RVV sweep entirely. Same Atlas
# kernel + same DMA goldens, but the host-side Saturn RVV bank-7
# loop is compiled out. Use this to determine whether the live-lock
# is in the Atlas DMA pump itself or in the Saturn-vs-VMEM-TL path.
add_saturn_test(saturn_atlas_8dma_rvv_vmem_nosaturn
    saturn_atlas_8dma_rvv_vmem.c
    ${SAC_KERNEL_HEADER_DIR}/kernel_sas_8dma_rvv_vmem.h
    ${SAC_GOLDEN_DIR}/saturn_atlas_8dma_rvv_vmem_golden.h)
target_include_directories(saturn_atlas_8dma_rvv_vmem_nosaturn PRIVATE
    ${SAC_KERNEL_HEADER_DIR}
    ${SAC_GOLDEN_DIR})
target_compile_definitions(saturn_atlas_8dma_rvv_vmem_nosaturn PRIVATE
    SAS_T2_SKIP_SATURN=1)

# T1: single-bank 3-way arb (LSU > DMA > Saturn TL on bank 3)
add_custom_command(
    OUTPUT ${SAC_KERNEL_HEADER_DIR}/kernel_sas_vmem_3way_arb.h
    COMMAND python3 ${SAC_KERNEL_GEN}
            ${SAC_ASM_DIR}/sas_vmem_3way_arb.S
            sas_vmem_3way_arb
            ${SAC_KERNEL_HEADER_DIR}/kernel_sas_vmem_3way_arb.h
    DEPENDS ${SAC_ASM_DIR}/sas_vmem_3way_arb.S
            ${SAC_KERNEL_GEN}
            ${SAC_ASSEMBLER}
    COMMENT "Assembling kernel_sas_vmem_3way_arb.h"
)

add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_vmem_3way_arb_golden.h
    COMMAND python3 ${SAC_GEN_DIR}/gen_saturn_atlas_vmem_3way_arb.py
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_vmem_3way_arb_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_vmem_3way_arb.py
    COMMENT "Generating saturn_atlas_vmem_3way_arb_golden.h"
)

add_saturn_test(saturn_atlas_vmem_3way_arb
    saturn_atlas_vmem_3way_arb.c
    ${SAC_KERNEL_HEADER_DIR}/kernel_sas_vmem_3way_arb.h
    ${SAC_GOLDEN_DIR}/saturn_atlas_vmem_3way_arb_golden.h)
target_include_directories(saturn_atlas_vmem_3way_arb PRIVATE
    ${SAC_KERNEL_HEADER_DIR}
    ${SAC_GOLDEN_DIR})

# 4-rung 2-bank arb: vector LSU + DMA + Saturn TL on bank 3, plus
# scalar LSU + DMA + Saturn TL on bank 4.
add_custom_command(
    OUTPUT ${SAC_KERNEL_HEADER_DIR}/kernel_sas_4rung_2bank_arb.h
    COMMAND python3 ${SAC_KERNEL_GEN}
            ${SAC_ASM_DIR}/sas_4rung_2bank_arb.S
            sas_4rung_2bank_arb
            ${SAC_KERNEL_HEADER_DIR}/kernel_sas_4rung_2bank_arb.h
    DEPENDS ${SAC_ASM_DIR}/sas_4rung_2bank_arb.S
            ${SAC_KERNEL_GEN}
            ${SAC_ASSEMBLER}
    COMMENT "Assembling kernel_sas_4rung_2bank_arb.h"
)

add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_4rung_2bank_arb_golden.h
    COMMAND python3 ${SAC_GEN_DIR}/gen_saturn_atlas_4rung_2bank_arb.py
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_4rung_2bank_arb_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_4rung_2bank_arb.py
    COMMENT "Generating saturn_atlas_4rung_2bank_arb_golden.h"
)

add_saturn_test(saturn_atlas_4rung_2bank_arb
    saturn_atlas_4rung_2bank_arb.c
    ${SAC_KERNEL_HEADER_DIR}/kernel_sas_4rung_2bank_arb.h
    ${SAC_GOLDEN_DIR}/saturn_atlas_4rung_2bank_arb_golden.h)
target_include_directories(saturn_atlas_4rung_2bank_arb PRIVATE
    ${SAC_KERNEL_HEADER_DIR}
    ${SAC_GOLDEN_DIR})

# T5: saturn_atlas_vmem_dram_dual — Saturn VMEM (bank 3 +0x1000)
# + DRAM (64 KiB ring, 4 KiB chunks rotating per iter) concurrent
# traffic while Atlas runs T1.5-style 4-rung 2-bank LSU + DMA.
add_custom_command(
    OUTPUT ${SAC_KERNEL_HEADER_DIR}/kernel_sas_vmem_dram_dual.h
    COMMAND python3 ${SAC_KERNEL_GEN}
            ${SAC_ASM_DIR}/sas_vmem_dram_dual.S
            sas_vmem_dram_dual
            ${SAC_KERNEL_HEADER_DIR}/kernel_sas_vmem_dram_dual.h
    DEPENDS ${SAC_ASM_DIR}/sas_vmem_dram_dual.S
            ${SAC_KERNEL_GEN}
            ${SAC_ASSEMBLER}
    COMMENT "Assembling kernel_sas_vmem_dram_dual.h"
)

add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_vmem_dram_dual_golden.h
    COMMAND python3 ${SAC_GEN_DIR}/gen_saturn_atlas_vmem_dram_dual.py
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_vmem_dram_dual_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_vmem_dram_dual.py
    COMMENT "Generating saturn_atlas_vmem_dram_dual_golden.h"
)

add_saturn_test(saturn_atlas_vmem_dram_dual
    saturn_atlas_vmem_dram_dual.c
    ${SAC_KERNEL_HEADER_DIR}/kernel_sas_vmem_dram_dual.h
    ${SAC_GOLDEN_DIR}/saturn_atlas_vmem_dram_dual_golden.h)
target_include_directories(saturn_atlas_vmem_dram_dual PRIVATE
    ${SAC_KERNEL_HEADER_DIR}
    ${SAC_GOLDEN_DIR})

# T4: Saturn DMEM RVV sweep + Atlas DMA pump (SBUS xbar contention)
add_custom_command(
    OUTPUT ${SAC_KERNEL_HEADER_DIR}/kernel_sas_dmem_sbus_dma.h
    COMMAND python3 ${SAC_KERNEL_GEN}
            ${SAC_ASM_DIR}/sas_dmem_sbus_dma.S
            sas_dmem_sbus_dma
            ${SAC_KERNEL_HEADER_DIR}/kernel_sas_dmem_sbus_dma.h
    DEPENDS ${SAC_ASM_DIR}/sas_dmem_sbus_dma.S
            ${SAC_KERNEL_GEN}
            ${SAC_ASSEMBLER}
    COMMENT "Assembling kernel_sas_dmem_sbus_dma.h"
)

add_custom_command(
    OUTPUT ${SAC_GOLDEN_DIR}/saturn_atlas_dmem_sbus_dma_golden.h
    COMMAND python3 ${SAC_GEN_DIR}/gen_saturn_atlas_dmem_sbus_dma.py
            --out-header ${SAC_GOLDEN_DIR}/saturn_atlas_dmem_sbus_dma_golden.h
    DEPENDS ${SAC_GEN_DIR}/gen_saturn_atlas_dmem_sbus_dma.py
    COMMENT "Generating saturn_atlas_dmem_sbus_dma_golden.h"
)

add_saturn_test(saturn_atlas_dmem_sbus_dma
    saturn_atlas_dmem_sbus_dma.c
    ${SAC_KERNEL_HEADER_DIR}/kernel_sas_dmem_sbus_dma.h
    ${SAC_GOLDEN_DIR}/saturn_atlas_dmem_sbus_dma_golden.h)
target_include_directories(saturn_atlas_dmem_sbus_dma PRIVATE
    ${SAC_KERNEL_HEADER_DIR}
    ${SAC_GOLDEN_DIR})

# saturn_atlas_imem_carousel_live (formerly T3) is currently unwired —
# broken by design, pending a protocol redesign with entry-marker
# handshakes (verify-after-halt instead of verify-during-staging).

#################################
# Disassembly
#################################

# Add a target to generate all disassemblies
add_custom_target(dump ALL)

# Function to add disassembly target for an executable
function(add_dump_target target_name)
    add_custom_target(${target_name}-dump
        BYPRODUCTS ${CMAKE_BINARY_DIR}/${target_name}.dump
        COMMAND ${CMAKE_OBJDUMP} -D $<TARGET_FILE:${target_name}> > ${CMAKE_BINARY_DIR}/${target_name}.dump
        DEPENDS ${target_name}
        COMMENT "Generating disassembly for ${target_name}"
    )
    add_dependencies(${target_name}-dump ${target_name})
    add_dependencies(dump ${target_name}-dump)
endfunction()

add_dump_target(pwm)
add_dump_target(blkdev)
add_dump_target(accum)
add_dump_target(charcount)
add_dump_target(cpp-hello)
add_dump_target(nic-loopback)
add_dump_target(big-blkdev)
add_dump_target(pingd)
add_dump_target(streaming-passthrough)
add_dump_target(streaming-fir)
add_dump_target(nvdla)
add_dump_target(spiflashread)
add_dump_target(spiflashwrite)
add_dump_target(fft)
add_dump_target(gcd)
add_dump_target(hello)
add_dump_target(mt-hello)
add_dump_target(symmetric)
add_dump_target(ctc-test)
add_dump_target(multi-ctc-test)
add_dump_target(triple-chiplet)
add_dump_target(atlas-alu-basic)
add_dump_target(saturn_atlas_vmem_smoke)
add_dump_target(saturn_atlas_pause_resume)
add_dump_target(saturn_atlas_pi0_slice)
add_dump_target(saturn_atlas_buffered_optA_end_jal)
add_dump_target(saturn_atlas_buffered_lower_first)
add_dump_target(saturn_atlas_buffered_optB_restart)
add_dump_target(saturn_atlas_smolvla_compose_large)
add_dump_target(saturn_atlas_smolvla_compose)
add_dump_target(saturn_atlas_smolvla_compose_native_bf16)
add_dump_target(saturn_atlas_tri_phase)
add_dump_target(saturn_atlas_tri_phase_large)
add_dump_target(saturn_atlas_tri_phase_native_bf16)
add_dump_target(saturn_atlas_fp8_quantize_bridge)
add_dump_target(saturn_atlas_fp8_dequant_bridge)
add_dump_target(saturn_atlas_quad_phase_large)
add_dump_target(saturn_atlas_8dma_rvv_vmem)
add_dump_target(saturn_atlas_8dma_rvv_vmem_nosaturn)
add_dump_target(saturn_atlas_vmem_3way_arb)
add_dump_target(saturn_atlas_4rung_2bank_arb)
add_dump_target(saturn_atlas_vmem_dram_dual)
add_dump_target(saturn_atlas_dmem_sbus_dma)


# Add custom command to generate spiflash.img from spiflash.py
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/spiflash.img
    COMMAND python3 ${CMAKE_SOURCE_DIR}/spiflash.py --outfile ${CMAKE_SOURCE_DIR}/spiflash.img
    DEPENDS ${CMAKE_SOURCE_DIR}/spiflash.py
    COMMENT "Generating spiflash.img"
)

# Add a target for spiflash.img
add_custom_target(spiflash_img ALL
    DEPENDS ${CMAKE_BINARY_DIR}/spiflash.img
)

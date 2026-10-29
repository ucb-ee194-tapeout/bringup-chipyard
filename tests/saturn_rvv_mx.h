/* saturn_rvv_mx.h
 *
 * Minimal vendored subset of Saturn's MX RVV inline-asm helpers, sufficient
 * for the Saturn↔Atlas cooperative tests that need native BF16/FP8 ops.
 *
 * Upstream: generators/saturn/benchmarks/common/rvv_mx.h (Saturn submodule
 * SHA de04946a; pulled into chipyard via commit f6a75c46 "add jtag + saturn
 * fp8/bf16 + tacit").
 *
 * Why we vendor: GCC 13.2.0 in our conda-env does not recognize zfbfmin /
 * zvfbfmin / zvfbfa / zvfofp8min in -march. Saturn's own MX benchmarks
 * compile with plain `-march=rv64gcv_zfh_zvfh` and emit MX instructions via
 * raw `.insn` encodings — exactly what the macros below do — so the toolchain
 * doesn't need to know the extension names. The VCS simv (EE290SimConfig +
 * VectorParams.mxParams) executes the encoded bytes directly.
 *
 * Scope: BF16↔FP8 / BF16↔FP32 convert helpers + altfmt vsetvli. Add Saturn
 * benchmark helpers (bdot, opu macros) only if a future test needs them.
 */

#ifndef SATURN_RVV_MX_H
#define SATURN_RVV_MX_H

/* ── VTYPE constants ───────────────────────────────────────────────────── */

#define LMUL_M1   "000"
#define LMUL_M2   "001"
#define LMUL_M4   "010"
#define LMUL_M8   "011"
#define LMUL_MF2  "111"
#define LMUL_MF4  "110"
#define LMUL_MF8  "101"

#define SEW_E8    "00"
#define SEW_E16   "01"
#define SEW_E32   "10"
#define SEW_E64   "11"

/* ── Register-name helper strings (V/X/F all map to xN; .insn takes raw
 *    register numbers regardless of the architectural class). ──────────── */

#define V0  "x0"
#define V1  "x1"
#define V2  "x2"
#define V3  "x3"
#define V4  "x4"
#define V5  "x5"
#define V6  "x6"
#define V7  "x7"
#define V8  "x8"
#define V9  "x9"
#define V10 "x10"
#define V11 "x11"
#define V12 "x12"
#define V13 "x13"
#define V14 "x14"
#define V15 "x15"
#define V16 "x16"
#define V17 "x17"
#define V18 "x18"
#define V19 "x19"
#define V20 "x20"
#define V21 "x21"
#define V22 "x22"
#define V23 "x23"
#define V24 "x24"
#define V25 "x25"
#define V26 "x26"
#define V27 "x27"
#define V28 "x28"
#define V29 "x29"
#define V30 "x30"
#define V31 "x31"

#define F0  "x0"
#define F1  "x1"
#define F2  "x2"
#define F3  "x3"
#define F4  "x4"
#define F5  "x5"
#define F6  "x6"
#define F7  "x7"
#define F8  "x8"
#define F9  "x9"
#define F10 "x10"
#define F11 "x11"
#define F12 "x12"
#define F13 "x13"
#define F14 "x14"
#define F15 "x15"

/* ── vsetvli with altfmt bit ───────────────────────────────────────────── */
/* At SEW=16, alt=0 selects FP16 and alt=1 selects BF16.
 * At SEW=8,  alt=0 selects E4M3 and alt=1 selects E5M2 (matching Saturn's
 * vec-mx-unary benchmark: e4m3_narrow uses alt=0, e5m2_narrow uses alt=1).
 * Encoding: VSETVLI rd, rs1, vtype where vtype's altfmt bit lives in
 * bit 8 of the immediate field. */

#define VSETVLI_ALTFMT(vl, avl, sew, lmul, alt) \
    asm volatile(".insn i 0x57, 0x7, %0, %1, 0b000" #alt "000" sew lmul \
                 : "=r"(vl) : "r"(avl))

#define VSETVLI_ALTFMT_X0(avl, sew, lmul, alt) \
    asm volatile(".insn i 0x57, 0x7, zero, %0, 0b000" #alt "000" sew lmul \
                 :: "r"(avl))

/* ── Vector BF16/FP8 conversion ─────────────────────────────────────────── */
/* These are funct6=0x25 (vfncvt/vfwcvt) variants. The rs1 field encodes
 * the specific convert operation:
 *   x29 (=29) → vfncvtbf16.f.f.w     (FP32→BF16 narrow at SEW=16/altfmt=1,
 *                                     also FP16→FP8 at SEW=8 with altfmt
 *                                     selecting E4M3/E5M2)
 *   x13 (=13) → vfwcvtbf16.f.f.v     (BF16→FP32 widen at SEW=16/altfmt=1,
 *                                     FP8→BF16 at SEW=8)
 *   x31 (=31) → vfncvtbf16.sat.f.f.w (saturating narrow — used for FP8
 *                                     where saturation matters)
 * The actual format (BF16 vs FP8 vs E4M3 vs E5M2) is selected via the
 * preceding VSETVLI_ALTFMT setting. */

#define VFNCVTBF16_F_F_W(rd, vs2) \
    asm volatile(".insn r 0x57, 0x1, 0x25, " rd ", x29, " vs2)

#define VFWCVTBF16_F_F_V(rd, vs2) \
    asm volatile(".insn r 0x57, 0x1, 0x25, " rd ", x13, " vs2)

#define VFNCVTBF16_SAT_F_F_W(rd, vs2) \
    asm volatile(".insn r 0x57, 0x1, 0x25, " rd ", x31, " vs2)

/* ── Scalar BF16 conversion ────────────────────────────────────────────── */
/* fcvt.bf16.s rd, rs1: FP32 → BF16 (RNE). rs2 encodes which narrow convert.
 * fcvt.s.bf16 rd, rs1: BF16 → FP32. */

#define FCVT_BF16_S(fd, fs1) \
    asm volatile(".insn r 0x53, 0x0, 0x22, " fd ", " fs1 ", x8")

#define FCVT_S_BF16(fd, fs1) \
    asm volatile(".insn r 0x53, 0x0, 0x20, " fd ", " fs1 ", x6")

/* ── Debug helper ──────────────────────────────────────────────────────── */

#define STALL(n) \
    for (int __i = 0; __i < (n); __i ++) { asm volatile("nop"); }

/* ── FP rounding mode ──────────────────────────────────────────────────
 * Pin frm = RNE (000) before any native BF16/FP8 narrow sequence whose
 * golden was generated assuming RNE rounding. Saturn's MX FMA/narrow
 * pipeline reads frm from fcsr; goldens for compose_native_bf16,
 * tri_phase_native_bf16, and fp8_quantize_bridge are all RNE-keyed
 * (bf16_mul = f32_to_bf16_bits_rne; bf16_tile_to_e4m3_bytes = RNE
 * narrow). Don't rely on ambient reset state.
 */
#define SATURN_SET_FRM_RNE() \
    asm volatile ("csrwi frm, 0" ::: "memory")

#endif /* SATURN_RVV_MX_H */

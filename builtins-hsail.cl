// REQUIRES: hsail-registered-target
// RUN: %clang_cc1 -triple hsail-unknown-unknown -S -emit-llvm -o - %s | FileCheck %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef __attribute__((ext_vector_type(4))) long long4;


// CHECK-LABEL: @test_smulhi(
// CHECK: tail call i32 @llvm.hsail.smulhi.i32(i32 %x, i32 %y)

// CHECK: declare i32 @llvm.hsail.smulhi.i32(i32, i32) #1
void test_smulhi(volatile global int* out, int x, int y)
{
  *out = __builtin_hsail_smulhi(x, y);
}

// CHECK-LABEL: @test_smulhil(
// CHECK: tail call i64 @llvm.hsail.smulhi.i64(i64 %x, i64 %y)

// CHECK: declare i64 @llvm.hsail.smulhi.i64(i64, i64) #1
void test_smulhil(volatile global long* out, long x, long y)
{
  *out = __builtin_hsail_smulhil(x, y);
}

// CHECK-LABEL: @test_umulhi(
// CHECK: tail call i32 @llvm.hsail.umulhi.i32(i32 %x, i32 %y)

// CHECK: declare i32 @llvm.hsail.umulhi.i32(i32, i32) #1
void test_umulhi(volatile global int* out, int x, int y)
{
  *out = __builtin_hsail_umulhi(x, y);
}

// CHECK-LABEL: @test_umulhil(
// CHECK: tail call i64 @llvm.hsail.umulhi.i64(i64 %x, i64 %y)

// CHECK: declare i64 @llvm.hsail.umulhi.i64(i64, i64) #1
void test_umulhil(volatile global long* out, long x, long y)
{
  *out = __builtin_hsail_umulhil(x, y);
}

// CHECK-LABEL: @test_smad24(
// CHECK: tail call i32 @llvm.hsail.smad24(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.smad24(i32, i32, i32) #1
void test_smad24(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_smad24(x, y, z);
}

// CHECK-LABEL: @test_umad24(
// CHECK: tail call i32 @llvm.hsail.umad24(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.umad24(i32, i32, i32) #1
void test_umad24(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_umad24(x, y, z);
}

// CHECK-LABEL: @test_smad24hi(
// CHECK: tail call i32 @llvm.hsail.smad24hi(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.smad24hi(i32, i32, i32) #1
void test_smad24hi(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_smad24hi(x, y, z);
}

// CHECK-LABEL: @test_umad24hi(
// CHECK: tail call i32 @llvm.hsail.umad24hi(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.umad24hi(i32, i32, i32) #1
void test_umad24hi(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_umad24hi(x, y, z);
}

// CHECK-LABEL: @test_smul24(
// CHECK: tail call i32 @llvm.hsail.smul24(i32 %x, i32 %y)

// CHECK: declare i32 @llvm.hsail.smul24(i32, i32) #1
void test_smul24(volatile global int* out, int x, int y)
{
  *out = __builtin_hsail_smul24(x, y);
}

// CHECK-LABEL: @test_umul24(
// CHECK: tail call i32 @llvm.hsail.umul24(i32 %x, i32 %y)

// CHECK: declare i32 @llvm.hsail.umul24(i32, i32) #1
void test_umul24(volatile global int* out, int x, int y)
{
  *out = __builtin_hsail_umul24(x, y);
}

// CHECK-LABEL: @test_smul24hi(
// CHECK: tail call i32 @llvm.hsail.smul24hi(i32 %x, i32 %y)

// CHECK: declare i32 @llvm.hsail.smul24hi(i32, i32) #1
void test_smul24hi(volatile global int* out, int x, int y)
{
  *out = __builtin_hsail_smul24hi(x, y);
}

// CHECK-LABEL: @test_umul24hi(
// CHECK: tail call i32 @llvm.hsail.umul24hi(i32 %x, i32 %y)

// CHECK: declare i32 @llvm.hsail.umul24hi(i32, i32) #1
void test_umul24hi(volatile global int* out, int x, int y)
{
  *out = __builtin_hsail_umul24hi(x, y);
}

// CHECK-LABEL: @test_sbitextract(
// CHECK: tail call i32 @llvm.hsail.sbitextract.i32(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.sbitextract.i32(i32, i32, i32) #1
void test_sbitextract(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_sbitextract(x, y, z);
}

// CHECK-LABEL: @test_sbitextractl(
// CHECK: tail call i64 @llvm.hsail.sbitextract.i64(i64 %x, i32 %y, i32 %z)

// CHECK: declare i64 @llvm.hsail.sbitextract.i64(i64, i32, i32) #1
void test_sbitextractl(volatile global int* out, long x, int y, int z)
{
  *out = __builtin_hsail_sbitextractl(x, y, z);
}

// CHECK-LABEL: @test_ubitextract(
// CHECK: tail call i32 @llvm.hsail.ubitextract.i32(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.ubitextract.i32(i32, i32, i32) #1
void test_ubitextract(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_ubitextract(x, y, z);
}

// CHECK-LABEL: @test_ubitextractl(
// CHECK: tail call i64 @llvm.hsail.ubitextract.i64(i64 %x, i32 %y, i32 %z)

// CHECK: declare i64 @llvm.hsail.ubitextract.i64(i64, i32, i32) #1
void test_ubitextractl(volatile global int* out, long x, int y, int z)
{
  *out = __builtin_hsail_ubitextractl(x, y, z);
}

// CHECK-LABEL: @test_sbitinsert(
// CHECK: tail call i32 @llvm.hsail.sbitinsert.i32(i32 %x, i32 %y, i32 %z, i32 %w)

// CHECK: declare i32 @llvm.hsail.sbitinsert.i32(i32, i32, i32, i32) #1
void test_sbitinsert(volatile global int* out, int x, int y, int z, int w)
{
  *out = __builtin_hsail_sbitinsert(x, y, z, w);
}

// CHECK-LABEL: @test_sbitinsertl(
// CHECK: tail call i64 @llvm.hsail.sbitinsert.i64(i64 %x, i64 %y, i32 %z, i32 %w)

// CHECK: declare i64 @llvm.hsail.sbitinsert.i64(i64, i64, i32, i32) #1
void test_sbitinsertl(volatile global long* out, long x, long y, int z, int w)
{
  *out = __builtin_hsail_sbitinsertl(x, y, z, w);
}

// CHECK-LABEL: @test_ubitinsert(
// CHECK: tail call i32 @llvm.hsail.ubitinsert.i32(i32 %x, i32 %y, i32 %z, i32 %w)

// CHECK: declare i32 @llvm.hsail.ubitinsert.i32(i32, i32, i32, i32) #1
void test_ubitinsert(volatile global int* out, int x, int y, int z, int w)
{
  *out = __builtin_hsail_ubitinsert(x, y, z, w);
}

// CHECK-LABEL: @test_ubitinsertl(
// CHECK: tail call i64 @llvm.hsail.ubitinsert.i64(i64 %x, i64 %y, i32 %z, i32 %w)

// CHECK: declare i64 @llvm.hsail.ubitinsert.i64(i64, i64, i32, i32) #1
void test_ubitinsertl(volatile global long* out, long x, long y, int z, int w)
{
  *out = __builtin_hsail_ubitinsertl(x, y, z, w);
}

// CHECK-LABEL: @test_bitmask(
// CHECK: tail call i32 @llvm.hsail.bitmask.i32(i32 %x, i32 %y)
// CHECK: tail call i32 @llvm.hsail.bitmask.i32(i32 1, i32 %y)
// CHECK: tail call i32 @llvm.hsail.bitmask.i32(i32 %x, i32 2)

// CHECK: declare i32 @llvm.hsail.bitmask.i32(i32, i32) #1
void test_bitmask(volatile global int* out, int x, int y)
{
  *out = __builtin_hsail_bitmask(x, y);
  *out = __builtin_hsail_bitmask(1, y);
  *out = __builtin_hsail_bitmask(x, 2);
}

// CHECK-LABEL: @test_bitmaskl(
// CHECK: tail call i64 @llvm.hsail.bitmask.i64(i32 %x, i32 %y)
// CHECK: tail call i64 @llvm.hsail.bitmask.i64(i32 1, i32 %y)
// CHECK: tail call i64 @llvm.hsail.bitmask.i64(i32 %x, i32 2)

// CHECK: declare i64 @llvm.hsail.bitmask.i64(i32, i32) #1
void test_bitmaskl(volatile global int* out, int x, int y)
{
  *out = __builtin_hsail_bitmaskl(x, y);
  *out = __builtin_hsail_bitmaskl(1, y);
  *out = __builtin_hsail_bitmaskl(x, 2);
}

// CHECK-LABEL: @test_bitrev(
// CHECK: tail call i32 @llvm.hsail.bitrev.i32(i32 %x)

// CHECK: declare i32 @llvm.hsail.bitrev.i32(i32) #1
void test_bitrev(volatile global int* out, int x)
{
  *out = __builtin_hsail_bitrev(x);
}

// CHECK-LABEL: @test_bitrevl(
// CHECK: tail call i64 @llvm.hsail.bitrev.i64(i64 %x)
// CHECK: declare i64 @llvm.hsail.bitrev.i64(i64) #1
void test_bitrevl(volatile global long* out, long x)
{
  *out = __builtin_hsail_bitrevl(x);
}

// CHECK-LABEL: @test_bitselect(
// CHECK: tail call i32 @llvm.hsail.bitselect.i32(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.bitselect.i32(i32, i32, i32) #1
void test_bitselect(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_bitselect(x, y, z);
}

// CHECK-LABEL: @test_bitselectl(
// CHECK: tail call i64 @llvm.hsail.bitselect.i64(i64 %x, i64 %y, i64 %z)

// CHECK: declare i64 @llvm.hsail.bitselect.i64(i64, i64, i64) #1
void test_bitselectl(volatile global long* out, long x, long y, long z)
{
  *out = __builtin_hsail_bitselectl(x, y, z);
}

// CHECK-LABEL: @test_sfirstbit(
// CHECK: tail call i32 @llvm.hsail.sfirstbit.i32(i32 %x)

// CHECK: declare i32 @llvm.hsail.sfirstbit.i32(i32) #1
void test_sfirstbit(volatile global int* out, int x)
{
  *out = __builtin_hsail_sfirstbit(x);
}

// CHECK-LABEL: @test_sfirstbitl(
// CHECK: tail call i32 @llvm.hsail.sfirstbit.i64(i64 %x)

// CHECK: declare i32 @llvm.hsail.sfirstbit.i64(i64) #1
void test_sfirstbitl(volatile global int* out, long x)
{
  *out = __builtin_hsail_sfirstbitl(x);
}

// CHECK-LABEL: @test_ufirstbit(
// CHECK: tail call i32 @llvm.hsail.ufirstbit.i32(i32 %x)

// CHECK: declare i32 @llvm.hsail.ufirstbit.i32(i32) #1
void test_ufirstbit(volatile global int* out, int x)
{
  *out = __builtin_hsail_ufirstbit(x);
}

// CHECK-LABEL: @test_ufirstbitl(
// CHECK: tail call i32 @llvm.hsail.ufirstbit.i64(i64 %x)

// CHECK: declare i32 @llvm.hsail.ufirstbit.i64(i64) #1
void test_ufirstbitl(volatile global int* out, long x)
{
  *out = __builtin_hsail_ufirstbitl(x);
}

// CHECK-LABEL: @test_lastbit(
// CHECK: tail call i32 @llvm.hsail.lastbit.i32(i32 %x)

// CHECK: declare i32 @llvm.hsail.lastbit.i32(i32) #1
void test_lastbit(volatile global int* out, int x)
{
  *out = __builtin_hsail_lastbit(x);
}

// CHECK-LABEL: @test_lastbitl(
// CHECK: tail call i32 @llvm.hsail.lastbit.i64(i64 %x)

// CHECK: declare i32 @llvm.hsail.lastbit.i64(i64) #1
void test_lastbitl(volatile global int* out, long x)
{
  *out = __builtin_hsail_lastbitl(x);
}

// CHECK-LABEL: @test_fadd_f64(
// CHECK: call double @llvm.hsail.fadd.f64(i1 false, i32 0, double %x, double %y)
// CHECK: call double @llvm.hsail.fadd.f64(i1 true, i32 0, double %x, double %y)
// CHECK: call double @llvm.hsail.fadd.f64(i1 true, i32 2, double %x, double %y)

// CHECK: declare double @llvm.hsail.fadd.f64(i1, i32, double, double) #1
void test_fadd_f64(volatile global double* out, double x, double y)
{
  *out = __builtin_hsail_fadd(0, 0, x, y);
  *out = __builtin_hsail_fadd(1, 0, x, y);
  *out = __builtin_hsail_fadd(1, 2, x, y);
}

// CHECK-LABEL: @test_fadd_f32(
// CHECK: call float @llvm.hsail.fadd.f32(i1 false, i32 0, float %x, float %y)
// CHECK: call float @llvm.hsail.fadd.f32(i1 true, i32 0, float %x, float %y)
// CHECK: call float @llvm.hsail.fadd.f32(i1 true, i32 3, float %x, float %y)

// CHECK: declare float @llvm.hsail.fadd.f32(i1, i32, float, float) #1
void test_fadd_f32(volatile global float* out, float x, float y)
{
  *out = __builtin_hsail_faddf(0, 0, x, y);
  *out = __builtin_hsail_faddf(1, 0, x, y);
  *out = __builtin_hsail_faddf(1, 3, x, y);
}

// CHECK-LABEL: @test_ceil_f64(
// CHECK: call double @llvm.hsail.fceil.f64(i1 false, double %x)
// CHECK: call double @llvm.hsail.fceil.f64(i1 true, double %x)

// CHECK: declare double @llvm.hsail.fceil.f64(i1, double) #1
void test_ceil_f64(volatile global double* out, double x)
{
  *out = __builtin_hsail_fceil(0, x);
  *out = __builtin_hsail_fceil(1, x);
}

// CHECK-LABEL: @test_ceil_f32(
// CHECK: call float @llvm.hsail.fceil.f32(i1 false, float %x)
// CHECK: call float @llvm.hsail.fceil.f32(i1 true, float %x)

// CHECK: declare float @llvm.hsail.fceil.f32(i1, float) #1
void test_ceil_f32(volatile global float* out, float x)
{
  *out = __builtin_hsail_fceilf(0, x);
  *out = __builtin_hsail_fceilf(1, x);
}

// CHECK-LABEL: @test_fdiv_f64(
// CHECK: call double @llvm.hsail.fdiv.f64(i1 false, i32 0, double %x, double %y)
// CHECK: call double @llvm.hsail.fdiv.f64(i1 true, i32 0, double %x, double %y)
// CHECK: call double @llvm.hsail.fdiv.f64(i1 true, i32 2, double %x, double %y)

// CHECK: declare double @llvm.hsail.fdiv.f64(i1, i32, double, double) #1
void test_fdiv_f64(volatile global double* out, double x, double y)
{
  *out = __builtin_hsail_fdiv(0, 0, x, y);
  *out = __builtin_hsail_fdiv(1, 0, x, y);
  *out = __builtin_hsail_fdiv(1, 2, x, y);
}

// CHECK-LABEL: @test_fdiv_f32(
// CHECK: call float @llvm.hsail.fdiv.f32(i1 false, i32 0, float %x, float %y)
// CHECK: call float @llvm.hsail.fdiv.f32(i1 true, i32 0, float %x, float %y)
// CHECK: call float @llvm.hsail.fdiv.f32(i1 true, i32 3, float %x, float %y)

// CHECK: declare float @llvm.hsail.fdiv.f32(i1, i32, float, float) #1
void test_fdiv_f32(volatile global float* out, float x, float y)
{
  *out = __builtin_hsail_fdivf(0, 0, x, y);
  *out = __builtin_hsail_fdivf(1, 0, x, y);
  *out = __builtin_hsail_fdivf(1, 3, x, y);
}

// CHECK-LABEL: @test_floor_f64(
// CHECK: call double @llvm.hsail.ffloor.f64(i1 false, double %x)
// CHECK: call double @llvm.hsail.ffloor.f64(i1 true, double %x)

// CHECK: declare double @llvm.hsail.ffloor.f64(i1, double) #1
void test_floor_f64(volatile global double* out, double x)
{
  *out = __builtin_hsail_ffloor(0, x);
  *out = __builtin_hsail_ffloor(1, x);
}

// CHECK-LABEL: @test_floor_f32(
// CHECK: call float @llvm.hsail.ffloor.f32(i1 false, float %x)
// CHECK: call float @llvm.hsail.ffloor.f32(i1 true, float %x)

// CHECK: declare float @llvm.hsail.ffloor.f32(i1, float) #1
void test_floor_f32(volatile global float* out, float x)
{
  *out = __builtin_hsail_ffloorf(0, x);
  *out = __builtin_hsail_ffloorf(1, x);
}

// CHECK-LABEL: @test_ffma_f64(
// CHECK: call double @llvm.hsail.ffma.f64(i1 false, i32 0, double %x, double %y, double %z)
// CHECK: call double @llvm.hsail.ffma.f64(i1 true, i32 0, double %x, double %y, double %z)
// CHECK: call double @llvm.hsail.ffma.f64(i1 true, i32 2, double %x, double %y, double %z)

// CHECK: declare double @llvm.hsail.ffma.f64(i1, i32, double, double, double) #1
void test_ffma_f64(volatile global double* out, double x, double y, double z)
{
  *out = __builtin_hsail_ffma(0, 0, x, y, z);
  *out = __builtin_hsail_ffma(1, 0, x, y, z);
  *out = __builtin_hsail_ffma(1, 2, x, y, z);
}

// CHECK-LABEL: @test_ffma_f32(
// CHECK: call float @llvm.hsail.ffma.f32(i1 false, i32 0, float %x, float %y, float %z)
// CHECK: call float @llvm.hsail.ffma.f32(i1 true, i32 0, float %x, float %y, float %z)
// CHECK: call float @llvm.hsail.ffma.f32(i1 true, i32 3, float %x, float %y, float %z)

// CHECK: declare float @llvm.hsail.ffma.f32(i1, i32, float, float, float) #1
void test_ffma_f32(volatile global float* out, float x, float y, float z)
{
  *out = __builtin_hsail_ffmaf(0, 0, x, y, z);
  *out = __builtin_hsail_ffmaf(1, 0, x, y, z);
  *out = __builtin_hsail_ffmaf(1, 3, x, y, z);
}

// CHECK-LABEL: @test_fract_f64(
// CHECK: call double @llvm.hsail.ffract.f64(i1 false, double %x)
// CHECK: call double @llvm.hsail.ffract.f64(i1 true, double %x)

// CHECK: declare double @llvm.hsail.ffract.f64(i1, double) #1
void test_fract_f64(volatile global double* out, double x)
{
  *out = __builtin_hsail_ffract(0, x);
  *out = __builtin_hsail_ffract(1, x);
}

// CHECK-LABEL: @test_fract_f32(
// CHECK: call float @llvm.hsail.ffract.f32(i1 false, float %x)
// CHECK: call float @llvm.hsail.ffract.f32(i1 true, float %x)

// CHECK: declare float @llvm.hsail.ffract.f32(i1, float) #1
void test_fract_f32(volatile global float* out, float x)
{
  *out = __builtin_hsail_ffractf(0, x);
  *out = __builtin_hsail_ffractf(1, x);
}

// CHECK-LABEL: @test_fmax_f64(
// CHECK: call double @llvm.hsail.fmax.f64(i1 false, double %x, double %y)
// CHECK: call double @llvm.hsail.fmax.f64(i1 true, double %x, double %y)

// CHECK: declare double @llvm.hsail.fmax.f64(i1, double, double) #1
void test_fmax_f64(volatile global double* out, double x, double y)
{
  *out = __builtin_hsail_fmax(0, x, y);
  *out = __builtin_hsail_fmax(1, x, y);
}

// CHECK-LABEL: @test_fmax_f32(
// CHECK: call float @llvm.hsail.fmax.f32(i1 false, float %x, float %y)
// CHECK: call float @llvm.hsail.fmax.f32(i1 true, float %x, float %y)

// CHECK: declare float @llvm.hsail.fmax.f32(i1, float, float) #1
void test_fmax_f32(volatile global float* out, float x, float y)
{
  *out = __builtin_hsail_fmaxf(0, x, y);
  *out = __builtin_hsail_fmaxf(1, x, y);
}

// CHECK-LABEL: @test_fmin_f64(
// CHECK: call double @llvm.hsail.fmin.f64(i1 false, double %x, double %y)
// CHECK: call double @llvm.hsail.fmin.f64(i1 true, double %x, double %y)

// CHECK: declare double @llvm.hsail.fmin.f64(i1, double, double) #1
void test_fmin_f64(volatile global double* out, double x, double y)
{
  *out = __builtin_hsail_fmin(0, x, y);
  *out = __builtin_hsail_fmin(1, x, y);
}

// CHECK-LABEL: @test_fmin_f32(
// CHECK: call float @llvm.hsail.fmin.f32(i1 false, float %x, float %y)
// CHECK: call float @llvm.hsail.fmin.f32(i1 true, float %x, float %y)

// CHECK: declare float @llvm.hsail.fmin.f32(i1, float, float) #1
void test_fmin_f32(volatile global float* out, float x, float y)
{
  *out = __builtin_hsail_fminf(0, x, y);
  *out = __builtin_hsail_fminf(1, x, y);
}

// CHECK-LABEL: @test_fmul_f64(
// CHECK: call double @llvm.hsail.fmul.f64(i1 false, i32 0, double %x, double %y)
// CHECK: call double @llvm.hsail.fmul.f64(i1 true, i32 0, double %x, double %y)
// CHECK: call double @llvm.hsail.fmul.f64(i1 true, i32 2, double %x, double %y)

// CHECK: declare double @llvm.hsail.fmul.f64(i1, i32, double, double) #1
void test_fmul_f64(volatile global double* out, double x, double y)
{
  *out = __builtin_hsail_fmul(0, 0, x, y);
  *out = __builtin_hsail_fmul(1, 0, x, y);
  *out = __builtin_hsail_fmul(1, 2, x, y);
}

// CHECK-LABEL: @test_fmul_f32(
// CHECK: call float @llvm.hsail.fmul.f32(i1 false, i32 0, float %x, float %y)
// CHECK: call float @llvm.hsail.fmul.f32(i1 true, i32 0, float %x, float %y)
// CHECK: call float @llvm.hsail.fmul.f32(i1 true, i32 3, float %x, float %y)

// CHECK: declare float @llvm.hsail.fmul.f32(i1, i32, float, float) #1
void test_fmul_f32(volatile global float* out, float x, float y)
{
  *out = __builtin_hsail_fmulf(0, 0, x, y);
  *out = __builtin_hsail_fmulf(1, 0, x, y);
  *out = __builtin_hsail_fmulf(1, 3, x, y);
}

// CHECK-LABEL: @test_rint_f64(
// CHECK: call double @llvm.hsail.frint.f64(i1 false, double %x)
// CHECK: call double @llvm.hsail.frint.f64(i1 true, double %x)

// CHECK: declare double @llvm.hsail.frint.f64(i1, double) #1
void test_rint_f64(volatile global double* out, double x)
{
  *out = __builtin_hsail_frint(0, x);
  *out = __builtin_hsail_frint(1, x);
}

// CHECK-LABEL: @test_rint_f32(
// CHECK: call float @llvm.hsail.frint.f32(i1 false, float %x)
// CHECK: call float @llvm.hsail.frint.f32(i1 true, float %x)

// CHECK: declare float @llvm.hsail.frint.f32(i1, float) #1
void test_rint_f32(volatile global float* out, float x)
{
  *out = __builtin_hsail_frintf(0, x);
  *out = __builtin_hsail_frintf(1, x);
}

// CHECK-LABEL: @test_sqrt_f64(
// CHECK: call double @llvm.hsail.fsqrt.f64(i1 false, i32 0, double %x)
// CHECK: call double @llvm.hsail.fsqrt.f64(i1 true, i32 0, double %x)
// CHECK: call double @llvm.hsail.fsqrt.f64(i1 true, i32 2, double %x)

// CHECK: declare double @llvm.hsail.fsqrt.f64(i1, i32, double) #1
void test_sqrt_f64(volatile global double* out, double x)
{
  *out = __builtin_hsail_fsqrt(0, 0, x);
  *out = __builtin_hsail_fsqrt(1, 0, x);
  *out = __builtin_hsail_fsqrt(1, 2, x);
}

// CHECK-LABEL: @test_sqrt_f32(
// CHECK: call float @llvm.hsail.fsqrt.f32(i1 false, i32 0, float %x)
// CHECK: call float @llvm.hsail.fsqrt.f32(i1 true, i32 0, float %x)
// CHECK: call float @llvm.hsail.fsqrt.f32(i1 true, i32 3, float %x)

// CHECK: declare float @llvm.hsail.fsqrt.f32(i1, i32, float) #1
void test_sqrt_f32(volatile global float* out, float x)
{
  *out = __builtin_hsail_fsqrtf(0, 0, x);
  *out = __builtin_hsail_fsqrtf(1, 0, x);
  *out = __builtin_hsail_fsqrtf(1, 3, x);
}

// CHECK-LABEL: @test_fsub_f64(
// CHECK: call double @llvm.hsail.fsub.f64(i1 false, i32 0, double %x, double %y)
// CHECK: call double @llvm.hsail.fsub.f64(i1 true, i32 0, double %x, double %y)
// CHECK: call double @llvm.hsail.fsub.f64(i1 true, i32 2, double %x, double %y)

// CHECK: declare double @llvm.hsail.fsub.f64(i1, i32, double, double) #1
void test_fsub_f64(volatile global double* out, double x, double y)
{
  *out = __builtin_hsail_fsub(0, 0, x, y);
  *out = __builtin_hsail_fsub(1, 0, x, y);
  *out = __builtin_hsail_fsub(1, 2, x, y);
}

// CHECK-LABEL: @test_fsub_f32(
// CHECK: call float @llvm.hsail.fsub.f32(i1 false, i32 0, float %x, float %y)
// CHECK: call float @llvm.hsail.fsub.f32(i1 true, i32 0, float %x, float %y)
// CHECK: call float @llvm.hsail.fsub.f32(i1 true, i32 3, float %x, float %y)

// CHECK: declare float @llvm.hsail.fsub.f32(i1, i32, float, float) #1
void test_fsub_f32(volatile global float* out, float x, float y)
{
  *out = __builtin_hsail_fsubf(0, 0, x, y);
  *out = __builtin_hsail_fsubf(1, 0, x, y);
  *out = __builtin_hsail_fsubf(1, 3, x, y);
}

// CHECK-LABEL: @test_fmad_f64(
// CHECK: call double @llvm.hsail.fmad.f64(i1 false, i32 0, double %x, double %y, double %z)
// CHECK: call double @llvm.hsail.fmad.f64(i1 true, i32 0, double %x, double %y, double %z)
// CHECK: call double @llvm.hsail.fmad.f64(i1 true, i32 2, double %x, double %y, double %z)

// CHECK: declare double @llvm.hsail.fmad.f64(i1, i32, double, double, double) #1
void test_fmad_f64(volatile global double* out, double x, double y, double z)
{
  *out = __builtin_hsail_fmad(0, 0, x, y, z);
  *out = __builtin_hsail_fmad(1, 0, x, y, z);
  *out = __builtin_hsail_fmad(1, 2, x, y, z);
}

// CHECK-LABEL: @test_fmad_f32(
// CHECK: call float @llvm.hsail.fmad.f32(i1 false, i32 0, float %x, float %y, float %z)
// CHECK: call float @llvm.hsail.fmad.f32(i1 true, i32 0, float %x, float %y, float %z)
// CHECK: call float @llvm.hsail.fmad.f32(i1 true, i32 3, float %x, float %y, float %z)

// CHECK: declare float @llvm.hsail.fmad.f32(i1, i32, float, float, float) #1
void test_fmad_f32(volatile global float* out, float x, float y, float z)
{
  *out = __builtin_hsail_fmadf(0, 0, x, y, z);
  *out = __builtin_hsail_fmadf(1, 0, x, y, z);
  *out = __builtin_hsail_fmadf(1, 3, x, y, z);
}

// CHECK-LABEL: @test_class_f32(
// CHECK: call i1 @llvm.hsail.class.f32(float %x, i32 1)
// CHECK: call i1 @llvm.hsail.class.f32(float %x, i32 %y)

// CHECK: declare i1 @llvm.hsail.class.f32(float, i32) #1
void test_class_f32(volatile global int* out, float x, int y)
{
  *out = __builtin_hsail_classf(x, 1);
  *out = __builtin_hsail_classf(x, y);

}

// CHECK-LABEL: @test_class_f64(
// CHECK: call i1 @llvm.hsail.class.f64(double %x, i32 1)
// CHECK: call i1 @llvm.hsail.class.f64(double %x, i32 %y)

// CHECK: declare i1 @llvm.hsail.class.f64(double, i32) #1
void test_class_f64(volatile global int* out, double x, int y)
{
  *out = __builtin_hsail_class(x, 1);
  *out = __builtin_hsail_class(x, y);
}


// CHECK-LABEL: @test_ncosf(
// CHECK: tail call float @llvm.hsail.ncos.f32(float %x)

// CHECK: declare float @llvm.hsail.ncos.f32(float) #1
void test_ncosf(volatile global float* out, float x)
{
  *out = __builtin_hsail_ncosf(x);
}

// CHECK-LABEL: @test_nexp2f(
// CHECK: tail call float @llvm.hsail.nexp2.f32(float %x)

// CHECK: declare float @llvm.hsail.nexp2.f32(float) #1
void test_nexp2f(volatile global float* out, float x)
{
  *out = __builtin_hsail_nexp2f(x);
}

// CHECK-LABEL: @test_nfma(
// CHECK: tail call double @llvm.hsail.nfma.f64(double %x, double %y, double %z)

// CHECK: declare double @llvm.hsail.nfma.f64(double, double, double) #1
void test_nfma(volatile global double* out, double x, double y, double z)
{
  *out = __builtin_hsail_nfma(x, y, z);
}

// CHECK-LABEL: @test_nfmaf(
// CHECK: tail call float @llvm.hsail.nfma.f32(float %x, float %y, float %z)

// CHECK: declare float @llvm.hsail.nfma.f32(float, float, float) #1
void test_nfmaf(volatile global float* out, float x, float y, float z)
{
  *out = __builtin_hsail_nfmaf(x, y, z);
}

// CHECK-LABEL: @test_nrcpf(
// CHECK: tail call float @llvm.hsail.nrcp.f32(float %x)

// CHECK: declare float @llvm.hsail.nrcp.f32(float) #1
void test_nrcpf(volatile global float* out, float x)
{
  *out = __builtin_hsail_nrcpf(x);
}

// CHECK-LABEL: @test_nrsqrtf(
// CHECK: tail call float @llvm.hsail.nrsqrt.f32(float %x)

// CHECK: declare float @llvm.hsail.nrsqrt.f32(float) #1
void test_nrsqrtf(volatile global float* out, float x)
{
  *out = __builtin_hsail_nrsqrtf(x);
}

// CHECK-LABEL: @test_nsinf(
// CHECK: tail call float @llvm.hsail.nsin.f32(float %x)

// CHECK: declare float @llvm.hsail.nsin.f32(float) #1
void test_nsinf(volatile global float* out, float x)
{
  *out = __builtin_hsail_nsinf(x);
}

// CHECK-LABEL: @test_nsqrtf(
// CHECK: tail call float @llvm.hsail.nsqrt.f32(float %x)

// CHECK: declare float @llvm.hsail.nsqrt.f32(float) #1
void test_nsqrtf(volatile global float* out, float x)
{
  *out = __builtin_hsail_nsqrtf(x);
}

// CHECK-LABEL: @test_bitalign(
// CHECK: tail call i32 @llvm.hsail.bitalign(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.bitalign(i32, i32, i32) #1
void test_bitalign(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_bitalign(x, y, z);
}

// CHECK-LABEL: @test_bytealign(
// CHECK: tail call i32 @llvm.hsail.bytealign(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.bytealign(i32, i32, i32) #1
void test_bytealign(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_bytealign(x, y, z);
}

// CHECK-LABEL: @test_lerp(
// CHECK: tail call i32 @llvm.hsail.lerp(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.lerp(i32, i32, i32) #1
void test_lerp(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_lerp(x, y, z);
}

// CHECK-LABEL: @test_packcvt(
// CHECK: tail call i32 @llvm.hsail.packcvt(float %x, float %y, float %z, float %w)

// declare i32 @llvm.hsail.packcvt(float, float, float, float) #1
void test_packcvt(volatile global int* out, float x, float y, float z, float w)
{
  *out = __builtin_hsail_packcvt(x, y, z, w);
}

// CHECK-LABEL: @test_unpackcvt(
// CHECK: tail call float @llvm.hsail.unpackcvt(i32 %x, i32 0)
// CHECK: tail call float @llvm.hsail.unpackcvt(i32 %x, i32 1)
// CHECK: tail call float @llvm.hsail.unpackcvt(i32 %x, i32 2)
// CHECK: tail call float @llvm.hsail.unpackcvt(i32 %x, i32 3)

// CHECK: declare float @llvm.hsail.unpackcvt(i32, i32) #1
void test_unpackcvt(volatile global float* out, int x)
{
  *out = __builtin_hsail_unpackcvt(x, 0);
  *out = __builtin_hsail_unpackcvt(x, 1);
  *out = __builtin_hsail_unpackcvt(x, 2);
  *out = __builtin_hsail_unpackcvt(x, 3);
}

// CHECK-LABEL: @test_sad_u32_u32(
// CHECK: tail call i32 @llvm.hsail.sad.u32.u32(i32 %x, i32 %y, i32 %z)
// CHECK: tail call i32 @llvm.hsail.sad.u32.u32(i32 1, i32 %y, i32 %z)
// CHECK: tail call i32 @llvm.hsail.sad.u32.u32(i32 %x, i32 2, i32 %z)
// CHECK: tail call i32 @llvm.hsail.sad.u32.u32(i32 %x, i32 %y, i32 3)

// CHECK: declare i32 @llvm.hsail.sad.u32.u32(i32, i32, i32) #1
void test_sad_u32_u32(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_sad_u32_u32(x, y, z);
  *out = __builtin_hsail_sad_u32_u32(1, y, z);
  *out = __builtin_hsail_sad_u32_u32(x, 2, z);
  *out = __builtin_hsail_sad_u32_u32(x, y, 3);
}

// CHECK-LABEL: @test_sad_u32_u16x2(
// CHECK: tail call i32 @llvm.hsail.sad.u32.u16x2(i32 %x, i32 %y, i32 %z)
// CHECK: tail call i32 @llvm.hsail.sad.u32.u16x2(i32 1, i32 %y, i32 %z)
// CHECK: tail call i32 @llvm.hsail.sad.u32.u16x2(i32 %x, i32 2, i32 %z)
// CHECK: tail call i32 @llvm.hsail.sad.u32.u16x2(i32 %x, i32 %y, i32 3)

// CHECK: declare i32 @llvm.hsail.sad.u32.u16x2(i32, i32, i32) #1
void test_sad_u32_u16x2(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_sad_u32_u16x2(x, y, z);
  *out = __builtin_hsail_sad_u32_u16x2(1, y, z);
  *out = __builtin_hsail_sad_u32_u16x2(x, 2, z);
  *out = __builtin_hsail_sad_u32_u16x2(x, y, 3);
}

// CHECK-LABEL: @test_sad_u32_u8x4(
// CHECK: tail call i32 @llvm.hsail.sad.u32.u8x4(i32 %x, i32 %y, i32 %z)
// CHECK: tail call i32 @llvm.hsail.sad.u32.u8x4(i32 1, i32 %y, i32 %z)
// CHECK: tail call i32 @llvm.hsail.sad.u32.u8x4(i32 %x, i32 2, i32 %z)
// CHECK: tail call i32 @llvm.hsail.sad.u32.u8x4(i32 %x, i32 %y, i32 3)

// CHECK: declare i32 @llvm.hsail.sad.u32.u8x4(i32, i32, i32) #1
void test_sad_u32_u8x4(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_sad_u32_u8x4(x, y, z);
  *out = __builtin_hsail_sad_u32_u8x4(1, y, z);
  *out = __builtin_hsail_sad_u32_u8x4(x, 2, z);
  *out = __builtin_hsail_sad_u32_u8x4(x, y, 3);
}

// CHECK-LABEL: @test_sadhi(
// CHECK: tail call i32 @llvm.hsail.sadhi(i32 %x, i32 %y, i32 %z)
// CHECK: tail call i32 @llvm.hsail.sadhi(i32 1, i32 %y, i32 %z)
// CHECK: tail call i32 @llvm.hsail.sadhi(i32 %x, i32 2, i32 %z)
// CHECK: tail call i32 @llvm.hsail.sadhi(i32 %x, i32 %y, i32 3)

// CHECK: declare i32 @llvm.hsail.sadhi(i32, i32, i32) #1
void test_sadhi(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_sadhi(x, y, z);
  *out = __builtin_hsail_sadhi(1, y, z);
  *out = __builtin_hsail_sadhi(x, 2, z);
  *out = __builtin_hsail_sadhi(x, y, 3);
}

// CHECK-LABLE: @test_segmentp(
// CHECK: tail call i1 @llvm.hsail.segmentp(i32 0, i1 false, i8 addrspace(4)* null)
// CHECK: tail call i1 @llvm.hsail.segmentp(i32 1, i1 false, i8 addrspace(4)* null)
// CHECK: tail call i1 @llvm.hsail.segmentp(i32 3, i1 false, i8 addrspace(4)* null)
// CHECK: tail call i1 @llvm.hsail.segmentp(i32 3, i1 true, i8 addrspace(4)* null)

// CHECK: declare i1 @llvm.hsail.segmentp(i32, i1, i8 addrspace(4)*) #1
void test_segmentp(volatile global int* out)
{
  typedef __attribute__((address_space(4))) char* flat_ptr;

  *out = __builtin_hsail_segmentp(0, false, (flat_ptr)0);
  *out = __builtin_hsail_segmentp(1, false, (flat_ptr)0);
  *out = __builtin_hsail_segmentp(3, false, (flat_ptr)0);
  *out = __builtin_hsail_segmentp(3, true, (flat_ptr)0);
}

// CHECK-LABEL: @test_memfence(
// CHECK: tail call void @llvm.hsail.memfence(i32 0, i32 0)

// CHECK: declare void @llvm.hsail.memfence(i32, i32) #2
void test_memfence()
{
  __builtin_hsail_memfence(0, 0);
}

// CHECK-LABEL: @test_imagefence(
// CHECK: tail call void @llvm.hsail.imagefence()

// CHECK: declare void @llvm.hsail.imagefence() #2
void test_imagefence()
{
  __builtin_hsail_imagefence();
}

// CHECK-LABEL: @test_barrier(
// CHECK: tail call void @llvm.hsail.barrier(i32 1)
// CHECK: tail call void @llvm.hsail.barrier(i32 33)
// CHECK: tail call void @llvm.hsail.barrier(i32 34)
void test_barrier()
{
  __builtin_hsail_barrier(1);
  __builtin_hsail_barrier(33);
  __builtin_hsail_barrier(34);
}

// CHECK: declare void @llvm.hsail.barrier(i32) #3

// CHECK-LABEL: @test_wavebarrier(
// CHECK: tail call void @llvm.hsail.wavebarrier()
// CHECK: tail call void @llvm.hsail.wavebarrier()
// CHECK: tail call void @llvm.hsail.wavebarrier()

// CHECK: declare void @llvm.hsail.wavebarrier() #3
void test_wavebarrier()
{
  __builtin_hsail_wavebarrier();
  __builtin_hsail_wavebarrier();
  __builtin_hsail_wavebarrier();
}

// CHECK-LABEL: @test_activelanecount(
// CHECK: tail call i32 @llvm.hsail.activelanecount(i32 1, i1 false)
// CHECK: tail call i32 @llvm.hsail.activelanecount(i32 34, i1 %tobool)

// CHECK: declare i32 @llvm.hsail.activelanecount(i32, i1) #4
void test_activelanecount(volatile int* out, int x)
{
  *out = __builtin_hsail_activelanecount(1, 0);
  *out = __builtin_hsail_activelanecount(34, x);
}


// CHECK-LABEL: @test_activelaneid(
// CHECK: tail call i32 @llvm.hsail.activelaneid(i32 1)
// CHECK: tail call i32 @llvm.hsail.activelaneid(i32 34)

// CHECK: declare i32 @llvm.hsail.activelaneid(i32) #5
void test_activelaneid(volatile int* out, int x)
{
  *out = __builtin_hsail_activelaneid(1);
  *out = __builtin_hsail_activelaneid(34);
}

// CHECK-LABEL: @test_activelanemask(
// CHECK: [[CALL0:%[0-9]+]] = tail call { i64, i64, i64, i64 } @llvm.hsail.activelanemask(i32 34, i1 true)
// CHECK-DAG: [[CALL0_ELT0:%[0-9]+]] = extractvalue { i64, i64, i64, i64 } [[CALL0]], 0
// CHECK-DAG: [[CALL0_ELT1:%[0-9]+]] = extractvalue { i64, i64, i64, i64 } [[CALL0]], 1
// CHECK-DAG: [[CALL0_ELT2:%[0-9]+]] = extractvalue { i64, i64, i64, i64 } [[CALL0]], 2
// CHECK-DAG: [[CALL0_ELT3:%[0-9]+]] = extractvalue { i64, i64, i64, i64 } [[CALL0]], 3

// CHECK-DAG: [[CALL0_INSERT0:%[0-9]+]] = insertelement <4 x i64> undef, i64 [[CALL0_ELT0]], i32 0
// CHECK-DAG: [[CALL0_INSERT1:%[0-9]+]] = insertelement <4 x i64> [[CALL0_INSERT0]], i64 [[CALL0_ELT1]], i32 1
// CHECK-DAG: [[CALL0_INSERT2:%[0-9]+]] = insertelement <4 x i64> [[CALL0_INSERT1]], i64 [[CALL0_ELT2]], i32 2
// CHECK-DAG: [[CALL0_INSERT3:%[0-9]+]] = insertelement <4 x i64> [[CALL0_INSERT2]], i64 [[CALL0_ELT3]], i32 3
// CHECK: store volatile <4 x i64> [[CALL0_INSERT3]],

// CHECK: tail call { i64, i64, i64, i64 } @llvm.hsail.activelanemask(i32 1, i1 false)
// CHECK: tail call { i64, i64, i64, i64 } @llvm.hsail.activelanemask(i32 34, i1 %tobool)

// CHECK: declare { i64, i64, i64, i64 } @llvm.hsail.activelanemask(i32, i1) #4
void test_activelanemask(volatile global long4* out, int x)
{
  *out = __builtin_hsail_activelanemask(34, true);
  *out = __builtin_hsail_activelanemask(1, false);
  *out = __builtin_hsail_activelanemask(34, x);
}

// CHECK-LABEL: @test_activelanepermute(
// CHECK: tail call i32 @llvm.hsail.activelanepermute.i32(i32 31, i32 %x, i32 %y, i32 %z, i1 %tobool)
// CHECK: tail call i32 @llvm.hsail.activelanepermute.i32(i32 31, i32 42, i32 %y, i32 %z, i1 %tobool)
// CHECK: tail call i32 @llvm.hsail.activelanepermute.i32(i32 31, i32 %x, i32 7, i32 %z, i1 %tobool)
// CHECK: tail call i32 @llvm.hsail.activelanepermute.i32(i32 31, i32 %x, i32 %y, i32 9, i1 %tobool)
// CHECK: tail call i32 @llvm.hsail.activelanepermute.i32(i32 31, i32 %x, i32 %y, i32 %z, i1 true)
void test_activelanepermute(volatile global int* out, int x, int y, int z, int w)
{
  *out = __builtin_hsail_activelanepermute(31, x, y, z, w);
  *out = __builtin_hsail_activelanepermute(31, 42, y, z, w);
  *out = __builtin_hsail_activelanepermute(31, x, 7, z, w);
  *out = __builtin_hsail_activelanepermute(31, x, y, 9, w);
  *out = __builtin_hsail_activelanepermute(31, x, y, z, true);
}

// CHECK: declare i32 @llvm.hsail.activelanepermute.i32(i32, i32, i32, i32, i1) #6

// CHECK-LABEL: @test_activelanepermutel(
// CHECK: tail call i64 @llvm.hsail.activelanepermute.i64(i32 31, i64 %conv, i32 %conv1, i64 %z, i1 %tobool)
// CHECK: tail call i64 @llvm.hsail.activelanepermute.i64(i32 31, i64 42, i32 %conv1, i64 %z, i1 %tobool)
// CHECK: tail call i64 @llvm.hsail.activelanepermute.i64(i32 31, i64 %conv, i32 7, i64 %z, i1 %tobool)
// CHECK: tail call i64 @llvm.hsail.activelanepermute.i64(i32 31, i64 %conv, i32 %conv1, i64 9, i1 %tobool)
// CHECK: tail call i64 @llvm.hsail.activelanepermute.i64(i32 31, i64 %conv, i32 %conv1, i64 %z, i1 true)

// CHECK: declare i64 @llvm.hsail.activelanepermute.i64(i32, i64, i32, i64, i1) #6
void test_activelanepermutel(volatile global long* out, int x, long y, long z, int w)
{
  *out = __builtin_hsail_activelanepermutel(31, x, y, z, w);
  *out = __builtin_hsail_activelanepermutel(31, 42, y, z, w);
  *out = __builtin_hsail_activelanepermutel(31, x, 7, z, w);
  *out = __builtin_hsail_activelanepermutel(31, x, y, 9, w);
  *out = __builtin_hsail_activelanepermutel(31, x, y, z, true);
}

// CHECK-LABEL: @test_currentworkgroupsize(
// CHECK: tail call i32 @llvm.hsail.currentworkgroupsize(i32 0)
// CHECK: tail call i32 @llvm.hsail.currentworkgroupsize(i32 1)
// CHECK: tail call i32 @llvm.hsail.currentworkgroupsize(i32 2)

// CHECK: declare i32 @llvm.hsail.currentworkgroupsize(i32) #1
void test_currentworkgroupsize(volatile global int* out)
{
  *out = __builtin_hsail_currentworkgroupsize(0);
  *out = __builtin_hsail_currentworkgroupsize(1);
  *out = __builtin_hsail_currentworkgroupsize(2);
}

// CHECK-LABEL: @test_currentworkitemflatid(
// CHECK: tail call i32 @llvm.hsail.currentworkitemflatid()

// CHECK: declare i32 @llvm.hsail.currentworkitemflatid() #1
void test_currentworkitemflatid(volatile global int* out)
{
  *out = __builtin_hsail_currentworkitemflatid();
}

// CHECK-LABEL: @test_dim(
// CHECK: tail call i32 @llvm.hsail.dim()

// CHECK: declare i32 @llvm.hsail.dim() #1
void test_dim(volatile global int* out)
{
  *out = __builtin_hsail_dim();
}

// CHECK-LABEL: @test_gridgroups(
// CHECK: tail call i32 @llvm.hsail.gridgroups(i32 0)
// CHECK: tail call i32 @llvm.hsail.gridgroups(i32 1)
// CHECK: tail call i32 @llvm.hsail.gridgroups(i32 2)

// CHECK: declare i32 @llvm.hsail.gridgroups(i32) #1
void test_gridgroups(volatile global int* out)
{
  *out = __builtin_hsail_gridgroups(0);
  *out = __builtin_hsail_gridgroups(1);
  *out = __builtin_hsail_gridgroups(2);
}

// CHECK-LABEL: @test_workgroupid(
// CHECK: tail call i32 @llvm.hsail.workgroupid(i32 0)
// CHECK: tail call i32 @llvm.hsail.workgroupid(i32 1)
// CHECK: tail call i32 @llvm.hsail.workgroupid(i32 2)

// CHECK: declare i32 @llvm.hsail.workgroupid(i32) #1
void test_workgroupid(volatile global int* out)
{
  *out = __builtin_hsail_workgroupid(0);
  *out = __builtin_hsail_workgroupid(1);
  *out = __builtin_hsail_workgroupid(2);
}

// CHECK-LABEL: @test_workgroupsize(
// CHECK: tail call i32 @llvm.hsail.workgroupsize(i32 0)
// CHECK: tail call i32 @llvm.hsail.workgroupsize(i32 1)
// CHECK: tail call i32 @llvm.hsail.workgroupsize(i32 2)

// CHECK: declare i32 @llvm.hsail.workgroupsize(i32) #1
void test_workgroupsize(volatile global int* out)
{
  *out = __builtin_hsail_workgroupsize(0);
  *out = __builtin_hsail_workgroupsize(1);
  *out = __builtin_hsail_workgroupsize(2);
}

// CHECK-LABEL: @test_workitemabsid(
// CHECK: tail call i32 @llvm.hsail.workitemabsid.i32(i32 0)
// CHECK: tail call i32 @llvm.hsail.workitemabsid.i32(i32 1)
// CHECK: tail call i32 @llvm.hsail.workitemabsid.i32(i32 2)

// CHECK: declare i32 @llvm.hsail.workitemabsid.i32(i32) #1
void test_workitemabsid(volatile global int* out)
{
  *out = __builtin_hsail_workitemabsid(0);
  *out = __builtin_hsail_workitemabsid(1);
  *out = __builtin_hsail_workitemabsid(2);
}

// CHECK-LABEL: @test_workitemabsidl(
// CHECK: tail call i64 @llvm.hsail.workitemabsid.i64(i32 0)
// CHECK: tail call i64 @llvm.hsail.workitemabsid.i64(i32 1)
// CHECK: tail call i64 @llvm.hsail.workitemabsid.i64(i32 2)

// CHECK: declare i64 @llvm.hsail.workitemabsid.i64(i32) #1
void test_workitemabsidl(volatile global long* out)
{
  *out = __builtin_hsail_workitemabsidl(0);
  *out = __builtin_hsail_workitemabsidl(1);
  *out = __builtin_hsail_workitemabsidl(2);
}

// CHECK-LABEL: @test_workitemflatabsid(
// CHECK: tail call i32 @llvm.hsail.workitemflatabsid.i32()

// CHECK: declare i32 @llvm.hsail.workitemflatabsid.i32() #1
void test_workitemflatabsid(volatile global int* out)
{
  *out = __builtin_hsail_workitemflatabsid();
}

// CHECK-LABEL: @test_workitemflatabsidl(
// CHECK: tail call i64 @llvm.hsail.workitemflatabsid.i64()

// CHECK: declare i64 @llvm.hsail.workitemflatabsid.i64() #1
void test_workitemflatabsidl(volatile global long* out)
{
  *out = __builtin_hsail_workitemflatabsidl();
}

// CHECK-LABEL: @test_workitemflatid(
// CHECK: tail call i32 @llvm.hsail.workitemflatid()

// CHECK: declare i32 @llvm.hsail.workitemflatid() #1
void test_workitemflatid(volatile global int* out)
{
  *out = __builtin_hsail_workitemflatid();
}

// CHECK-LABEL: @test_workitemid(
// CHECK: tail call i32 @llvm.hsail.workitemid(i32 0)
// CHECK: tail call i32 @llvm.hsail.workitemid(i32 1)
// CHECK: tail call i32 @llvm.hsail.workitemid(i32 2)

// CHECK: declare i32 @llvm.hsail.workitemid(i32) #1
void test_workitemid(volatile global int* out)
{
  *out = __builtin_hsail_workitemid(0);
  *out = __builtin_hsail_workitemid(1);
  *out = __builtin_hsail_workitemid(2);
}

// CHECK-LABEL: @test_clock(
// CHECK: tail call i64 @llvm.hsail.clock()
void test_clock(volatile long* out)
{
  *out = __builtin_hsail_clock();
}

// CHECK: declare i64 @llvm.hsail.clock() #2

// CHECK-LABEL: @test_cuid(
// CHECK: tail call i32 @llvm.hsail.cuid()

// CHECK: declare i32 @llvm.hsail.cuid() #1
void test_cuid(volatile int* out)
{
  *out = __builtin_hsail_cuid();
}

// CHECK-LABEL: @test_groupbaseptr(
// CHECK: tail call i8 addrspace(3)* @llvm.hsail.groupbaseptr()
// CHECK: load i8, i8 addrspace(3)*

// CHECK: declare i8 addrspace(3)* @llvm.hsail.groupbaseptr() #1
void test_groupbaseptr(volatile char* out)
{
  *out = *__builtin_hsail_groupbaseptr();
}

// CHECK-LABEL: @test_kernargbaseptr(
// CHECK: tail call i8 addrspace(7)* @llvm.hsail.kernargbaseptr()
// CHECK: load i8, i8 addrspace(7)*

// CHECK: declare i8 addrspace(7)* @llvm.hsail.kernargbaseptr() #1
void test_kernargbaseptr(volatile char* out)
{
  *out = *__builtin_hsail_kernargbaseptr();
}

// CHECK-LABEL: @test_laneid(
// CHECK: tail call i32 @llvm.hsail.laneid()
void test_laneid(volatile int* out)
{
  *out = __builtin_hsail_laneid();
}

// CHECK: declare i32 @llvm.hsail.laneid() #1

// CHECK-LABEL: @test_maxcuid(
// CHECK: tail call i32 @llvm.hsail.maxcuid()
void test_maxcuid(volatile int* out)
{
  *out = __builtin_hsail_maxcuid();
}

// CHECK: declare i32 @llvm.hsail.maxcuid() #1

// CHECK-LABEL: @test_maxwaveid(
// CHECK: tail call i32 @llvm.hsail.maxwaveid()
void test_maxwaveid(volatile int* out)
{
  *out = __builtin_hsail_maxwaveid();
}

// CHECK: declare i32 @llvm.hsail.maxwaveid() #1


// CHECK-LABEL: @test_waveid(
// CHECK: tail call i32 @llvm.hsail.waveid()
void test_waveid(volatile int* out)
{
  *out = __builtin_hsail_waveid();
}

// CHECK: declare i32 @llvm.hsail.waveid() #1

// CHECK-LABEL: @test_gcn_msad(
// CHECK: tail call i32 @llvm.hsail.gcn.msad(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.gcn.msad(i32, i32, i32) #1
void test_gcn_msad(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_gcn_msad(x, y, z);
}

// CHECK-LABEL: @test_gcn_qsad(
// CHECK: tail call i64 @llvm.hsail.gcn.qsad(i64 %x, i64 %y, i64 %z)

// CHECK: declare i64 @llvm.hsail.gcn.qsad(i64, i64, i64) #1
void test_gcn_qsad(volatile global long* out, long x, long y, long z)
{
  *out = __builtin_hsail_gcn_qsad(x, y, z);
}

// CHECK-LABEL: @test_gcn_mqsad(
// CHECK: tail call i64 @llvm.hsail.gcn.mqsad(i64 %x, i32 %y, i64 %z)

// CHECK: declare i64 @llvm.hsail.gcn.mqsad(i64, i32, i64) #1
void test_gcn_mqsad(volatile global long* out, long x, int y, long z)
{
  *out = __builtin_hsail_gcn_mqsad(x, y, z);
}

// CHECK-LABEL: @test_gcn_sadw(
// CHECK: tail call i32 @llvm.hsail.gcn.sadw(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.gcn.sadw(i32, i32, i32) #1
void test_gcn_sadw(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_gcn_sadw(x, y, z);
}

// CHECK-LABEL: @test_gcn_sadd(
// CHECK: tail call i32 @llvm.hsail.gcn.sadd(i32 %x, i32 %y, i32 %z)

// CHECK: declare i32 @llvm.hsail.gcn.sadd(i32, i32, i32) #1
void test_gcn_sadd(volatile global int* out, int x, int y, int z)
{
  *out = __builtin_hsail_gcn_sadd(x, y, z);
}

// CHECK: attributes #1 = { nounwind readnone }
// CHECK: attributes #2 = { nounwind }
// CHECK: attributes #3 = { convergent noduplicate nounwind }
// CHECK: attributes #4 = { convergent nounwind readonly }
// CHECK: attributes #5 = { nounwind readonly }
// CHECK: attributes #6 = { convergent nounwind }

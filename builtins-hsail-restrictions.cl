// REQUIRES: hsail-registered-target
// RUN: %clang_cc1 -triple hsail-unknown-unknown -verify -pedantic -fsyntax-only %s

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef __attribute__((ext_vector_type(4))) long long4;


void test_sqrt_builtin(volatile global double* out,
                       volatile float* outf,
                       double x,
                       float xf,
                       int var)
{
  *out = __builtin_hsail_fsqrt(var, 0, x); // expected-error {{argument to '__builtin_hsail_fsqrt' must be a constant integer}}
  *outf = __builtin_hsail_fsqrtf(var, 0, xf); // expected-error {{argument to '__builtin_hsail_fsqrtf' must be a constant integer}}

  *out = __builtin_hsail_fsqrt(0, var, x); // expected-error {{argument to '__builtin_hsail_fsqrt' must be a constant integer}}
  *outf = __builtin_hsail_fsqrtf(0, var, xf); // expected-error {{argument to '__builtin_hsail_fsqrtf' must be a constant integer}}
}

void test_unpackcvt(volatile global float* out, int x, int y)
{
  *out = __builtin_hsail_unpackcvt(x, y); // expected-error {{argument to '__builtin_hsail_unpackcvt' must be a constant integer}}
}

void test_segmentp(volatile global int* out, int x)
{
  *out = __builtin_hsail_segmentp(x, false, (__attribute__((address_space(4))) char*)0); // expected-error {{argument to '__builtin_hsail_segmentp' must be a constant integer}}
  *out = __builtin_hsail_segmentp(0, x, (__attribute__((address_space(4))) char*)0); // expected-error {{argument to '__builtin_hsail_segmentp' must be a constant integer}}
}

void test_memfence(int x)
{
  __builtin_hsail_memfence(x, 0); // expected-error {{argument to '__builtin_hsail_memfence' must be a constant integer}}
  __builtin_hsail_memfence(0, x); // expected-error {{argument to '__builtin_hsail_memfence' must be a constant integer}}
}

void test_barrier_builtin(int x)
{
  __builtin_hsail_barrier(x); // expected-error {{argument to '__builtin_hsail_barrier' must be a constant integer}}
}

void test_activelanecount(volatile global int* out, int x)
{
  *out = __builtin_hsail_activelanecount(x, 0); // expected-error {{argument to '__builtin_hsail_activelanecount' must be a constant integer}}
}

void test_activelaneid(volatile global int* out, int x)
{
  *out = __builtin_hsail_activelaneid(x); // expected-error {{argument to '__builtin_hsail_activelaneid' must be a constant integer}}
}

void test_activelanemask(volatile global long4* out, int x)
{
  *out = __builtin_hsail_activelanemask(x, false); // expected-error {{argument to '__builtin_hsail_activelanemask' must be a constant integer}}
}

void test_activelanepermute(volatile global int* out, int x)
{
  *out = __builtin_hsail_activelanepermute(x, 0, 0, 0, 0); // expected-error {{argument to '__builtin_hsail_activelanepermute' must be a constant integer}}
}

void test_activelanepermutel(volatile global long* out, int x)
{
  *out = __builtin_hsail_activelanepermutel(x, 0, 0, 0, 0); // expected-error {{argument to '__builtin_hsail_activelanepermutel' must be a constant integer}}
}

void test_currentworkgroupsize(volatile global int* out, int x)
{
  *out = __builtin_hsail_currentworkgroupsize(x); // expected-error {{argument to '__builtin_hsail_currentworkgroupsize' must be a constant integer}}
}

void test_workitemabsid(volatile global int* out, int x)
{
  *out = __builtin_hsail_workitemabsid(x); // expected-error {{argument to '__builtin_hsail_workitemabsid' must be a constant integer}}
}

void test_workitemabsidl(volatile global long* out, int x)
{
  *out = __builtin_hsail_workitemabsidl(x); // expected-error {{argument to '__builtin_hsail_workitemabsidl' must be a constant integer}}
}

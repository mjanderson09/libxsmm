/******************************************************************************
** Copyright (c) 2015-2018, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>

#include <stdlib.h>
#include <string.h>
#if defined(_DEBUG)
# include <stdio.h>
#endif

#if !defined(ITYPE)
# define ITYPE double
#endif
#if !defined(OTYPE)
# define OTYPE ITYPE
#endif
#if !defined(CHECK_FPE) && 0
# define CHECK_FPE
#endif
#if !defined(REFERENCE_BLAS)
# define REFERENCE_BLAS LIBXSMM_GEMM_SYMBOL
#endif
#if !defined(LIBXSMM_BLAS)
# define LIBXSMM_BLAS LIBXSMM_XGEMM_SYMBOL
/*# define LIBXSMM_BLAS LIBXSMM_YGEMM_SYMBOL*/
#endif


LIBXSMM_GEMM_SYMBOL_DECL(LIBXSMM_GEMM_CONST, ITYPE);


int main(void)
{
#if !defined(__BLAS) || (0 != __BLAS)
  libxsmm_blasint m[]               = { 0, 0, 1, 1, 1, 2, 3, 3, 1,   64,  64,    16,    16, 350, 350, 350, 350, 350,  5, 10, 12, 20,   32,    9 };
  libxsmm_blasint n[]               = { 0, 1, 1, 1, 2, 2, 3, 1, 3,    8, 239, 13824, 65792,  16,   1,  25,   4,   9, 13,  1, 10,  6,   33,    9 };
  libxsmm_blasint k[]               = { 0, 1, 1, 1, 2, 2, 3, 2, 2,   64,  64,    16,    16,  20,   1,  35,   4,  10, 70,  1, 12,  6,  192, 1742 };
  libxsmm_blasint lda[]             = { 0, 1, 1, 1, 1, 2, 3, 3, 1,   64,  64,    16,    16, 350, 350, 350, 350, 350,  5, 22, 22, 22,   32,    9 };
  libxsmm_blasint ldb[]             = { 0, 1, 1, 1, 2, 2, 3, 2, 2, 9216, 240,    16,    16,  35,  35,  35,  35,  35, 70,  1, 20,  8, 2048, 1742 };
  libxsmm_blasint ldc[]             = { 0, 1, 0, 1, 1, 2, 3, 3, 1, 4096, 240,    16,    16, 350, 350, 350, 350, 350,  5, 22, 12, 20, 2048,    9 };
  LIBXSMM_GEMM_CONST OTYPE alpha[]  = { 1, 1, 1, 1, 1, 1, 1, 1, 1,    1,   1,     1,     1,   1,   1,   1,   1,   1,  1,  1,  1,  1,    1,    1 };
  LIBXSMM_GEMM_CONST OTYPE beta[]   = { 0, 1, 0, 1, 1, 1, 1, 0, 0,    0,   1,     0,     0,   0,   0,   0,   0,   0,  0,  0,  0,  0,    0,    0 };
  LIBXSMM_GEMM_CONST char transa = 'N', transb = 'N';
  const int begin = 3, end = sizeof(m) / sizeof(*m);
  libxsmm_blasint max_size_a = 0, max_size_b = 0, max_size_c = 0;
  libxsmm_matdiff_info diff;
  ITYPE *a = 0, *b = 0;
  OTYPE *c = 0, *d = 0;
  int result = EXIT_SUCCESS, test;
# if defined(CHECK_FPE) && defined(__SSE__)
  const unsigned int fpemask = _MM_GET_EXCEPTION_MASK(); /* backup FPE mask */
  const unsigned int fpcheck = _MM_MASK_INVALID | _MM_MASK_OVERFLOW;
  unsigned int fpstate = 0;
  _MM_SET_EXCEPTION_MASK(fpemask | fpcheck);
  _MM_SET_EXCEPTION_STATE(0);
# endif
  for (test = begin; test < end; ++test) {
    const libxsmm_blasint size_a = lda[test] * k[test], size_b = ldb[test] * n[test], size_c = ldc[test] * n[test];
    assert(m[test] <= lda[test] && k[test] <= ldb[test] && m[test] <= ldc[test]);
    max_size_a = LIBXSMM_MAX(max_size_a, size_a);
    max_size_b = LIBXSMM_MAX(max_size_b, size_b);
    max_size_c = LIBXSMM_MAX(max_size_c, size_c);
  }

  a = (ITYPE*)libxsmm_malloc((size_t)(max_size_a * sizeof(ITYPE)));
  b = (ITYPE*)libxsmm_malloc((size_t)(max_size_b * sizeof(ITYPE)));
  c = (OTYPE*)libxsmm_malloc((size_t)(max_size_c * sizeof(OTYPE)));
  d = (OTYPE*)libxsmm_malloc((size_t)(max_size_c * sizeof(OTYPE)));
  assert(0 != a && 0 != b && 0 != c && 0 != d);

  LIBXSMM_MATINIT(ITYPE, 42, a, max_size_a, 1, max_size_a, 1.0);
  LIBXSMM_MATINIT(ITYPE, 24, b, max_size_b, 1, max_size_b, 1.0);
  LIBXSMM_MATINIT(OTYPE,  0, c, max_size_c, 1, max_size_c, 1.0);
  LIBXSMM_MATINIT(OTYPE,  0, d, max_size_c, 1, max_size_c, 1.0);
  memset(&diff, 0, sizeof(diff));

  for (test = begin; test < end && EXIT_SUCCESS == result; ++test) {
    libxsmm_matdiff_info diff_test;

    LIBXSMM_BLAS(ITYPE)(&transa, &transb, m + test, n + test, k + test,
      alpha + test, a, lda + test, b, ldb + test, beta + test, c, ldc + test);

# if defined(CHECK_FPE) && defined(__SSE__)
    fpstate = _MM_GET_EXCEPTION_STATE() & ~fpcheck;
    result = (0 == fpstate ? EXIT_SUCCESS : EXIT_FAILURE);
    if (EXIT_SUCCESS != result) {
#   if defined(_DEBUG)
      fprintf(stderr, "FPE(%i): state=%u\n", test + 1, fpstate);
#   endif
    }
    else
# endif
    {
      REFERENCE_BLAS(ITYPE)(&transa, &transb, m + test, n + test, k + test,
        alpha + test, a, lda + test, b, ldb + test, beta + test, d, ldc + test);

      result = libxsmm_matdiff(LIBXSMM_DATATYPE(OTYPE), m[test], n[test], d, c, ldc + test, ldc + test, &diff_test);
      if (EXIT_SUCCESS == result) {
        if (1.0 >= (1000.0 * diff_test.normf_rel)) {
          libxsmm_matdiff_reduce(&diff, &diff_test);
        }
        else {
# if defined(_DEBUG)
          fprintf(stderr, "Diff(%i): L2abs=%f Linf=%f\n", test + 1, diff_test.l2_abs, diff_test.linf_abs);
# endif
          result = EXIT_FAILURE;
        }
      }
    }
  }

# if defined(CHECK_FPE) && defined(__SSE__)
  _MM_SET_EXCEPTION_MASK(fpemask); /* restore FPE mask */
  _MM_SET_EXCEPTION_STATE(0); /* clear FPE state */
# endif
  libxsmm_free(a);
  libxsmm_free(b);
  libxsmm_free(c);
  libxsmm_free(d);

  return result;
#else
# if defined(_DEBUG)
  fprintf(stderr, "Warning: skipped the test due to missing BLAS support!\n");
# endif
  return EXIT_SUCCESS;
#endif
}


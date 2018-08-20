/******************************************************************************
** Copyright (c) 2016-2018, Intel Corporation                                **
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
/* Alexander Heinecke, Hans Pabst, Dhiraj Kalamkar,
   Rajkishore Barik (Intel Corp.)
******************************************************************************/
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

# define USE_OVERWRITE
/*# define USE_BWD_NO_FILTER_TRANSPOSE_OVERWRITE*/
/*#define USE_FUSED_RELU_BWD*/

#include "bench_defines.h"
//#define USE_FUSED_BATCH_NORM_FWD
//#define USE_ELEMENTWISE_FWD
//#define USE_FUSED_BATCH_NORM_RELU_FWD
//#define USE_FUSE_LEVEL_FOURPASS
//#define USE_FUSE_LEVEL_NAIVE
//#define USE_FUSE_LEVEL_IFM
//#define USE_FUSE_LEVEL_KERNEL
//#define USE_FUSE_LEVEL_JIT

//#define USE_FUSED_BATCH_NORM_BWD
//#define USE_SPLIT_BWD
//#define USE_ELEMENTWISE_BWD
//#define USE_FUSED_BATCH_STATS_BWD
//#define USE_FUSED_BATCH_STATS_RELU_BWD

#define CHKERR_LIBXSMM_DNN(A) { const int chkerr_libxsmm_dnn_ = A; if (LIBXSMM_DNN_SUCCESS != chkerr_libxsmm_dnn_) { \
  fprintf(stderr, "%s\n", libxsmm_dnn_get_error(chkerr_libxsmm_dnn_)); global_status = chkerr_libxsmm_dnn_; } \
}

typedef struct {
  int nImg;
  int nIfm;
  int nOfm;
  int ifhp;
  int ifwp;
  int ifh;
  int ifw;
  int ofhp;
  int ofwp;
  int ofh;
  int ofw;
  int pad_h;
  int pad_w;
  int pad_h_in;
  int pad_w_in;
  int pad_h_out;
  int pad_w_out;
  int kh;
  int kw;
  int stride_h;
  int stride_w;
} naive_conv_t;

LIBXSMM_INLINE void zero_buf(float* buf, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    buf[i] = 0.0f;
  }
}

LIBXSMM_INLINE void copy_buf(float* src, float* dst, size_t size) {
  int i;
#if defined(_OPENMP)
# pragma omp parallel for private(i)
#endif
  for (i = 0; i < (int)size; ++i) {
    dst[i] = src[i];
  }
}

LIBXSMM_INLINE void init_buf(float* buf, size_t size, int initPos, int initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < (int)size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? libxsmm_rand_f64() : (0.05 - libxsmm_rand_f64()/10.0)));
	buf[i] = buf[i] * buf[i];
  }
}

LIBXSMM_INLINE void set_zeropad_nchw(float* nchw, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, float, input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          if (h < pad_h || h >= H-pad_h || w < pad_w || w >= W-pad_w)
            LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W) = 0.0;
        }
      }
    }
  }
}

LIBXSMM_INLINE void copy_internal_nchw(float* dst , float* src, int N, int C, int H, int W, int pad_h, int pad_w)
{
  LIBXSMM_VLA_DECL(4, float, input, src, C, H, W);
  LIBXSMM_VLA_DECL(4, float, new_input, dst, C, H+2*pad_h, W+2*pad_w);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( c = 0; c < C; c++ ) {
      for ( h = 0; h < H; h++ ) {
        for ( w = 0; w < W; w++ ) {
          LIBXSMM_VLA_ACCESS(4, new_input, n, c, h+pad_h, w+pad_w, C, H+2*pad_h, W+2*pad_w) =  LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_copy_NCHW_to_NHWC(const float* nchw, float* nhwc, int N, int H, int W, int C)
{
  LIBXSMM_VLA_DECL(4,       float, output, nhwc, H, W, C);
  LIBXSMM_VLA_DECL(4, const float,  input, nchw, C, H, W);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( h = 0; h < H; h++ ) {
      for ( w = 0; w < W; w++ ) {
        for ( c = 0; c < C; c++ ) {
          LIBXSMM_VLA_ACCESS(4, output, n, h, w, c, H, W, C) =
          LIBXSMM_VLA_ACCESS(4,  input, n, c, h, w, C, H, W);
        }
      }
    }
  }
}


LIBXSMM_INLINE void naive_copy_NHWC_to_NCHW(const float* nhwc, float* nchw, int N, int H, int W, int C)
{
  LIBXSMM_VLA_DECL(4,       float, output, nchw, C, H, W);
  LIBXSMM_VLA_DECL(4, const float,  input, nhwc, H, W, C);
  int n, h, w, c;

  for ( n = 0; n < N; n++ ) {
    for ( h = 0; h < H; h++ ) {
      for ( w = 0; w < W; w++ ) {
        for ( c = 0; c < C; c++ ) {
          LIBXSMM_VLA_ACCESS(4, output, n, c, h, w, C, H, W) =
          LIBXSMM_VLA_ACCESS(4,  input, n, h, w, c, H, W, C);
        }
      }
    }
  }
}


LIBXSMM_INLINE void naive_copy_KCRS_to_RSCK(const float* kcrs, float* rsck, int R, int S, int C, int K)
{
  LIBXSMM_VLA_DECL(4,       float, output, rsck, S, C, K);
  LIBXSMM_VLA_DECL(4, const float,  input, kcrs, C, R, S);
  int r, s, c, k;

  for ( r = 0; r < R; r++ ) {
    for ( s = 0; s < S; s++ ) {
      for ( c = 0; c < C; c++ ) {
        for ( k = 0; k < K; k++ ) {
          LIBXSMM_VLA_ACCESS(4, output, r, s, c, k, S, C, K) =
          LIBXSMM_VLA_ACCESS(4,  input, k, c, r, s, C, R, S);
        }
      }
    }
  }
}


LIBXSMM_INLINE void naive_copy_RSCK_to_KCRS(const float* rsck, float* kcrs, int R, int S, int C, int K)
{
  LIBXSMM_VLA_DECL(4, const float,  input, rsck, S, C, K);
  LIBXSMM_VLA_DECL(4,       float, output, kcrs, C, R, S);
  int r, s, c, k;

  for ( r = 0; r < R; r++ ) {
    for ( s = 0; s < S; s++ ) {
      for ( c = 0; c < C; c++ ) {
        for ( k = 0; k < K; k++ ) {
          LIBXSMM_VLA_ACCESS(4, output, k, c, r, s, C, R, S) =
            LIBXSMM_VLA_ACCESS(4,  input, r, s, c, k, S, C, K);
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_conv_fp(naive_conv_t* param, float* input, float * input_left, float* output, const float* filter, const float* bias, const float* expect, const float* stddev, const float* gamma, const float* beta, float * input_st)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4,       float, output_t, output + (pad_h_out * ofwp + pad_w_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4,       float,  input_t,  input + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4,       float,  input_left_t,  input_left + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4,       float,  input_st_t,  input_st + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float, filter_t, filter, nIfm, kh, kw);
  for (img = 0; img < nImg; ++img) {
    for (ifm = 0; ifm < nIfm; ++ifm) {
      for (ij = 0; ij < ifh; ++ij) {
        for (ii = 0; ii < ifw; ++ii) {
#if defined(USE_FUSED_BATCH_NORM_FWD) || defined(USE_FUSED_BATCH_NORM_RELU_FWD)
	  LIBXSMM_VLA_ACCESS(4,  input_st_t, img, ifm, ij, ii, nIfm, ifhp, ifwp) = LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij, ii, nIfm, ifhp, ifwp);
#endif
#if defined(USE_FUSED_BATCH_NORM_FWD) || defined(USE_FUSED_BATCH_NORM_RELU_FWD)
	  LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij, ii, nIfm, ifhp, ifwp) = (LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij, ii, nIfm, ifhp, ifwp) - expect[ifm]) * stddev[ifm] * gamma[ifm] + beta[ifm];
#endif

#if defined(USE_ELEMENTWISE_FWD) 
          LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij, ii, nIfm, ifhp, ifwp) = (LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij, ii, nIfm, ifhp, ifwp) + LIBXSMM_VLA_ACCESS(4,  input_left_t, img, ifm, ij, ii, nIfm, ifhp, ifwp));
#endif

#if defined(USE_FUSED_BATCH_NORM_RELU_FWD)
          LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij, ii, nIfm, ifhp, ifwp) = (LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij, ii, nIfm, ifhp, ifwp) > 0.f) ? LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij, ii, nIfm, ifhp, ifwp) : 0.f;
#endif
	}
      }
    }
  }

#if defined(USE_FUSED_BIAS) || defined(USE_FUSED_BIAS_RELU)
#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (oj = 0; oj < ofh; ++oj) {
        for (oi = 0; oi < ofw; ++oi) {
          LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) = bias[ofm];
        }
      }
    }
  }
#else
  LIBXSMM_UNUSED(bias);
#endif

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (ifm = 0; ifm < nIfm; ++ifm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) +=
                  LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }
#if defined(USE_FUSED_RELU) || defined(USE_FUSED_BIAS_RELU)
      for (oj = 0; oj < ofh; ++oj) {
        for (oi = 0; oi < ofw; ++oi) {
          LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) =
           (LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) < 0.0f) ? 0.0f : LIBXSMM_VLA_ACCESS(  4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp);
        }
      }
#endif
    }
  }
}

LIBXSMM_INLINE void naive_conv_bp(naive_conv_t* param, float* input, float* input_left, float * split_left, float* output, const float* filter, const float* naive_input_save, const float * bmean1, const float * brstd1,
                                  const float * bmean2, const float * brstd2, const float * dgamma, const float * dbeta,
                                  float * lcl_gamma_beta, const float * input_r, const float * input_r2, const float * gamma)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;
  float nhw = nImg * ofh * ofw;
  float recp_nhw = 1.0f/nhw; 

  LIBXSMM_VLA_DECL(3, float, lcl_gamma_beta_t, lcl_gamma_beta, nImg, nIfm );
  LIBXSMM_VLA_DECL(4, const float, input_r_t, input_r+ (pad_h_out * ofwp + pad_w_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, float, output_t, output + (pad_h_out * ofwp + pad_w_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4,       float,  input_t,  input + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4,       float,  input_left_t,  input_left + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4,       float,  split_left_t,  split_left + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float, input_r2_t, input_r2+ (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4,       float,  naive_input_t,  naive_input_save + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4, const float, filter_t, filter, nIfm, kh, kw);

#ifdef USE_FUSED_BATCH_NORM_BWD
  for (img = 0; img < nImg; ++img) {
    for (ofm = 0; ofm < nOfm; ++ofm) {
      for (oj = 0; oj < ofh; ++oj) {
        for (oi = 0; oi < ofw; ++oi) {
	  LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) = gamma[ofm] * brstd1[ofm] * recp_nhw * (nhw*LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) - 
	                                                                        (dbeta[ofm] + (LIBXSMM_VLA_ACCESS(4, input_r_t, img, ofm, oj, oi, nOfm, ofhp, ofwp) - bmean1[ofm]) * dgamma[ofm] * brstd1[ofm])); 
										
        }
      }
    }
  }
#endif

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (img = 0; img < nImg; ++img) {
    for (ifm = 0; ifm < nIfm; ++ifm) {
      for (ofm = 0; ofm < nOfm; ++ofm) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp) +=
                  LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp)
                * LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw);
              }
            }
          }
        }
      }

      for (ij = 0; ij < ifh; ++ij) {
        for (ii = 0; ii < ifw; ++ii) {
#if defined(USE_SPLIT_BWD)
          LIBXSMM_VLA_ACCESS(4, input_t, img, ifm, ij, ii , nIfm, ifhp, ifwp) += LIBXSMM_VLA_ACCESS(4, split_left_t, img, ifm, ij, ii , nIfm, ifhp, ifwp);
#endif
#if defined(USE_FUSED_BATCH_STATS_RELU_BWD) || defined(USE_FUSED_RELU_BWD)
          if ( LIBXSMM_VLA_ACCESS(4,  naive_input_t, img, ifm, ij, ii , nIfm, ifhp, ifwp) == 0.0 ) {
            LIBXSMM_VLA_ACCESS(4, input_t, img, ifm, ij, ii , nIfm, ifhp, ifwp) = 0.0;
	  }
#endif
#if defined(USE_ELEMENTWISE_BWD)
          LIBXSMM_VLA_ACCESS(4, input_left_t, img, ifm, ij, ii , nIfm, ifhp, ifwp) = LIBXSMM_VLA_ACCESS(4, input_t, img, ifm, ij, ii , nIfm, ifhp, ifwp);
#endif

#if defined(USE_FUSED_BATCH_STATS_RELU_BWD) || defined(USE_FUSED_BATCH_STATS_BWD)
          LIBXSMM_VLA_ACCESS(3, lcl_gamma_beta_t, 0, img, ifm, nImg, nIfm) += (LIBXSMM_VLA_ACCESS(4,  input_r2_t, img, ifm, ij, ii, nIfm, ifhp, ifwp) - bmean2[ifm]) *  LIBXSMM_VLA_ACCESS(4, input_t, img, ifm, ij, ii , nIfm, ifhp, ifwp)  *brstd2[ifm];
          LIBXSMM_VLA_ACCESS(3, lcl_gamma_beta_t, 1, img, ifm, nImg, nIfm) += LIBXSMM_VLA_ACCESS(4, input_t, img, ifm, ij, ii , nIfm, ifhp, ifwp);
#endif
        }
      }
    }
  }
}

LIBXSMM_INLINE void naive_conv_wu(naive_conv_t* param, const float* input, const float* output, float* filter)
{
  int nImg      = param->nImg;
  int nIfm      = param->nIfm;
  int nOfm      = param->nOfm;
  int ifhp      = param->ifhp;
  int ifwp      = param->ifwp;
  int ofhp      = param->ofhp;
  int ofwp      = param->ofwp;
  int ifh       = param->ifh;
  int ifw       = param->ifw;
  int ofh       = param->ofh;
  int ofw       = param->ofw;
  int pad_h     = param->pad_h;
  int pad_w     = param->pad_w;
  int pad_h_in  = param->pad_h_in;
  int pad_w_in  = param->pad_w_in;
  int pad_h_out = param->pad_h_out;
  int pad_w_out = param->pad_w_out;
  int kh        = param->kh;
  int kw        = param->kw;
  int stride_h  = param->stride_h;
  int stride_w  = param->stride_w;
  /* loop counters */
  int img, ofm, ifm, oj, oi, ij, ii, kj, ki;

  LIBXSMM_VLA_DECL(4, const float, output_t, output + (pad_h_out * ofwp + pad_w_out), nOfm, ofhp, ofwp);
  LIBXSMM_VLA_DECL(4, const float,  input_t,  input + (pad_h_in * ifwp + pad_w_in), nIfm, ifhp, ifwp);
  LIBXSMM_VLA_DECL(4,       float, filter_t, filter, nIfm, kh, kw);

#if defined(_OPENMP)
# pragma omp parallel for LIBXSMM_OPENMP_COLLAPSE(2) private(img, ofm, ifm, oj, oi, ij, ii, kj, ki)
#endif
  for (ofm = 0; ofm < nOfm; ++ofm) {
    for (ifm = 0; ifm < nIfm; ++ifm) {
      for (img = 0; img < nImg; ++img) {
        for (oj = 0; oj < ofh; ++oj) {
          ij = oj * stride_h - pad_h;
          for (oi = 0; oi < ofw; ++oi) {
            ii = oi * stride_w - pad_w;
            for (kj = 0; kj < kh; ++kj) {
              if (ij+kj < 0 || ij+kj >= ifh) continue;
              for (ki = 0; ki < kw; ++ki) {
                if (ii+ki < 0 || ii+ki >= ifw) continue;
                LIBXSMM_VLA_ACCESS(4, filter_t, ofm, ifm, kj, ki, nIfm, kh, kw) +=
                  LIBXSMM_VLA_ACCESS(4,  input_t, img, ifm, ij + kj, ii + ki, nIfm, ifhp, ifwp)
                * LIBXSMM_VLA_ACCESS(4, output_t, img, ofm, oj, oi, nOfm, ofhp, ofwp);
              }
            }
          }
        }
      }
    }
  }
}

int main(int argc, char* argv[])
{
  float *naive_input, *naive_output, *naive_output_save, *naive_filter, *naive_filter_wu, *naive_output_bp, *naive_output_wu, *naive_libxsmm_output;
  float *naive_output_bp_save;
  float *naive_libxsmm_input, *naive_libxsmm_filter, *naive_input_save, *naive_filter_save, *naive_filter_kcrs;
  float *input_nhwc, *output_nhwc, *filter_rsck, *dinput_nhwc, *dinput_left_nhwc, *doutput_nhwc, *dfilter_rsck, *naive_output_nhwc, *naive_input_nhwc;
  float *naive_bias, *bias_libxsmm, *naive_dbias, *dbias_libxsmm, *bias_nhwc, *dbias_nhwc;
  float *naive_expect, *naive_stddev, *naive_gamma, *naive_beta;
  float *naive_dgamma, *naive_dbeta, *naive_lcl_gamma_beta, *naive_input_st_bwd, *naive_input_st_bwd2, *naive_bmean1, *naive_brstd1, *naive_gamma_bwd, *naive_bmean2, *naive_brstd2;
  float *expect_libxsmm, *stddev_libxsmm, *gamma_libxsmm, *beta_libxsmm;
  float *bmean1_libxsmm, *brstd1_libxsmm, *input_st_bwd_libxsmm, *input_st_bwd2_libxsmm, *lcl_gamma_beta_libxsmm, *gamma_bwd_libxsmm, *bmean2_libxsmm, *brstd2_libxsmm;
  float *input_libxsmm, *filter_libxsmm, *output_libxsmm, *dinput_libxsmm, *dinput_left_libxsmm, *dfilter_libxsmm, *doutput_libxsmm, *filtertr_libxsmm, *input_save_libxsmm; 
  float *dbeta_libxsmm, *dgamma_libxsmm;
  float *naive_input_st, *naive_libxsmm_input_st, *input_st_libxsmm;
  float *naive_input_left, *naive_libxsmm_input_left, *input_left_libxsmm;
  float *naive_split_left, *naive_libxsmm_split_left, *split_left_libxsmm;
  float *naive_libxsmm_lcl_gamma_beta;
  float *batchstats_libxsmm;
  float *ifm_batchstats_libxsmm;

  int ifhp, ifwp, ofhp, ofwp, ofh, ofw;
  int stride_h, stride_w, pad_h, pad_w, pad_h_in, pad_w_in, pad_h_out, pad_w_out;
  naive_conv_t naive_param;
  void* scratch;
  size_t scratch_size = 0;

  /* some parameters we can overwrite via cli,
     default is some inner layer of overfeat */
  int iters = 10;         /* repetitions of benchmark */
  int ifw = 14;           /* input width, "W" */
  int ifh = 20;           /* input height, "H" */
  int nImg = 32;          /* mini-batch size, "N" */
  int nIfm = 256;         /* number of input feature maps, "C" */
  int nOfm = 512;         /* number of output feature maps, "K" */
  int kh = 3;             /* filter height, "R" */
  int kw = 3;             /* filter width, "S" */
  int padh = 0;           /* padding in input, height */
  int padw = 0;           /* padding in input, width */
  int stride = 1;         /* stride when accessing inputs */
  int padding_mode = 0;   /* padding mode */
  char type = 'A';        /* 'A': ALL, 'F': FP, 'B': BP, 'U', WU */
  char format = 'A';      /* 'A': ALL, 'L': LIBXSMM, 'T': Tensorflow, 'M', Mixed */

  const char *const env_check = getenv("CHECK"), *const env_winograd = getenv("WINOGRAD");
  const double check = LIBXSMM_ABS(0 == env_check ? 0 : atof(env_check));
  const int algo_winograd = (0 == env_winograd ? 0 : atoi(env_winograd));

#if defined(_OPENMP)
  int nThreads = omp_get_max_threads(); /* number of threads */
#else
  int nThreads = 1; /* number of threads */
#endif

  unsigned long long l_start, l_end;
  double l_total = 0.0;
  double flops = 0.0;
  int i;

  libxsmm_dnn_conv_desc conv_desc;
  conv_desc.fuse_ops = LIBXSMM_DNN_CONV_FUSE_NONE;

#if defined(USE_FUSE_LEVEL_FOURPASS)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_BN_FUSE_LEVEL_FOURPASS;
#endif
#if defined(USE_FUSE_LEVEL_NAIVE)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_BN_FUSE_LEVEL_NAIVE;
#endif
#if defined(USE_FUSE_LEVEL_IFM)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_BN_FUSE_LEVEL_IFM;
#endif
#if defined(USE_FUSE_LEVEL_KERNEL)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_BN_FUSE_LEVEL_KERNEL;
#endif
#if defined(USE_FUSE_LEVEL_JIT)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_BN_FUSE_LEVEL_JIT;
#endif

#if defined(USE_FUSED_BATCH_STATS)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_BATCH_STATS;
#endif
#if defined(USE_FUSED_BATCH_NORM_FWD)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_FWD;
#endif
#if defined(USE_ELEMENTWISE_FWD)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_ELEMENTWISE_FWD;
#endif
#if defined(USE_ELEMENTWISE_BWD)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_ELEMENTWISE_BWD;
#endif
#if defined(USE_SPLIT_BWD)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_SPLIT_BWD;
#endif
#if defined(USE_FUSED_BATCH_NORM_RELU_FWD)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU_FWD;
#endif
#if defined(USE_FUSED_BATCH_NORM_BWD)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_BWD;
#endif
#if defined(USE_FUSED_BATCH_STATS_BWD)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_BWD;
#endif
#if defined(USE_FUSED_BATCH_STATS_RELU_BWD)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_RELU_BWD;
#endif
#if defined(USE_FUSED_RELU_BWD)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_RELU_BWD;
#endif
#if defined(USE_FUSED_BIAS)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_BIAS;
#endif
#if defined(USE_FUSED_RELU_FWD)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_RELU_FWD;
#endif
#if defined(USE_FUSED_MAX_STATS)
    conv_desc.fuse_ops |= LIBXSMM_DNN_CONV_FUSE_MAX_STATS;
#endif


  libxsmm_dnn_layer* libxsmm_handle;
  libxsmm_dnn_tensor* libxsmm_input;
  libxsmm_dnn_tensor* libxsmm_output;
  libxsmm_dnn_tensor* libxsmm_filter;
  libxsmm_dnn_tensor* libxsmm_dinput;
  libxsmm_dnn_tensor* libxsmm_dinput_left;
  libxsmm_dnn_tensor* libxsmm_doutput;
  libxsmm_dnn_tensor* libxsmm_dfilter;
  libxsmm_dnn_tensor* libxsmm_filter_tr;
  libxsmm_dnn_tensor* libxsmm_bias;
  libxsmm_dnn_tensor* libxsmm_dbias;
  libxsmm_dnn_tensor* libxsmm_input_st;
  libxsmm_dnn_tensor* libxsmm_input_left;
  libxsmm_dnn_tensor* libxsmm_split_left;
  libxsmm_dnn_tensor* libxsmm_input_save;
  libxsmm_dnn_tensor* libxsmm_input_st_bwd;
  libxsmm_dnn_tensor* libxsmm_input_st_bwd2;
  libxsmm_dnn_tensor* libxsmm_gamma_bwd;
  libxsmm_dnn_tensor* libxsmm_dgamma;
  libxsmm_dnn_tensor* libxsmm_dbeta;
  libxsmm_dnn_tensor* libxsmm_lcl_gamma_beta;
  libxsmm_dnn_tensor* libxsmm_bmean1;
  libxsmm_dnn_tensor* libxsmm_brstd1;
  libxsmm_dnn_tensor* libxsmm_bmean2;
  libxsmm_dnn_tensor* libxsmm_brstd2;
  libxsmm_dnn_tensor* libxsmm_expect;
  libxsmm_dnn_tensor* libxsmm_stddev;
  libxsmm_dnn_tensor* libxsmm_gamma;
  libxsmm_dnn_tensor* libxsmm_beta;

#ifdef USE_FUSED_BATCH_STATS
  libxsmm_dnn_tensor* libxsmm_batchstats;
#endif
#ifdef USE_FUSE_LEVEL_FOURPASS
  libxsmm_dnn_tensor* libxsmm_ifm_batchstats;
#endif
  libxsmm_dnn_tensor_datalayout* libxsmm_layout;
  libxsmm_dnn_err_t status;
  libxsmm_dnn_err_t global_status = LIBXSMM_DNN_SUCCESS;

  libxsmm_matdiff_info norms_fwd, norms_bwd, norms_upd, diff, norms_batchstats, norms_ifm_batchstats;
  memset(&norms_fwd, 0, sizeof(norms_fwd));
  memset(&norms_bwd, 0, sizeof(norms_bwd));
  memset(&norms_upd, 0, sizeof(norms_upd));
  memset(&norms_batchstats, 0, sizeof(norms_batchstats));
  memset(&norms_ifm_batchstats, 0, sizeof(norms_ifm_batchstats));
  memset(&diff, 0, sizeof(diff));

  if (argc > 1 && !strncmp(argv[1], "-h", 3)) {
    printf("Usage: %s iters inpWidth inpHeight nImg nIfm nOfm kw kh pad stride type format padding_mode\n", argv[0]);
    return 0;
  }
  libxsmm_srand(1);

  /* reading new values from cli */
  i = 1;
  if (argc > i) iters      = atoi(argv[i++]);
  if (argc > i) ifw        = atoi(argv[i++]);
  if (argc > i) ifh        = atoi(argv[i++]);
  if (argc > i) nImg       = atoi(argv[i++]);
  if (argc > i) nIfm       = atoi(argv[i++]);
  if (argc > i) nOfm       = atoi(argv[i++]);
  if (argc > i) kw         = atoi(argv[i++]);
  if (argc > i) kh         = atoi(argv[i++]);
  if (argc > i) padw       = atoi(argv[i++]);
  if (argc > i) padh       = atoi(argv[i++]);
  if (argc > i) stride     = atoi(argv[i++]);
  if (argc > i) type       = *(argv[i++]);
  if (argc > i) format     = *(argv[i++]);
  if (argc > i) padding_mode = atoi(argv[i++]);

  if (type != 'A' && type != 'F' && type != 'B' && type != 'U') {
    printf("type needs to be 'A' (All), 'F' (FP only), 'B' (BP only), 'U' (WU only)\n");
    return 0;
  }

  stride_w = stride;
  stride_h = stride;
  pad_w = padw;
  pad_h = padh;

  if (0 == padding_mode) {
    pad_h_in = 0;
    pad_w_in = 0;
    pad_h_out = 0;
    pad_w_out = 0;
  }
  else {
    /* TODO: change "1" to "0" if "padding_mode = -1" is acknowledged */
    if (1 < padding_mode) pad_w = padding_mode;
    pad_h_in = pad_h;
    pad_w_in = pad_w;
    pad_h_out = pad_h;
    pad_w_out = pad_w;
  }

  printf("Pad h in: %d\n", pad_h_in);
  printf("Pad w in: %d\n", pad_w_in);

  /* deriving some values for naive code */
  ofh = (ifh + 2 * pad_h - kh) / stride_h + 1;
  ofw = (ifw + 2 * pad_w - kw) / stride_w + 1;
  ifhp = ifh + 2 * pad_h_in;
  ifwp = ifw + 2 * pad_w_in;
  ofhp = ofh + 2 * pad_h_out;
  ofwp = ofw + 2 * pad_w_out;

  /* set struct for naive convolution */
  naive_param.nImg = nImg;
  naive_param.nIfm = nIfm;
  naive_param.nOfm = nOfm;
  naive_param.ifhp = ifhp;
  naive_param.ifwp = ifwp;
  naive_param.ofhp = ofhp;
  naive_param.ofwp = ofwp;
  naive_param.ifh = ifh;
  naive_param.ifw = ifw;
  naive_param.ofh = ofh;
  naive_param.ofw = ofw;
  naive_param.pad_h = pad_h;
  naive_param.pad_w = pad_w;
  naive_param.pad_h_in = pad_h_in;
  naive_param.pad_w_in = pad_w_in;
  naive_param.pad_h_out = pad_h_out;
  naive_param.pad_w_out = pad_w_out;
  naive_param.kh = kh;
  naive_param.kw = kw;
  naive_param.stride_h = stride_h;
  naive_param.stride_w = stride_w;

#if defined(__SSE3__)
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  _MM_SET_ROUNDING_MODE(_MM_ROUND_NEAREST);
#endif

  /* print some summary */
  printf("##########################################\n");
  printf("#          Setting Up (Common)           #\n");
  printf("##########################################\n");
  printf("PARAMS: W:%d  H:%d  N:%d  C:%d  K:%d  R:%d  S:%d  P:%d  Q:%d  STRIDE:%d\n", ifw, ifh, nImg, nIfm, nOfm, kw, kh, ofh, ofw, stride);
  printf("PARAMS: ITERS:%d", iters); if (LIBXSMM_FEQ(0, check)) printf("  Threads:%d\n", nThreads); else printf("\n");
  printf(" InImg %dx%d Padded (%dx%d)\n", ifh, ifw, ifhp, ifwp);
  printf("OutImg %dx%d Padded (%dx%d)\n", ofh, ofw, ofhp, ofwp);
  printf("SIZE Input  (MB): %10.2f MiB\n", (double)(nImg*nIfm*ifhp*ifwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output (MB): %10.2f MiB\n", (double)(nImg*nOfm*ofhp*ofwp*sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Input   (1): %10.2f MiB\n", (double)(1*nIfm*ifhp*ifwp*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Output  (1): %10.2f MiB\n", (double)(1*nOfm*ofhp*ofwp*   sizeof(float))/(1024.0*1024.0) );
  printf("SIZE Weight     : %10.2f MiB\n", (double)(nIfm*nOfm*kw*kh*    sizeof(float))/(1024.0*1024.0) );
#if defined(USE_OVERWRITE)
  printf("Using Overwrite Option\n");
#endif




  /* allocate data */
  naive_input           = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_input_save      = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_output          = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_save     = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_bp       = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_bp_save  = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_wu       = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxsmm_output  = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_libxsmm_input   = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_filter          = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_filter_save     = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_filter_wu       = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_filter_kcrs     = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  naive_libxsmm_filter  = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  input_nhwc            = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  doutput_nhwc          = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  dinput_nhwc           = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  dinput_left_nhwc           = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  output_nhwc           = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_output_nhwc     = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  naive_input_nhwc      = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  filter_rsck           = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  dfilter_rsck          = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  input_libxsmm         = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  input_save_libxsmm    = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  filter_libxsmm        = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  output_libxsmm        = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  dinput_libxsmm        = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  dinput_left_libxsmm        = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  dfilter_libxsmm       = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  doutput_libxsmm       = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  filtertr_libxsmm      = (float*)libxsmm_aligned_malloc( nOfm*nIfm*kh*kw*    sizeof(float), 2097152);
  batchstats_libxsmm    = (float*)libxsmm_aligned_malloc( 2*nImg*nOfm*        sizeof(float), 2097152);
  ifm_batchstats_libxsmm    = (float*)libxsmm_aligned_malloc( 2*nImg*nIfm*        sizeof(float), 2097152);
  naive_bias            = (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  naive_dbias           = (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  bias_libxsmm          = (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  dbias_libxsmm         = (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  bias_nhwc             = (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  dbias_nhwc            = (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  naive_input_st        = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_input_left      = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_split_left      = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_input_st_bwd	= (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp* sizeof(float), 2097152);
  naive_input_st_bwd2	= (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp* sizeof(float), 2097152);
  naive_dgamma		= (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  naive_dbeta		= (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  naive_lcl_gamma_beta		= (float*)libxsmm_aligned_malloc( 2*nImg*nIfm*               sizeof(float), 2097152);
  naive_gamma_bwd		= (float*)libxsmm_aligned_malloc( nOfm* sizeof(float), 2097152);
  naive_expect 		= (float*)libxsmm_aligned_malloc( nIfm*               sizeof(float), 2097152);
  naive_stddev 		= (float*)libxsmm_aligned_malloc( nIfm*               sizeof(float), 2097152);
  naive_gamma		= (float*)libxsmm_aligned_malloc( nIfm*               sizeof(float), 2097152);
  naive_beta		= (float*)libxsmm_aligned_malloc( nIfm*               sizeof(float), 2097152);
  naive_bmean1 		= (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  naive_brstd1 		= (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  naive_bmean2 		= (float*)libxsmm_aligned_malloc( nIfm*               sizeof(float), 2097152);
  naive_brstd2 		= (float*)libxsmm_aligned_malloc( nIfm*               sizeof(float), 2097152);

  input_st_libxsmm      = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  input_left_libxsmm    = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  split_left_libxsmm    = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  dbeta_libxsmm		= (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  dgamma_libxsmm	= (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  lcl_gamma_beta_libxsmm	= (float*)libxsmm_aligned_malloc( 2*nImg*nIfm*               sizeof(float), 2097152);
  input_st_bwd_libxsmm	= (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*     sizeof(float), 2097152);
  input_st_bwd2_libxsmm	= (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*     sizeof(float), 2097152);
  gamma_bwd_libxsmm	= (float*)libxsmm_aligned_malloc( nOfm*     sizeof(float), 2097152);
  expect_libxsmm 	= (float*)libxsmm_aligned_malloc( nIfm*               sizeof(float), 2097152);
  stddev_libxsmm 	= (float*)libxsmm_aligned_malloc( nIfm*               sizeof(float), 2097152);
  gamma_libxsmm		= (float*)libxsmm_aligned_malloc( nIfm*               sizeof(float), 2097152);
  beta_libxsmm		= (float*)libxsmm_aligned_malloc( nIfm*               sizeof(float), 2097152);
  bmean1_libxsmm 	= (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  brstd1_libxsmm 	= (float*)libxsmm_aligned_malloc( nOfm*               sizeof(float), 2097152);
  bmean2_libxsmm 	= (float*)libxsmm_aligned_malloc( nIfm*               sizeof(float), 2097152);
  brstd2_libxsmm 	= (float*)libxsmm_aligned_malloc( nIfm*               sizeof(float), 2097152);

  naive_libxsmm_lcl_gamma_beta		= (float*)libxsmm_aligned_malloc( 2*nImg*nIfm*               sizeof(float), 2097152);
  naive_libxsmm_input_st= (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_libxsmm_input_left= (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  naive_libxsmm_split_left= (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);

  /* initialize data */
  float *naive_input_tmp           = (float*)libxsmm_aligned_malloc( nImg*nIfm*ifhp*ifwp*sizeof(float), 2097152);
  if (padding_mode == 0 ) {
    init_buf(naive_input,          nImg*nIfm*ifhp*ifwp, 0, 0);
  } else {
    init_buf(naive_input_tmp,          nImg*nIfm*ifh*ifw, 0, 0);
    copy_internal_nchw( naive_input , naive_input_tmp, nImg, nIfm, ifh, ifw, pad_h, pad_w);
  }
#if defined(USE_FUSED_RELU_BWD)
  /* Initialize some entries with zeros  */
  {
    int i;
    for (i = 0; i < nImg*nIfm*ifhp*ifwp; i++ ) {
      if ( ((i%16) == 2) || ((i%16) == 3) || ((i%16) == 7) || ((i%16) == 14) ) {
        naive_input[i] = 0.0;
      }
    }
  }
#endif

  float *naive_output_bp_tmp       = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  float *naive_output_wu_tmp       = (float*)libxsmm_aligned_malloc( nImg*nOfm*ofhp*ofwp*sizeof(float), 2097152);
  if (padding_mode == 0 ) {
    init_buf(naive_output_bp,      nImg*nOfm*ofhp*ofwp, 0, 0);
    copy_buf(naive_output_bp, naive_output_bp_save, nImg*nOfm*ofhp*ofwp);
    init_buf(naive_output_wu,      nImg*nOfm*ofhp*ofwp, 0, 0);
  } else {
    init_buf(naive_output_bp_tmp,      nImg*nOfm*ofh*ofw, 0, 0);
    copy_internal_nchw( naive_output_bp , naive_output_bp_tmp, nImg, nOfm, ofh, ofw, pad_h, pad_w);
    copy_internal_nchw( naive_output_bp_save , naive_output_bp_tmp, nImg, nOfm, ofh, ofw, pad_h, pad_w);
    init_buf(naive_output_wu_tmp,      nImg*nOfm*ofh*ofw, 0, 0);
    copy_internal_nchw( naive_output_wu , naive_output_wu_tmp, nImg, nOfm, ofh, ofw, pad_h, pad_w); 
  }
  set_zeropad_nchw(naive_input, nImg, nIfm, ifhp, ifwp, pad_h_in, pad_w_in);
  set_zeropad_nchw(naive_output_bp, nImg, nOfm, ofhp, ofwp, pad_h_out, pad_w_out);
  set_zeropad_nchw(naive_output_bp_save, nImg, nOfm, ofhp, ofwp, pad_h_out, pad_w_out);
  set_zeropad_nchw(naive_output_wu, nImg, nOfm, ofhp, ofwp, pad_h_out, pad_w_out);

  copy_buf(naive_input, naive_input_save, nImg*nIfm*ifhp*ifwp);
  zero_buf(naive_output_save,    nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_input_st,     nImg*nIfm*ifhp*ifwp);
  init_buf(naive_input_left,     nImg*nIfm*ifhp*ifwp, 0, 0);
  init_buf(naive_split_left,     nImg*nIfm*ifhp*ifwp, 0, 0);
  zero_buf(naive_lcl_gamma_beta,     2*nImg*nIfm);

  init_buf(naive_output,       nImg*nOfm*ofhp*ofwp, 0, 0);
  set_zeropad_nchw(naive_output, nImg, nOfm, ofhp, ofwp, pad_h_out, pad_w_out);
  copy_buf(naive_output, naive_output_save, nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_libxsmm_output, nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_libxsmm_input,  nImg*nIfm*ifhp*ifwp);
  zero_buf(naive_libxsmm_input_st,  nImg*nIfm*ifhp*ifwp);
  zero_buf(naive_libxsmm_input_left,  nImg*nIfm*ifhp*ifwp);
  zero_buf(naive_libxsmm_split_left,  nImg*nIfm*ifhp*ifwp);
  zero_buf(naive_libxsmm_lcl_gamma_beta,  2*nImg*nIfm);
  init_buf(naive_filter,         nOfm*nIfm*kh*kw, 0, 0);
  copy_buf(naive_filter, naive_filter_wu, nOfm*nIfm*kh*kw);
  zero_buf(naive_libxsmm_filter, nOfm*nIfm*kh*kw);
  naive_copy_NCHW_to_NHWC(naive_input, input_nhwc, nImg, ifhp, ifwp, nIfm);
  zero_buf(output_nhwc,          nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_output_nhwc,    nImg*nOfm*ofhp*ofwp);
  zero_buf(naive_input_nhwc,     nImg*nIfm*ifhp*ifwp);
  naive_copy_KCRS_to_RSCK(naive_filter, filter_rsck, kh, kw, nIfm, nOfm);
  init_buf(naive_bias,           nOfm, 0, 0);
  init_buf(naive_dbias,          nOfm, 0, 0);
  copy_buf(naive_bias, bias_nhwc, nOfm);
  copy_buf(naive_dbias, dbias_nhwc, nOfm);

  zero_buf( input_st_libxsmm , nImg*nIfm*ifhp*ifwp );
  zero_buf( input_left_libxsmm , nImg*nIfm*ifhp*ifwp );
  zero_buf( split_left_libxsmm , nImg*nIfm*ifhp*ifwp );
  zero_buf(naive_libxsmm_lcl_gamma_beta, 	2*nImg*nIfm);
  init_buf(naive_expect, 	nIfm, 0, 0);
  init_buf(naive_stddev, 	nIfm, 0, 0);
  init_buf(naive_gamma, 	nIfm, 0, 0);
  init_buf(naive_beta, 		nIfm, 0, 0);
  init_buf(naive_bmean1, 	nOfm, 0, 0);
  init_buf(naive_brstd1, 	nOfm, 0, 0);
  init_buf(naive_bmean2, 	nIfm, 0, 0);
  init_buf(naive_brstd2, 	nIfm, 0, 0);
  init_buf(naive_input_st_bwd, 	nImg*nOfm*ofhp*ofwp, 0, 0);
  init_buf(naive_input_st_bwd2, nImg*nIfm*ifhp*ifwp, 0, 0);
  init_buf(naive_gamma_bwd, 	nOfm, 0, 0);
  copy_buf(naive_expect, expect_libxsmm, nIfm);
  copy_buf(naive_stddev, stddev_libxsmm, nIfm);
  copy_buf(naive_gamma, gamma_libxsmm, nIfm);
  copy_buf(naive_beta, beta_libxsmm, nIfm);
  copy_buf(naive_bmean1, bmean1_libxsmm, nOfm);
  copy_buf(naive_brstd1, brstd1_libxsmm, nOfm);
  copy_buf(naive_bmean2, bmean2_libxsmm, nIfm);
  copy_buf(naive_brstd2, brstd2_libxsmm, nIfm);
  copy_buf(naive_input_st_bwd, input_st_bwd_libxsmm,	nImg*nOfm*ofhp*ofwp);
  copy_buf(naive_input_st_bwd2, input_st_bwd2_libxsmm,	nImg*nIfm*ifhp*ifwp);
  copy_buf(naive_gamma_bwd, gamma_bwd_libxsmm,	nOfm);

  /* first touch LIBXSMM */
  zero_buf( input_libxsmm    , nImg*nIfm*ifhp*ifwp );
  zero_buf( input_save_libxsmm    , nImg*nIfm*ifhp*ifwp );
  zero_buf( filter_libxsmm   , nOfm*nIfm*kh*kw );
  zero_buf( output_libxsmm   , nImg*nOfm*ofhp*ofwp );
  zero_buf( dinput_libxsmm   , nImg*nIfm*ifhp*ifwp );
  zero_buf( dinput_left_libxsmm   , nImg*nIfm*ifhp*ifwp );
  zero_buf( dfilter_libxsmm  , nOfm*nIfm*kh*kw );
  zero_buf( doutput_libxsmm  , nImg*nOfm*ofhp*ofwp );
  zero_buf( filtertr_libxsmm , nOfm*nIfm*kh*kw );

  if (LIBXSMM_NEQ(0, check)) {
    printf("##########################################\n");
    printf("#         Computing Reference ...        #\n");
    printf("##########################################\n");
    if (type == 'A' || type == 'F') {
#ifdef USE_OVERWRITE
      zero_buf(naive_output,    nImg*nOfm*ofhp*ofwp);
#endif
    naive_conv_fp(&naive_param, naive_input, naive_input_left, naive_output, naive_filter, naive_bias, naive_expect, naive_stddev, naive_gamma, naive_beta, naive_input_st);
    }
    if ( (type == 'A' || type == 'B') && (nIfm > 3) ) {
#ifdef USE_OVERWRITE
      zero_buf(naive_input,         nImg*nIfm*ifhp*ifwp);
#endif
      naive_conv_bp(&naive_param, naive_input, naive_input_left, naive_split_left, naive_output_bp, naive_filter, naive_input_save, naive_bmean1, naive_brstd1, naive_bmean2, naive_brstd2, naive_dgamma, naive_dbeta, naive_lcl_gamma_beta, naive_input_st_bwd, naive_input_st_bwd2, naive_gamma_bwd);
    }
    if (type == 'A' || type == 'U') {
      /* NB: We reuse naive_input_save for weight update because the input should not
       * have been modified between forward propagation and weight update; it further
       * helps in exploiting reuse to converted data. */
#ifdef USE_OVERWRITE
      zero_buf(naive_filter_wu,          nOfm*nIfm*kh*kw);
#endif
      naive_conv_wu(&naive_param, naive_input_save, naive_output_wu, naive_filter_wu);
    }
    printf("##########################################\n");
    printf("#      Computing Reference ... done      #\n");
    printf("##########################################\n");
  }

  if (format == 'A' || format == 'L') {
    printf("\n");
    printf("##########################################\n");
    printf("#      Setting Up  (custom-Storage)      #\n");
    printf("##########################################\n");

    /* setup LIBXSMM handle */
    conv_desc.N = nImg;
    conv_desc.C = nIfm;
    conv_desc.H = ifh;
    conv_desc.W = ifw;
    conv_desc.K = nOfm;
    conv_desc.R = kh;
    conv_desc.S = kw;
    conv_desc.u = stride_h;
    conv_desc.v = stride_w;
    conv_desc.pad_h = pad_h;
    conv_desc.pad_w = pad_w;
    conv_desc.pad_h_in = pad_h_in;
    conv_desc.pad_w_in = pad_w_in;
    conv_desc.pad_h_out = pad_h_out;
    conv_desc.pad_w_out = pad_w_out;
    conv_desc.threads = nThreads;
    conv_desc.algo = (0 == algo_winograd ? LIBXSMM_DNN_CONV_ALGO_DIRECT : LIBXSMM_DNN_CONV_ALGO_AUTO);
    conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
    conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
#if defined(USE_BWD_NO_FILTER_TRANSPOSE_OVERWRITE)
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_BWD_NO_FILTER_TRANSPOSE_OVERWRITE;
#elif defined(USE_OVERWRITE)
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
#else
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
#endif
    /*conv_desc.options = LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE;*/
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

    libxsmm_handle = libxsmm_dnn_create_conv_layer( conv_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers and filter */
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input  = libxsmm_dnn_link_tensor( libxsmm_layout,  input_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dinput = libxsmm_dnn_link_tensor( libxsmm_layout, dinput_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_output  = libxsmm_dnn_link_tensor( libxsmm_layout,  output_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_doutput = libxsmm_dnn_link_tensor( libxsmm_layout, doutput_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_filter  = libxsmm_dnn_link_tensor( libxsmm_layout,  filter_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dfilter = libxsmm_dnn_link_tensor( libxsmm_layout, dfilter_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bias  = libxsmm_dnn_link_tensor( libxsmm_layout,  bias_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dbias = libxsmm_dnn_link_tensor( libxsmm_layout, dbias_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_save  = libxsmm_dnn_link_tensor( libxsmm_layout,  input_save_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ST, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_st  = libxsmm_dnn_link_tensor( libxsmm_layout, input_st_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_LEFT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_left  = libxsmm_dnn_link_tensor( libxsmm_layout, input_left_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_SPLIT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_split_left  = libxsmm_dnn_link_tensor( libxsmm_layout, split_left_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_LEFT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dinput_left  = libxsmm_dnn_link_tensor( libxsmm_layout, dinput_left_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ST_BWD, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_st_bwd  = libxsmm_dnn_link_tensor( libxsmm_layout, input_st_bwd_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ST_BWD2, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input_st_bwd2  = libxsmm_dnn_link_tensor( libxsmm_layout, input_st_bwd2_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_LCL_GAMMA_BETA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_lcl_gamma_beta  = libxsmm_dnn_link_tensor( libxsmm_layout,  lcl_gamma_beta_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_EXPECT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_expect  = libxsmm_dnn_link_tensor( libxsmm_layout,  expect_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_STDDEV, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_stddev  = libxsmm_dnn_link_tensor( libxsmm_layout,  stddev_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_GAMMA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_gamma  = libxsmm_dnn_link_tensor( libxsmm_layout,  gamma_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_BETA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_beta  = libxsmm_dnn_link_tensor( libxsmm_layout,  beta_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    // BWD specific
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_BMEAN1, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bmean1  = libxsmm_dnn_link_tensor( libxsmm_layout,  bmean1_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_BRSTD1, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_brstd1  = libxsmm_dnn_link_tensor( libxsmm_layout,  brstd1_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_BMEAN2, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bmean2  = libxsmm_dnn_link_tensor( libxsmm_layout,  bmean2_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_BRSTD2, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_brstd2  = libxsmm_dnn_link_tensor( libxsmm_layout,  brstd2_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_GAMMA_BWD, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_gamma_bwd  = libxsmm_dnn_link_tensor( libxsmm_layout, gamma_bwd_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_BETA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dbeta  = libxsmm_dnn_link_tensor( libxsmm_layout,  dbeta_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_GAMMA, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dgamma  = libxsmm_dnn_link_tensor( libxsmm_layout,  dgamma_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );



    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER_TRANS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_filter_tr  = libxsmm_dnn_link_tensor( libxsmm_layout, filtertr_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

#ifdef USE_FUSED_BATCH_STATS
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_BATCH_STATS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_batchstats  = libxsmm_dnn_link_tensor( libxsmm_layout, batchstats_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
#endif

#ifdef USE_FUSE_LEVEL_FOURPASS
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_IFM_BATCH_STATS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_ifm_batchstats  = libxsmm_dnn_link_tensor( libxsmm_layout, ifm_batchstats_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
#endif

    /* copy in data to LIBXSMM format */
    /* we can also use the layout functions and set the data on our
       own external to the library, @TODO, we plan to add an example here */

#if defined(USE_FUSE_LEVEL_NAIVE) || defined(USE_FUSE_LEVEL_FOURPASS)
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input_st,  (void*)naive_input_save,  LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
#else
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input,  (void*)naive_input_save,  LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
#endif
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input_save,  (void*)naive_input_save,  LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input_left,  (void*)naive_input_left,  LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_split_left,  (void*)naive_split_left,  LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_output, (void*)naive_output_save, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_filter, (void*)naive_filter,      LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_bias,   (void*)naive_bias,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_expect,   (void*)naive_expect,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_stddev,   (void*)naive_stddev,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_gamma,   (void*)naive_gamma,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_beta,   (void*)naive_beta,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input_st_bwd,  (void*)naive_input_st_bwd,  LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input_st_bwd2,  (void*)naive_input_st_bwd2,  LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );

    // BWD specific
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_gamma_bwd,   (void*)naive_gamma_bwd,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_bmean1,   (void*)naive_bmean1,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_brstd1,   (void*)naive_brstd1,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_bmean2,   (void*)naive_bmean2,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_brstd2,   (void*)naive_brstd2,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_dgamma,   (void*)naive_dgamma,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_dbeta,   (void*)naive_dbeta,        LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );

    zero_buf(filtertr_libxsmm, nOfm*nIfm*kh*kw);
    zero_buf(batchstats_libxsmm, 2*nImg*nOfm);
    zero_buf(ifm_batchstats_libxsmm, 2*nImg*nIfm);

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input,      LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input_save,  LIBXSMM_DNN_REGULAR_INPUT_SAVE ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input_st,   LIBXSMM_DNN_REGULAR_INPUT_ST ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input_left,   LIBXSMM_DNN_REGULAR_INPUT_LEFT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_split_left,   LIBXSMM_DNN_REGULAR_INPUT_SPLIT) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input_st_bwd,   LIBXSMM_DNN_REGULAR_INPUT_ST_BWD ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input_st_bwd2,   LIBXSMM_DNN_REGULAR_INPUT_ST_BWD2 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_expect,       LIBXSMM_DNN_REGULAR_EXPECT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_stddev,       LIBXSMM_DNN_REGULAR_STDDEV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_gamma,       LIBXSMM_DNN_REGULAR_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_beta,       LIBXSMM_DNN_REGULAR_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_lcl_gamma_beta,   LIBXSMM_DNN_REGULAR_LCL_GAMMA_BETA ) );

    // BWD specific
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_gamma_bwd,       LIBXSMM_DNN_REGULAR_GAMMA_BWD ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_bmean1,       LIBXSMM_DNN_REGULAR_BMEAN1 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_brstd1,       LIBXSMM_DNN_REGULAR_BRSTD1 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_bmean2,       LIBXSMM_DNN_REGULAR_BMEAN2 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_brstd2,       LIBXSMM_DNN_REGULAR_BRSTD2 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dgamma,   LIBXSMM_DNN_GRADIENT_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dbeta,   LIBXSMM_DNN_GRADIENT_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dinput,     LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dinput_left,     LIBXSMM_DNN_GRADIENT_INPUT_LEFT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_output,     LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_doutput,    LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_filter,     LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dfilter,    LIBXSMM_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_bias,       LIBXSMM_DNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dbias,      LIBXSMM_DNN_GRADIENT_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_filter_tr,  LIBXSMM_DNN_REGULAR_FILTER_TRANS ) );
#ifdef USE_FUSED_BATCH_STATS
   CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_batchstats, LIBXSMM_DNN_BATCH_STATS ) );
#endif
#ifdef USE_FUSE_LEVEL_FOURPASS
   CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_ifm_batchstats, LIBXSMM_DNN_IFM_BATCH_STATS ) );
#endif

    /* let's allocate and bind scratch */
    scratch_size = libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
    CHKERR_LIBXSMM_DNN( status );
    scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
    /* set scratch to bogus to make sure that libxsmm takes care of zeroing internally */
    init_buf( (float*)scratch, scratch_size/4, 0, 0 );

    if ((type == 'A' || type == 'F') && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("#   Correctness - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
#if defined(USE_FUSE_LEVEL_NAIVE) || defined(USE_FUSE_LEVEL_FOURPASS)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input_st,  (void*)naive_input_save,  LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
#else
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input,  (void*)naive_input_save,  LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
#endif
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input_left,  (void*)naive_input_left,  LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_split_left,  (void*)naive_split_left,  LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }
      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_output, (void*)naive_libxsmm_output, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_input_st, (void*)naive_libxsmm_input_st, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_input, (void*)naive_libxsmm_input, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );

      /* compare */
      printf("Output\n");
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nOfm*ofhp*ofwp, 1, naive_output, naive_libxsmm_output, 0, 0, &norms_fwd);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);

#if defined(USE_FUSED_BATCH_NORM_FWD) || defined(USE_FUSED_BATCH_NORM_RELU_FWD)
      printf("Input streaming store\n");
      //libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input_st, naive_libxsmm_input_st, 0, 0, &norms_fwd);
      printf("Input store:\n");
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      //libxsmm_matdiff_reduce(&diff, &norms_fwd);

      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input, naive_libxsmm_input, 0, 0, &norms_fwd);
      printf("Input overwritten by BN\n");
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
#endif

#if defined(USE_FUSED_BATCH_STATS)
      {
        float *ch_sum, *ch_sum_fuse;
        float *ch_sum2, *ch_sum2_fuse;
        int img_i = 0;
        int ch_i = 0;
        int ch_j = 0;
        int pxl_i = 0;
        LIBXSMM_VLA_DECL(4, float, sum_fuse,  batchstats_libxsmm, nOfm/16, nImg, 16);
        LIBXSMM_VLA_DECL(3, float, sum_naive, naive_output,       nOfm, ofhp*ofwp);

        ch_sum       = (float*) malloc(nOfm*sizeof(float));
        ch_sum_fuse  = (float*) malloc(nOfm*sizeof(float));
        ch_sum2      = (float*) malloc(nOfm*sizeof(float));
        ch_sum2_fuse = (float*) malloc(nOfm*sizeof(float));

        for ( ch_i = 0; ch_i < nOfm; ++ch_i ) {
          ch_sum_fuse[ch_i] = 0.0f;
          ch_sum2_fuse[ch_i] = 0.0f;
          ch_sum[ch_i] = 0.0f;
          ch_sum2[ch_i] = 0.0f;
        }
        for ( ch_i = 0; ch_i < nOfm/16; ++ch_i ) {
          for ( img_i = 0; img_i < nImg; ++img_i ) {
            for ( ch_j = 0; ch_j < 16; ++ch_j ) {
              ch_sum_fuse[(ch_i*16) + ch_j]  += LIBXSMM_VLA_ACCESS(4, sum_fuse, 0, ch_i, img_i, ch_j, nOfm/16, nImg, 16);
              ch_sum2_fuse[(ch_i*16) + ch_j] += LIBXSMM_VLA_ACCESS(4, sum_fuse, 1, ch_i, img_i, ch_j, nOfm/16, nImg, 16);
            }
          }
        }
        for ( img_i = 0; img_i < nImg; ++img_i ) {
          for ( ch_i = 0; ch_i < nOfm; ++ch_i ) {
            for ( pxl_i = 0; pxl_i < ofhp*ofwp; ++pxl_i ) {
              const float f = LIBXSMM_VLA_ACCESS(3, sum_naive, img_i, ch_i, pxl_i, nOfm, ofhp*ofwp);
              ch_sum2[ch_i] += f * f;
              ch_sum[ch_i]  += f;
            }
          }
        }

        libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nOfm, 1, ch_sum, ch_sum_fuse, 0, 0, &norms_batchstats);
        printf("Channel Sum:\n");
        printf("L1 reference  : %.25g\n", norms_batchstats.l1_ref);
        printf("L1 test       : %.25g\n", norms_batchstats.l1_tst);
        printf("L2 abs.error  : %.24f\n", norms_batchstats.l2_abs);
        printf("L2 rel.error  : %.24f\n", norms_batchstats.l2_rel);
        printf("Linf abs.error: %.24f\n", norms_batchstats.linf_abs);
        printf("Linf rel.error: %.24f\n", norms_batchstats.linf_rel);
        printf("Check-norm    : %.24f\n", norms_batchstats.normf_rel);

        libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nOfm, 1, ch_sum2, ch_sum2_fuse, 0, 0, &norms_batchstats);
        printf("Channel Sum2:\n");
        printf("L1 reference  : %.25g\n", norms_batchstats.l1_ref);
        printf("L1 test       : %.25g\n", norms_batchstats.l1_tst);
        printf("L2 abs.error  : %.24f\n", norms_batchstats.l2_abs);
        printf("L2 rel.error  : %.24f\n", norms_batchstats.l2_rel);
        printf("Linf abs.error: %.24f\n", norms_batchstats.linf_abs);
        printf("Linf rel.error: %.24f\n", norms_batchstats.linf_rel);
        printf("Check-norm    : %.24f\n", norms_batchstats.normf_rel);

        free(ch_sum);
        free(ch_sum2);
        free(ch_sum_fuse);
        free(ch_sum2_fuse);
      }
#endif

#if defined(USE_FUSE_LEVEL_FOURPASS)
      {
        float *ch_sum, *ch_sum_fuse;
        float *ch_sum2, *ch_sum2_fuse;
        int img_i = 0;
        int ch_i = 0;
        int ch_j = 0;
        int pxl_i = 0;
        int pxl_j = 0;
        LIBXSMM_VLA_DECL(4, float, sum_fuse,  ifm_batchstats_libxsmm, nIfm/16, nImg, 16);
        LIBXSMM_VLA_DECL(3, float, sum_naive, naive_input_save,       nIfm, ifhp*ifwp);

        ch_sum       = (float*) malloc(nIfm*sizeof(float));
        ch_sum_fuse  = (float*) malloc(nIfm*sizeof(float));
        ch_sum2      = (float*) malloc(nIfm*sizeof(float));
        ch_sum2_fuse = (float*) malloc(nIfm*sizeof(float));

        for ( ch_i = 0; ch_i < nIfm; ++ch_i ) {
          ch_sum_fuse[ch_i] = 0.0f;
          ch_sum2_fuse[ch_i] = 0.0f;
          ch_sum[ch_i] = 0.0f;
          ch_sum2[ch_i] = 0.0f;
        }
        for ( ch_i = 0; ch_i < nIfm/16; ++ch_i ) {
          for ( img_i = 0; img_i < nImg; ++img_i ) {
            for ( ch_j = 0; ch_j < 16; ++ch_j ) {
              ch_sum_fuse[(ch_i*16) + ch_j]  += LIBXSMM_VLA_ACCESS(4, sum_fuse, 0, ch_i, img_i, ch_j, nIfm/16, nImg, 16);
              ch_sum2_fuse[(ch_i*16) + ch_j] += LIBXSMM_VLA_ACCESS(4, sum_fuse, 1, ch_i, img_i, ch_j, nIfm/16, nImg, 16);
			}
          }
        }
        for ( img_i = 0; img_i < nImg; ++img_i ) {
          for ( ch_i = 0; ch_i < nIfm; ++ch_i ) {
		    float chan_sum = 0.0;
            for ( pxl_i = 0; pxl_i < ifh ; ++pxl_i ) {
            for ( pxl_j = 0 ; pxl_j < ifw ; ++pxl_j) {
              int pxl_i_pad = pxl_i + pad_h_in;
              int pxl_j_pad = pxl_j + pad_w_in;
              int pxl = pxl_j_pad + pxl_i_pad * ifwp;
              const float f = LIBXSMM_VLA_ACCESS(3, sum_naive, img_i, ch_i, pxl, nIfm, ifhp*ifwp);
              chan_sum  += f;
            }
            }
            ch_sum[ch_i] += chan_sum / (float)(ifh * ifw);
		  }
		}
        for ( img_i = 0; img_i < nImg; ++img_i ) {
          for ( ch_i = 0; ch_i < nIfm; ++ch_i ) {
		    float chan_sum2 = 0.0;
            for ( pxl_i = 0; pxl_i < ifh; ++pxl_i ) {
            for ( pxl_j = 0 ; pxl_j < ifw ; ++pxl_j) {
              int pxl_i_pad = pxl_i + pad_h_in;
              int pxl_j_pad = pxl_j + pad_w_in;
              int pxl = pxl_j_pad + pxl_i_pad * ifwp;
              const float f = LIBXSMM_VLA_ACCESS(3, sum_naive, img_i, ch_i, pxl, nIfm, ifhp*ifwp);
			  chan_sum2 += (f-naive_expect[ch_i]) * (f-naive_expect[ch_i]);
            }
            }
            ch_sum2[ch_i] += chan_sum2 / (float)(ifh * ifw);
          }
        }

        libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nIfm, 1, ch_sum, ch_sum_fuse, 0, 0, &norms_ifm_batchstats);
        printf("Channel Sum:\n");
        printf("L1 reference  : %.25g\n", norms_ifm_batchstats.l1_ref);
        printf("L1 test       : %.25g\n", norms_ifm_batchstats.l1_tst);
        printf("L2 abs.error  : %.24f\n", norms_ifm_batchstats.l2_abs);
        printf("L2 rel.error  : %.24f\n", norms_ifm_batchstats.l2_rel);
        printf("Linf abs.error: %.24f\n", norms_ifm_batchstats.linf_abs);
        printf("Linf rel.error: %.24f\n", norms_ifm_batchstats.linf_rel);
        printf("Check-norm    : %.24f\n", norms_ifm_batchstats.normf_rel);

        libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nIfm, 1, ch_sum2, ch_sum2_fuse, 0, 0, &norms_ifm_batchstats);
        printf("Channel Sum2:\n");
        printf("L1 reference  : %.25g\n", norms_ifm_batchstats.l1_ref);
        printf("L1 test       : %.25g\n", norms_ifm_batchstats.l1_tst);
        printf("L2 abs.error  : %.24f\n", norms_ifm_batchstats.l2_abs);
        printf("L2 rel.error  : %.24f\n", norms_ifm_batchstats.l2_rel);
        printf("Linf abs.error: %.24f\n", norms_ifm_batchstats.linf_abs);
        printf("Linf rel.error: %.24f\n", norms_ifm_batchstats.linf_rel);
        printf("Check-norm    : %.24f\n", norms_ifm_batchstats.normf_rel);

        free(ch_sum);
        free(ch_sum2);
        free(ch_sum_fuse);
        free(ch_sum2_fuse);
      }
#endif



    }

    if ( (type == 'A' || type == 'B') && (nIfm > 3) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("#   Correctness - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* let's do some additional init such that we can run passes standalone */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor(    libxsmm_doutput, (void*)naive_output_bp, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor(    libxsmm_doutput, (void*)naive_output_bp_save, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor(    libxsmm_dinput, (void*)naive_input_save, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor(    libxsmm_input_save, (void*)naive_input_save, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
#if defined(USE_BWD_NO_FILTER_TRANSPOSE_OVERWRITE)
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_trans_reg_filter( libxsmm_handle ) );
#endif

      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }

      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_dinput, (void*)naive_libxsmm_input, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_dinput_left, (void*)naive_libxsmm_input_left, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_doutput, (void*)naive_libxsmm_output, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_lcl_gamma_beta, (void*)naive_libxsmm_lcl_gamma_beta, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );

      /* compare */
      printf("input\n");
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input, naive_libxsmm_input, 0, 0, &norms_bwd);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);

#ifdef USE_ELEMENTWISE_BWD
      printf("input_left\n");
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input_left, naive_libxsmm_input_left, 0, 0, &norms_bwd);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
#endif

#ifdef USE_FUSED_BATCH_NORM_BWD
      printf("output\n");
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nOfm*ofhp*ofwp, 1, naive_output_bp, naive_libxsmm_output, 0, 0, &norms_bwd);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
#endif

#if defined(USE_FUSED_BATCH_STATS_BWD) || defined(USE_FUSED_BATCH_STATS_RELU_BWD)
      printf("lcl_gamma_beta\n");
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, 2*nImg*nIfm, 1, naive_lcl_gamma_beta, naive_libxsmm_lcl_gamma_beta, 0, 0, &norms_bwd);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
#endif

    }

    if ((type == 'A' || type == 'U') && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("#   Correctness - UPD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* let's do some additional init such that we can run passes standalone */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_input, (void*)naive_input_save, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_doutput, (void*)naive_output_wu, LIBXSMM_DNN_TENSOR_FORMAT_NCHW ) );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_dfilter, (void*)naive_filter, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
      }
      if (conv_desc.options == LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE) {
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_reduce_wu_filters( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
      }
      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_dfilter, (void*)naive_libxsmm_filter, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nOfm*nIfm*kh*kw, 1, naive_filter_wu, naive_libxsmm_filter, 0, 0, &norms_upd);
      printf("L1 reference  : %.25g\n", norms_upd.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd);
    }

    if (type == 'A' || type == 'F') {
      printf("##########################################\n");
      printf("#   Performance - FWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();
#if defined(_OPENMP)
#     pragma omp parallel private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

	  double input_size = (double)(nImg*nIfm*ifhp*ifwp*sizeof(float))/(1024.0*1024.0);
	  double output_size = (double)(nImg*nOfm*ofhp*ofwp*sizeof(float))/(1024.0*1024.0);
	  double weight_size = (double)(nIfm*nOfm*kw*kh*    sizeof(float))/(1024.0*1024.0);

      printf("PERFDUMP,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
        norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel, input_size, output_size, weight_size);
    }

    if ( (type == 'A' || type == 'B') && (nIfm > 3) ) {
      printf("##########################################\n");
      printf("#   Performance - BWD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel  private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("bp time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_bwd.l1_ref, norms_bwd.l1_tst,
        norms_bwd.l2_abs, norms_bwd.l2_rel, norms_bwd.linf_abs, norms_bwd.linf_rel, norms_bwd.normf_rel);
    }

    if (type == 'A' || type == 'U') {
      printf("##########################################\n");
      printf("#   Performance - UPD (custom-Storage)   #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();

#if defined(_OPENMP)
#     pragma omp parallel private(i)
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        for (i = 0; i < iters; ++i) {
          libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
          if (conv_desc.options == LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE) {
            CHKERR_LIBXSMM_DNN( libxsmm_dnn_reduce_wu_filters( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
          }
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP  = %.5g\n", flops*1e-9/(double)iters);
      printf("wu time = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS  = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_upd.l1_ref, norms_upd.l1_tst,
        norms_upd.l2_abs, norms_upd.l2_rel, norms_upd.linf_abs, norms_upd.linf_rel, norms_upd.normf_rel);
    }

    /* clean-up */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
    libxsmm_free(scratch);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_LEFT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_EXPECT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_STDDEV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER_TRANS ) );
#ifdef USE_FUSED_BATCH_STATS
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_BATCH_STATS ) );
#endif
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input_save ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput_left ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_doutput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dfilter ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_lcl_gamma_beta ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_expect ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_stddev ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_gamma ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_beta ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bmean1 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_brstd1 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bmean2 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_brstd2 ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter_tr ) );
#ifdef USE_FUSED_BATCH_STATS
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_batchstats ) );
#endif
#ifdef USE_FUSE_LEVEL_FOURPASS
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_ifm_batchstats ) );
#endif
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_conv_layer( libxsmm_handle ) );
  }

  if (format == 'A' || format == 'T') {
    printf("\n");
    printf("##########################################\n");
    printf("#    Setting Up - (NHWC/RSCK-Storage)    #\n");
    printf("##########################################\n");

    /* setup LIBXSMM handle */
    conv_desc.N = nImg;
    conv_desc.C = nIfm;
    conv_desc.H = ifh;
    conv_desc.W = ifw;
    conv_desc.K = nOfm;
    conv_desc.R = kh;
    conv_desc.S = kw;
    conv_desc.u = stride_h;
    conv_desc.v = stride_w;
    conv_desc.pad_h = pad_h;
    conv_desc.pad_w = pad_w;
    conv_desc.pad_h_in = pad_h_in;
    conv_desc.pad_w_in = pad_w_in;
    conv_desc.pad_h_out = pad_h_out;
    conv_desc.pad_w_out = pad_w_out;
    conv_desc.threads = nThreads;
    conv_desc.algo = (0 == algo_winograd ? LIBXSMM_DNN_CONV_ALGO_DIRECT : LIBXSMM_DNN_CONV_ALGO_AUTO);
    conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NHWC;
    conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_RSCK;
#ifdef USE_OVERWRITE
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
#else
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
#endif
    /*conv_desc.options = LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE;*/
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

    libxsmm_handle = libxsmm_dnn_create_conv_layer( conv_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers and filter */
    naive_copy_NCHW_to_NHWC(naive_input_save, input_nhwc, nImg, ifhp, ifwp, nIfm);
    naive_copy_NCHW_to_NHWC(naive_output_save, output_nhwc, nImg, ofhp, ofwp, nOfm);

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input  = libxsmm_dnn_link_tensor( libxsmm_layout,  input_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dinput = libxsmm_dnn_link_tensor( libxsmm_layout, dinput_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dinput_left = libxsmm_dnn_link_tensor( libxsmm_layout, dinput_left_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_output  = libxsmm_dnn_link_tensor( libxsmm_layout,  output_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_doutput = libxsmm_dnn_link_tensor( libxsmm_layout, doutput_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_filter  = libxsmm_dnn_link_tensor( libxsmm_layout,  filter_rsck, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dfilter = libxsmm_dnn_link_tensor( libxsmm_layout, dfilter_rsck, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bias  = libxsmm_dnn_link_tensor( libxsmm_layout,  bias_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dbias = libxsmm_dnn_link_tensor( libxsmm_layout, dbias_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dinput, LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dinput_left, LIBXSMM_DNN_GRADIENT_INPUT_LEFT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_output, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_doutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_filter, LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dfilter, LIBXSMM_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_bias, LIBXSMM_DNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dbias, LIBXSMM_DNN_GRADIENT_BIAS ) );

    /* let's allocate and bind scratch */
    scratch_size = libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
    CHKERR_LIBXSMM_DNN( status );
    scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
    /* set scratch to bogus to make sure that libxsmm takes care of zeroing internally */
    init_buf( (float*)scratch, scratch_size/4, 0, 0 );

    if ((type == 'A' || type == 'F') && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("#  Correctness - FWD (NHWC/RSCK-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }
      /* copy output data into NCHW storage in user code */
      naive_copy_NHWC_to_NCHW(output_nhwc, naive_output_nhwc, nImg, ofhp, ofwp, nOfm);

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nOfm*ofhp*ofwp, 1, naive_output, naive_output_nhwc, 0, 0, &norms_fwd);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
    }

    if ( (type == 'A' || type == 'B') && (nIfm > 3) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("# Correctness - BWD (NHWC/RSCK-Storage)  #\n");
      printf("##########################################\n");
      /* let's do some additional init such that we can run passes standalone */
      naive_copy_NCHW_to_NHWC(naive_output_bp, doutput_nhwc, nImg, ofhp, ofwp, nOfm);
      naive_copy_NCHW_to_NHWC(naive_input_save, dinput_nhwc, nImg, ifhp, ifwp, nIfm);
      naive_copy_NCHW_to_NHWC(naive_input_left, dinput_left_nhwc, nImg, ifhp, ifwp, nIfm);
      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }
      /* copy input data into NCHW storage in user code */
      naive_copy_NHWC_to_NCHW(dinput_nhwc, naive_input_nhwc, nImg, ifhp, ifwp, nIfm);

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input, naive_input_nhwc, 0, 0, &norms_bwd);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
    }

    if ((type == 'A' || type == 'U') && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("# Correctness - UPD (NHWC/RSCK-Storage)  #\n");
      printf("##########################################\n");
      /* let's do some additional init such that we can run passes standalone */
      naive_copy_NCHW_to_NHWC(naive_input_save, input_nhwc, nImg, ifhp, ifwp, nIfm);
      naive_copy_NCHW_to_NHWC(naive_output_wu, doutput_nhwc, nImg, ofhp, ofwp, nOfm);
      naive_copy_KCRS_to_RSCK(naive_filter, dfilter_rsck, kh, kw, nIfm, nOfm);
      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
      }
      if (conv_desc.options == LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE) {
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_reduce_wu_filters( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
      }
      /* copy input data into KCRS storage in user code */
      naive_copy_RSCK_to_KCRS(dfilter_rsck, naive_filter_kcrs, kh, kw, nIfm, nOfm);

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nOfm*nIfm*kh*kw, 1, naive_filter_wu, naive_filter_kcrs, 0, 0, &norms_upd);
      printf("L1 reference  : %.25g\n", norms_upd.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd);
    }

    if (type == 'A' || type == 'F') {
      printf("##########################################\n");
      printf("#  Performance - FWD (NHWC/RSCK-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();
      for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#       pragma omp parallel
#endif
        {
#if defined(_OPENMP)
          const int tid = omp_get_thread_num();
#else
          const int tid = 0;
#endif
          libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP (NHWC,RSCK)  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time (NHWC,RSCK) = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS (NHWC,RSCK) = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP-NHWC-RSCK,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
        norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
    }

    if ( (type == 'A' || type == 'B') && (nIfm > 3) ) {
      printf("##########################################\n");
      printf("#  Performance - BWD (NHWC/RSCK-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();
      for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#       pragma omp parallel
#endif
        {
#if defined(_OPENMP)
          const int tid = omp_get_thread_num();
#else
          const int tid = 0;
#endif
          libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP (NHWC,RSCK)  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time (NHWC,RSCK) = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS (NHWC,RSCK) = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP-NHWC-RSCK,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_bwd.l1_ref, norms_bwd.l1_tst,
        norms_bwd.l2_abs, norms_bwd.l2_rel, norms_bwd.linf_abs, norms_bwd.linf_rel, norms_bwd.normf_rel);
    }

    if (type == 'A' || type == 'U') {
      printf("##########################################\n");
      printf("#  Performance - UPD (NHWC/RSCK-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();
      for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#       pragma omp parallel
#endif
        {
#if defined(_OPENMP)
          const int tid = omp_get_thread_num();
#else
          const int tid = 0;
#endif
          libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
        }
        if (conv_desc.options == LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE) {
          CHKERR_LIBXSMM_DNN( libxsmm_dnn_reduce_wu_filters( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP (NHWC,RSCK)  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time (NHWC,RSCK) = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS (NHWC,RSCK) = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP-NHWC-RSCK,WU,%s,%i,%i,%i,%i,%i,%i,%i, %i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_upd.l1_ref, norms_upd.l1_tst,
        norms_upd.l2_abs, norms_upd.l2_rel, norms_upd.linf_abs, norms_upd.linf_rel, norms_upd.normf_rel);
    }

    /* clean-up */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
    libxsmm_free(scratch);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_LEFT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_EXPECT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_STDDEV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput_left ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_doutput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dfilter ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_conv_layer( libxsmm_handle ) );
  }

  if (format == 'A' || format == 'M') {
    printf("\n");
    printf("##########################################\n");
    printf("#   Setting Up - (NHWC/custom-Storage)   #\n");
    printf("##########################################\n");

    /* setup LIBXSMM handle */
    conv_desc.N = nImg;
    conv_desc.C = nIfm;
    conv_desc.H = ifh;
    conv_desc.W = ifw;
    conv_desc.K = nOfm;
    conv_desc.R = kh;
    conv_desc.S = kw;
    conv_desc.u = stride_h;
    conv_desc.v = stride_w;
    conv_desc.pad_h = pad_h;
    conv_desc.pad_w = pad_w;
    conv_desc.pad_h_in = pad_h_in;
    conv_desc.pad_w_in = pad_w_in;
    conv_desc.pad_h_out = pad_h_out;
    conv_desc.pad_w_out = pad_w_out;
    conv_desc.threads = nThreads;
    conv_desc.algo = (0 == algo_winograd ? LIBXSMM_DNN_CONV_ALGO_DIRECT : LIBXSMM_DNN_CONV_ALGO_AUTO);
    conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_NHWC;
    conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
#ifdef USE_OVERWRITE
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;
#else
    conv_desc.options = LIBXSMM_DNN_CONV_OPTION_NONE;
#endif
    /*conv_desc.options = LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE;*/
    conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
    conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

    libxsmm_handle = libxsmm_dnn_create_conv_layer( conv_desc, &status );
    CHKERR_LIBXSMM_DNN( status );

    /* setup LIBXSMM buffers and filter */
    naive_copy_NCHW_to_NHWC(naive_output_save, output_nhwc, nImg, ofhp, ofwp, nOfm);
    naive_copy_NCHW_to_NHWC(naive_input_save, input_nhwc, nImg, ifhp, ifwp, nIfm);

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_INPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_input  = libxsmm_dnn_link_tensor( libxsmm_layout,  input_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dinput = libxsmm_dnn_link_tensor( libxsmm_layout, dinput_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dinput_left = libxsmm_dnn_link_tensor( libxsmm_layout, dinput_left_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_OUTPUT, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_output  = libxsmm_dnn_link_tensor( libxsmm_layout,  output_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_doutput = libxsmm_dnn_link_tensor( libxsmm_layout, doutput_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_FILTER, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_filter  = libxsmm_dnn_link_tensor( libxsmm_layout,  filter_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dfilter = libxsmm_dnn_link_tensor( libxsmm_layout, dfilter_libxsmm, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_BIAS, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_bias  = libxsmm_dnn_link_tensor( libxsmm_layout,  bias_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dbias = libxsmm_dnn_link_tensor( libxsmm_layout, dbias_nhwc, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* copy in data to LIBXSMM format */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_filter, (void*)naive_filter, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dinput, LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dinput_left, LIBXSMM_DNN_GRADIENT_INPUT_LEFT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_output, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_doutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_filter, LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dfilter, LIBXSMM_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_bias, LIBXSMM_DNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dbias, LIBXSMM_DNN_GRADIENT_BIAS ) );

    /* let's allocate and bind scratch */
    scratch_size = libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
    CHKERR_LIBXSMM_DNN( status );
    scratch = libxsmm_aligned_scratch( scratch_size, 2097152 );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
    /* set scratch to bogus to make sure that libxsmm takes care of zeroing internally */
    init_buf( (float*)scratch, scratch_size/4, 0, 0 );

    if ((type == 'A' || type == 'F') && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("# Correctness - FWD(NHWC/custom-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
      }
      /* copy output data into NCHW storage in user code */
      naive_copy_NHWC_to_NCHW(output_nhwc, naive_output_nhwc, nImg, ofhp, ofwp, nOfm);

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nOfm*ofhp*ofwp, 1, naive_output, naive_output_nhwc, 0, 0, &norms_fwd);
      printf("L1 reference  : %.25g\n", norms_fwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_fwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_fwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_fwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_fwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_fwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_fwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_fwd);
    }

    if ( (type == 'A' || type == 'B') && (nIfm > 3) && LIBXSMM_NEQ(0, check) ) {
      printf("##########################################\n");
      printf("# Correctness - BWD(NHWC/custom-Storage) #\n");
      printf("##########################################\n");
      /* let's do some additional init such that we can run passes standalone */
      naive_copy_NCHW_to_NHWC(naive_output_bp, doutput_nhwc, nImg, ofhp, ofwp, nOfm);
      naive_copy_NCHW_to_NHWC(naive_input_save, dinput_nhwc, nImg, ifhp, ifwp, nIfm);
      naive_copy_NCHW_to_NHWC(naive_input_left, dinput_left_nhwc, nImg, ifhp, ifwp, nIfm);
      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
      }
      /* copy input data into NCHW storage in user code */
      naive_copy_NHWC_to_NCHW(dinput_nhwc, naive_input_nhwc, nImg, ifhp, ifwp, nIfm);

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nImg*nIfm*ifhp*ifwp, 1, naive_input, naive_input_nhwc, 0, 0, &norms_bwd);
      printf("L1 reference  : %.25g\n", norms_bwd.l1_ref);
      printf("L1 test       : %.25g\n", norms_bwd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_bwd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_bwd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_bwd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_bwd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_bwd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_bwd);
    }

    if ((type == 'A' || type == 'U') && LIBXSMM_NEQ(0, check)) {
      printf("##########################################\n");
      printf("# Correctness - UPD(NHWC/custom-Storage) #\n");
      printf("##########################################\n");
      /* let's do some additional init such that we can run passes standalone */
      naive_copy_NCHW_to_NHWC(naive_input_save, input_nhwc, nImg, ifhp, ifwp, nIfm);
      naive_copy_NCHW_to_NHWC(naive_output_wu, doutput_nhwc, nImg, ofhp, ofwp, nOfm);
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_dfilter, (void*)naive_filter, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
      /* run LIBXSMM convolutions */
#if defined(_OPENMP)
#     pragma omp parallel
#endif
      {
#if defined(_OPENMP)
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
      }
      if (conv_desc.options == LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE) {
        CHKERR_LIBXSMM_DNN( libxsmm_dnn_reduce_wu_filters( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
      }
      /* copy out data */
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_dfilter, (void*)naive_libxsmm_filter, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );

      /* compare */
      libxsmm_matdiff(LIBXSMM_DATATYPE_F32, nOfm*nIfm*kh*kw, 1, naive_filter_wu, naive_libxsmm_filter, 0, 0, &norms_upd);
      printf("L1 reference  : %.25g\n", norms_upd.l1_ref);
      printf("L1 test       : %.25g\n", norms_upd.l1_tst);
      printf("L2 abs.error  : %.24f\n", norms_upd.l2_abs);
      printf("L2 rel.error  : %.24f\n", norms_upd.l2_rel);
      printf("Linf abs.error: %.24f\n", norms_upd.linf_abs);
      printf("Linf rel.error: %.24f\n", norms_upd.linf_rel);
      printf("Check-norm    : %.24f\n", norms_upd.normf_rel);
      libxsmm_matdiff_reduce(&diff, &norms_upd);
    }

    if (type == 'A' || type == 'F') {
      printf("##########################################\n");
      printf("# Performance - FWD(NHWC/custom-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();
      for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#       pragma omp parallel
#endif
        {
#if defined(_OPENMP)
          const int tid = omp_get_thread_num();
#else
          const int tid = 0;
#endif
          libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP (NHWC,custom)  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time (NHWC,custom) = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS (NHWC,custom) = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP-NHWC-custom,FP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_fwd.l1_ref, norms_fwd.l1_tst,
        norms_fwd.l2_abs, norms_fwd.l2_rel, norms_fwd.linf_abs, norms_fwd.linf_rel, norms_fwd.normf_rel);
    }

    if ( (type == 'A' || type == 'B') && (nIfm > 3) ) {
      printf("##########################################\n");
      printf("# Performance - BWD(NHWC/custom-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();
      for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#       pragma omp parallel
#endif
        {
#if defined(_OPENMP)
          const int tid = omp_get_thread_num();
#else
          const int tid = 0;
#endif
          libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP (NHWC,custom)  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time (NHWC,custom) = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS (NHWC,custom) = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP-NHWC-custom,BP,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_bwd.l1_ref, norms_bwd.l1_tst,
        norms_bwd.l2_abs, norms_bwd.l2_rel, norms_bwd.linf_abs, norms_bwd.linf_rel, norms_bwd.normf_rel);
    }

    if (type == 'A' || type == 'U') {
      printf("##########################################\n");
      printf("# Performance - UPD(NHWC/custom-Storage) #\n");
      printf("##########################################\n");
      /* run LIBXSMM convolution for performance */
      l_start = libxsmm_timer_tick();
      for (i = 0; i < iters; ++i) {
#if defined(_OPENMP)
#       pragma omp parallel
#endif
        {
#if defined(_OPENMP)
          const int tid = omp_get_thread_num();
#else
          const int tid = 0;
#endif
          libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid );
        }
        if (conv_desc.options == LIBXSMM_DNN_CONV_OPTION_UPD_NO_FILTER_REDUCE) {
          CHKERR_LIBXSMM_DNN( libxsmm_dnn_reduce_wu_filters( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
        }
      }
      l_end = libxsmm_timer_tick();
      l_total = libxsmm_timer_duration(l_start, l_end);
      flops = (double)nImg * (double)nIfm * (double)nOfm * (double)ofh * (double)ofw * (double)(2 * kh * kw) * (double)iters;

      printf("GFLOP (NHWC,custom)  = %.5g\n", flops*1e-9/(double)iters);
      printf("fp time (NHWC,custom) = %.5g\n", ((double)(l_total/iters)));
      printf("GFLOPS (NHWC,custom) = %.5g\n", (flops*1e-9)/l_total);

      printf("PERFDUMP-NHWC-custom,WU,%s,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%i,%.5g,%.5g,%f,%f,%f,%f,%f,%f,%f\n", LIBXSMM_VERSION, nThreads, nImg, nIfm, nOfm,
        ifw, ifh, kw, kh, stride, padw, padh, ((double)(l_total/iters)), (flops*1e-9)/l_total, norms_upd.l1_ref, norms_upd.l1_tst,
        norms_upd.l2_abs, norms_upd.l2_rel, norms_upd.linf_abs, norms_upd.linf_rel, norms_upd.normf_rel);
    }

    /* clean-up */
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL ) );
    libxsmm_free(scratch);
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_LEFT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_EXPECT ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_STDDEV ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_GAMMA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_REGULAR_BETA ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_release_tensor( libxsmm_handle, LIBXSMM_DNN_GRADIENT_BIAS ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_input ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dinput_left ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_output ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_doutput ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_filter ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dfilter ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_bias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_tensor( libxsmm_dbias ) );
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_destroy_conv_layer( libxsmm_handle ) );
  }

  /* deallocate data */
  libxsmm_free(naive_input);
  libxsmm_free(naive_input_save);
  libxsmm_free(naive_output);
  libxsmm_free(naive_output_save);
  libxsmm_free(naive_output_bp);
  libxsmm_free(naive_output_bp_save);
  libxsmm_free(naive_output_wu);
  libxsmm_free(naive_libxsmm_output);
  libxsmm_free(naive_libxsmm_input);
  libxsmm_free(naive_filter);
  libxsmm_free(naive_filter_save);
  libxsmm_free(naive_filter_wu);
  libxsmm_free(naive_filter_kcrs);
  libxsmm_free(naive_libxsmm_filter);
  libxsmm_free(input_nhwc);
  libxsmm_free(output_nhwc);
  libxsmm_free(dinput_nhwc);
  libxsmm_free(dinput_left_nhwc);
  libxsmm_free(doutput_nhwc);
  libxsmm_free(naive_output_nhwc);
  libxsmm_free(naive_input_nhwc);
  libxsmm_free(filter_rsck);
  libxsmm_free(dfilter_rsck);
  libxsmm_free(input_libxsmm);
  libxsmm_free(filter_libxsmm);
  libxsmm_free(output_libxsmm);
  libxsmm_free(dinput_libxsmm);
  libxsmm_free(dinput_left_libxsmm);
  libxsmm_free(dfilter_libxsmm);
  libxsmm_free(doutput_libxsmm);
  libxsmm_free(filtertr_libxsmm);
#ifdef FP64_BN_STATS   
  libxsmm_free(batchstats_libxsmm);
#endif
  libxsmm_free(naive_bias);
  libxsmm_free(naive_dbias);
  libxsmm_free(bias_nhwc);
  libxsmm_free(dbias_nhwc);
  libxsmm_free(bias_libxsmm);
  libxsmm_free(dbias_libxsmm);
  libxsmm_free(naive_lcl_gamma_beta);
  libxsmm_free(naive_expect);
  libxsmm_free(naive_stddev);
  libxsmm_free(naive_bmean1);
  libxsmm_free(naive_brstd1);
  libxsmm_free(naive_bmean2);
  libxsmm_free(naive_brstd2);
  libxsmm_free(naive_gamma);
  libxsmm_free(naive_beta);

  { const char *const env_check_scale = getenv("CHECK_SCALE");
    const double check_scale = LIBXSMM_ABS(0 == env_check_scale ? 1.0 : atof(env_check_scale));
    if (LIBXSMM_NEQ(0, check) && (check < 100.0 * check_scale * diff.normf_rel) && (global_status == LIBXSMM_DNN_SUCCESS)) {
      fprintf(stderr, "FAILED with an error of %f%%!\n", 100.0 * diff.normf_rel);
      exit(EXIT_FAILURE);
    }
  }

  /* some empty lines at the end */
  printf("\n\n\n");

  return EXIT_SUCCESS;
}


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
/* Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/

int imgofm1, img, ofm1, ifm1, oj, ij, oi, ii, kj, ki, ifm2, ofm2;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int work = handle->desc.N * handle->blocksofm;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;

/* offset output pointer in case of physical output padding */
element_output_type *const out = ((element_output_type*)handle->reg_output->data) + (handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * (handle->blocksofm*handle->ofmblock);

/* padding via stack allocated buffers */
const int padded_w = handle->desc.W + (2 * handle->desc.pad_w);
const int padded_h = handle->desc.H + (2 * handle->desc.pad_h);
const int scratch7_size = padded_h * padded_w * handle->ifmblock;
#if 0 /* TODO: no VLAs */
element_input_type *const input_scratch_padding = (element_input_type*)(((char*)handle->scratch7) + ltid * LIBXSMM_UP2(scratch7_size * sizeof(element_input_type), LIBXSMM_CACHELINE));
#else
element_input_type input_scratch_padding[scratch7_size];
#endif
for ( ii = 0; ii < scratch7_size; ++ii ) { input_scratch_padding[ii] = (element_input_type)0; }

{ /* open new scope for additional variable declarations (C89) */
  LIBXSMM_VLA_DECL(5, element_output_type, output, out, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock);
  LIBXSMM_VLA_DECL(5, const element_input_type, input, (element_input_type*)handle->reg_input->data, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
#ifdef LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
  LIBXSMM_VLA_DECL(6, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock);
#endif
#ifdef LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_RSCK
  LIBXSMM_VLA_DECL(6, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock);
#endif
  LIBXSMM_VLA_DECL(3, element_input_type, input_padded, input_scratch_padding, padded_w, handle->ifmblock);

  /* perform convolution */
  for (imgofm1 = thr_begin; imgofm1 < thr_end; ++imgofm1) {
    img = imgofm1 / handle->blocksofm;
    ofm1 = imgofm1 % handle->blocksofm;
    /* handle fused bias addition */
    if ( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) ) {
      LIBXSMM_VLA_DECL(2, element_output_type, bias, (element_output_type*)handle->reg_bias->data, handle->ofmblock);
      element_output_type* temp_ptr_2 = &(LIBXSMM_VLA_ACCESS(  2, bias, ofm1, 0, handle->ofmblock));
      /* copy bias into output feature map */
      for (oj = 0; oj < handle->ofh; ++oj) {
        element_output_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output, img, oj, 0, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
        for (oi = 0; oi < handle->ofw; ++oi) {
          LIBXSMM_PRAGMA_SIMD
          for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
            temp_ptr[ofm2] = temp_ptr_2[ofm2];
          }
          temp_ptr += handle->blocksofm*handle->ofmblock;
        }
      }
    }
    for (ifm1 = 0; ifm1 < handle->blocksifm; ++ifm1) {
      /* reset result buffer to zero when intent is to overwrite when first block
         of input channels should be convoluted */
      if ( (ifm1 == 0) && ((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) == 0) ) {
        /* set output feature map to zero */
        for (oj = 0; oj < handle->ofh; ++oj) {
          element_output_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output, img, oj, 0, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
          for (oi = 0; oi < handle->ofw; ++oi) {
            LIBXSMM_PRAGMA_SIMD
            for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
              temp_ptr[ofm2] = (element_output_type)0;
            }
            temp_ptr += handle->blocksofm*handle->ofmblock;
          }
        }
      }
      /* check if we need padding, for now we do physical padding on the fly, however we can play with N parameter of the GEMM */
      /* @TODO: add variant which deals with multiple GEMMS by varying N to deal with padding */
      if ( (handle->desc.pad_h == handle->desc.pad_h_in) && (handle->desc.pad_w == handle->desc.pad_w_in) ) {
        /* run convolution */
        for (oj = 0; oj < handle->ofh; ++oj) {
          ij = oj * handle->desc.u;
          ii = 0; oi = 0;
          for (kj = 0; kj < handle->desc.R; ++kj) {
            for (ki = 0; ki< handle->desc.S; ++ki) {
#ifdef LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
              gemm_kernel( &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock),
#endif
#ifdef LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_RSCK
              gemm_kernel( &LIBXSMM_VLA_ACCESS(6, weight, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock),
#endif
                           &LIBXSMM_VLA_ACCESS(5,  input,  img, ij + kj, ii + ki, ifm1, 0, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock),
                           &LIBXSMM_VLA_ACCESS(5, output,  img, oj, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock) );
            }
          }
        }
      } else {
        /* copy into stack buffer for physical padding */
        for (ij = 0; ij < handle->desc.H; ++ij) {
          for (ii = 0; ii < handle->desc.W; ++ii) {
            LIBXSMM_PRAGMA_SIMD
            for (ifm2 = 0; ifm2 < handle->ifmblock; ++ifm2) {
              LIBXSMM_VLA_ACCESS(3, input_padded, ij + handle->desc.pad_h, ii + handle->desc.pad_w, ifm2, padded_w, handle->ifmblock) =
                LIBXSMM_VLA_ACCESS(5,  input, img, ij, ii, ifm1, ifm2, handle->ifhp, handle->ifwp, handle->blocksifm, handle->ifmblock);
            }
          }
        }

        /* run convolution */
        for (oj = 0; oj < handle->ofh; ++oj) {
          ij = oj * handle->desc.u;
          ii = 0; oi = 0;
          for (kj = 0; kj < handle->desc.R; ++kj) {
            for (ki = 0; ki< handle->desc.S; ++ki) {
#ifdef LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_CUSTOM
              gemm_kernel( &LIBXSMM_VLA_ACCESS(6, weight, ofm1, ifm1, kj, ki, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, handle->ifmblock, handle->ofmblock),
#endif
#ifdef LIBXSMM_DNN_TPL_FWD_DIRECT_GENERIC_NHWC_RSCK
              gemm_kernel( &LIBXSMM_VLA_ACCESS(6, weight, kj, ki, ifm1, 0, ofm1, 0, handle->desc.S, handle->blocksifm, handle->ifmblock, handle->blocksofm, handle->ofmblock),
#endif
                           &LIBXSMM_VLA_ACCESS(3, input_padded, ij + kj, ii + ki, 0, padded_w, handle->ifmblock),
                           &LIBXSMM_VLA_ACCESS(5, output, img, oj, oi, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock) );
            }
          }
        }
      }
    }
    /* ReLU handling */
    if ( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU) > 0) ) {
      /* Apply relu to output feature map */
      for (oj = 0; oj < handle->ofh; ++oj) {
        element_output_type* temp_ptr   = &(LIBXSMM_VLA_ACCESS(  5, output, img, oj, 0, ofm1, 0, handle->ofhp, handle->ofwp, handle->blocksofm, handle->ofmblock));
        for (oi = 0; oi < handle->ofw; ++oi) {
          LIBXSMM_PRAGMA_SIMD
          for (ofm2 = 0; ofm2 < handle->ofmblock; ++ofm2) {
            temp_ptr[ofm2] = (element_output_type)(temp_ptr[ofm2] < 0 ? 0 : temp_ptr[ofm2]);
          }
          temp_ptr += handle->blocksofm*handle->ofmblock;
        }
      }
    }
  }
}


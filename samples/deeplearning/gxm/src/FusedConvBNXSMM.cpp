/******************************************************************************
** Copyright (c) 2017-2018, Intel Corporation                                **
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
/* Sasikanth Avancha, Dhiraj Kalamkar, Alexander Heinecke (Intel Corp.)
******************************************************************************/

#include "FusedConvBNXSMM.hpp"

using namespace std;

FusedConvBNXSMM::FusedConvBNXSMM(FusedConvBNImplParams* gp, int engine) : FusedConvBNImpl(gp, engine)
{
  conv_desc.N = gp->batch_size;
  conv_desc.C = gp->nInput;
  conv_desc.H = gp->iHeight;
  conv_desc.W = gp->iWidth;
  conv_desc.K = gp->nOutput;
  conv_desc.R = gp->kh;
  conv_desc.S = gp->kw;
  conv_desc.u = gp->stride_h;
  conv_desc.v = gp->stride_w;

  if(gp->physical_padding)
  {
    conv_desc.pad_h_in = gp->ipad_h;
    conv_desc.pad_w_in = gp->ipad_w;
  }
  else
  {
    conv_desc.pad_h_in = 0;
    conv_desc.pad_w_in = 0;
  }

  conv_desc.pad_w = gp->pad_w;
  conv_desc.pad_h = gp->pad_h;

  if(gp->physical_padding)
  {
    conv_desc.pad_h_out = gp->opad_h;
    conv_desc.pad_w_out = gp->opad_w;
  }
  else
  {
    conv_desc.pad_h_out = 0;
    conv_desc.pad_w_out = 0;
  }

  conv_desc.threads = gp->num_threads;
  conv_desc.algo = LIBXSMM_DNN_CONV_ALGO_DIRECT;
  conv_desc.buffer_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  conv_desc.filter_format = LIBXSMM_DNN_TENSOR_FORMAT_LIBXSMM;
  conv_desc.options = LIBXSMM_DNN_CONV_OPTION_OVERWRITE;

  int my_fuse_ops = (int)LIBXSMM_DNN_CONV_FUSE_NONE;
  if(gp->relu_fwd)
    my_fuse_ops |= (int)LIBXSMM_DNN_CONV_FUSE_RELU_FWD;
  if(gp->relu_bwd)
    my_fuse_ops |= (int)LIBXSMM_DNN_CONV_FUSE_RELU_BWD;
  if(gp->bstats_fwd)
    my_fuse_ops |= (int)LIBXSMM_DNN_CONV_FUSE_BATCH_STATS;
  if(gp->bstats_bwd)
    my_fuse_ops |= (int)LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_BWD;
  if(gp->bn_fwd)
    my_fuse_ops |= (int)LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_FWD;
  if(gp->bn_bwd)
    my_fuse_ops |= (int)LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_BWD;
  if(gp->bn_relu_fwd)
    my_fuse_ops |= (int)LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU_FWD;
  if(gp->bstats_relu_bwd)
    my_fuse_ops |= (int)LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_RELU_BWD;
  if(gp->eltwise)
  {
    my_fuse_ops |= (int)LIBXSMM_DNN_CONV_FUSE_ELEMENTWISE_FWD;
    my_fuse_ops |= (int)LIBXSMM_DNN_CONV_FUSE_ELEMENTWISE_BWD;
  }
  if(gp->split)
    my_fuse_ops |= (int)LIBXSMM_DNN_CONV_FUSE_SPLIT_BWD;

  conv_desc.fuse_ops = (libxsmm_dnn_conv_fuse_op)my_fuse_ops;

  assert(gp->in_data_type == DT_FLOAT && gp->out_data_type == DT_FLOAT);
  conv_desc.datatype_in = LIBXSMM_DNN_DATATYPE_F32;
  conv_desc.datatype_out = LIBXSMM_DNN_DATATYPE_F32;

  libxsmm_handle = libxsmm_dnn_create_conv_layer( conv_desc, &status );
  CHKERR_LIBXSMM_DNN( status );

  top_layout_type = LIBXSMM_CUSTOM_LAYOUT;
  top_layout = libxsmm_handle;
  gbot_layout_type = LIBXSMM_CUSTOM_LAYOUT;
  gbot_layout = libxsmm_handle;
}

void FusedConvBNXSMM::reduce_batch_stats(void *bstats_ip, float *bmeanp, float *brstdp, TensorBuf *gmeanpb, TensorBuf *grstdpb, int nFM, int fh, int fw)
{
  int nImg = gp->batch_size;
  int nBfm = nFM/VLEN;
  const float nhw = nImg*fh*fw;
  const float recp_nhw = 1.0f/nhw;
  const float nhw_ratio = nhw/(nhw - 1);

  float *gmeanp = (float*)gmeanpb->getBuffer();
  float *grstdp = (float*)grstdpb->getBuffer();

  __assume_aligned(bmeanp, 64);
  __assume_aligned(brstdp, 64);
  __assume_aligned(gmeanp, 64);
  __assume_aligned(grstdp, 64);

  float (* __restrict bmean)[VLEN] = (float (*)[VLEN])bmeanp;
  float (* __restrict brstd)[VLEN] = (float (*)[VLEN])brstdp;
  float (* __restrict gmean)[VLEN] = (float (*)[VLEN])gmeanp;
  float (* __restrict grstd)[VLEN] = (float (*)[VLEN])grstdp;

  float (* __restrict ibstats)[nImg][VLEN]  = (float (*)[*][VLEN])bstats_ip;
  float (* __restrict ibstats2)[nImg][VLEN] = (float (*)[*][VLEN])((float*)(bstats_ip + nFM*nImg*sizeof(float) ));

#ifdef __AVX512F__
  __m512  vrecp_nhw  = _mm512_set1_ps(recp_nhw);
  __m512  veps       = _mm512_set1_ps(gp->eps);
  __m512  vmmf       = _mm512_set1_ps(gp->mmf);
  __m512  vnhw_ratio = _mm512_set1_ps(nhw_ratio);
  float one          = 1.0;
  __m512  vone       = _mm512_set1_ps(one);

#if 1
#ifdef _OPENMP
#pragma omp parallel for
#endif
#endif
  for (int b = 0; b < nBfm; ++b) {
    __m512 tmp1  = _mm512_setzero_ps();
    __m512 tmpd1 = _mm512_setzero_ps();

    /* reduce over images */
    for (int n = 0; n < nImg; ++n) {
      tmp1 = _mm512_add_ps( tmp1, _mm512_load_ps(&(ibstats[b][n][0]) ) );
      tmpd1 = _mm512_add_ps( tmpd1, _mm512_load_ps(&(ibstats2[b][n][0]) ) );
    }

    __m512 vtbmeanA   = _mm512_mul_ps( vrecp_nhw, tmp1 );
    __m512 vtbmean2A  = _mm512_mul_ps( vtbmeanA, vtbmeanA );
    __m512 vtbmean_2A = _mm512_mul_ps( vrecp_nhw, tmpd1 );
#ifdef __AVX512ER__
    __m512 vtbrstd_A  = _mm512_rsqrt28_ps( _mm512_add_ps( _mm512_sub_ps( vtbmean_2A, vtbmean2A ), veps) );
#else
    __m512 vtbrstd_A  = _mm512_rsqrt14_ps( _mm512_add_ps( _mm512_sub_ps( vtbmean_2A, vtbmean2A ), veps) );
#endif
    _mm512_store_ps( &(bmean[b][0]), vtbmeanA );
    _mm512_store_ps( &(brstd[b][0]), vtbrstd_A );

    if(!gp->use_global_stats)
    {
      _mm512_store_ps( &(gmeanp[b*16]), _mm512_add_ps( _mm512_mul_ps( _mm512_load_ps( &(gmeanp[b*16]) ), vmmf), vtbmeanA));
      _mm512_store_ps( &(grstdp[b*16]), _mm512_add_ps( _mm512_mul_ps( _mm512_load_ps( &(grstdp[b*16]) ), vmmf), _mm512_mul_ps(vnhw_ratio, vtbrstd_A)));
    }
  }
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int b = 0; b < nBfm; ++b) {
    float tmp[16];
    float tmpd[16];
#pragma omp simd
    for (int v = 0; v < 16; ++v) {
      tmp[v] = 0.0;
      tmpd[v] = 0.0;
    }
    /* reduce over images */
    for (int n = 0; n < nImg; ++n) {
#pragma omp simd
      for(int v = 0; v < 16; ++v) {
        tmp[v] += ibstats[b][n][v];
        tmpd[v] += ibstats2[b][n][v];
      }
    }
    /* calculate expectation and standard derivation */
#pragma omp simd
    for (int v = 0; v < 16; ++v) {
      const float tbmean = (recp_nhw*tmp[v]) ;
      const float tbmean2  = tbmean * tbmean;
      const float tbmean_2 = recp_nhw * tmpd[v];
      const float tbrstd = 1.0/sqrt(tbmean_2 - tbmean2 + gp->eps);
      bmean[b][v] = tbmean;
      brstd[b][v] = tbrstd;
      if(!gp->use_global_stats)
      {
        gmeanp[(b*16)+v] = gmeanp[(b*16)+v] * gp->mmf + tbmean;
        grstdp[(b*16)+v] = grstdp[(b*16)+v] * gp->mmf + nhw_ratio*tbrstd;
      }
    }
  }
#endif
}

void FusedConvBNXSMM::forwardPropagate(vector<TensorBuf *>& inp, TensorBuf *weightp, TensorBuf *gammap, TensorBuf *betap, TensorBuf* my_gammap, TensorBuf* my_betap, TensorBuf *gmeanp, TensorBuf *grstdp, vector<TensorBuf *>& outp, int tid)
{
#ifdef TIMING_OV
  struct timeval tvs, tve;

  gettimeofday(&tvs, NULL);
#endif
  assert(bot_compute_engine[0] != -1);
  assert(top_compute_engine[0] != -1);

  // Conv input
  in_ptr = inp[0]->getBuffer();
  in_prv_ptr = inp[0]->getPrivBuffer();

  if(gp->eltwise)
    in_res_ptr = inp[1]->getBuffer();

  // Conv output
  out_ptr = outp[0]->getBuffer();
  out_prv_ptr = outp[0]->getPrivBuffer();

  if(gp->split)
  {
    outp[1]->setBuffer(in_ptr);
    int res_size = conv_desc.N * conv_desc.C * (gp->iHeight + 2*gp->ipad_h) * (gp->iWidth + 2*gp->ipad_w);
    outp[1]->setBufferSize(res_size);
  }

  // Conv Weight
  wt_ptr = weightp->getBuffer();
  void *wt_prv_ptr = NULL;

  //Stats of output appended to output buffer
  int offset = conv_desc.N * conv_desc.K * (gp->oHeight + 2*gp->opad_h) * (gp->oWidth + 2*gp->opad_w);
  void *out_stats_ptr = out_ptr + offset * sizeof(float);
memset(out_stats_ptr, 0, 2*conv_desc.N * conv_desc.K * sizeof(float));

  if(gp->bn_fwd || gp->bn_relu_fwd)
  {
    // Reduced Batch stats from previous layer
    int inp_off = conv_desc.N * conv_desc.C * (gp->iHeight + 2*gp->ipad_h) * (gp->iWidth + 2*gp->ipad_w);
    void *in_stats_ptr = in_ptr + inp_off*sizeof(float);
    expect = (float*)(in_stats_ptr + 2*conv_desc.N*conv_desc.C*sizeof(float));
    rstdev = expect + conv_desc.C;

    // Compute reduced batch stats to use in FWD BN operation along with global mean/rstdev (if required)
    reduce_batch_stats(in_stats_ptr, expect, rstdev, gmeanp, grstdp, conv_desc.C, conv_desc.H, conv_desc.W);

  // Saved input for BWD
    sin_ptr = (void*)(rstdev + conv_desc.C); 

    //Gamma
    gamma_ptr = gammap->getBuffer();

    // Beta
    beta_ptr = betap->getBuffer();
  }

  void *scratch = scratchp->getBuffer();

  if(libxsmm_input == NULL && libxsmm_input_left == NULL && libxsmm_filter == NULL && libxsmm_output == NULL && libxsmm_input_st == NULL && libxsmm_gamma == NULL && libxsmm_beta == NULL && libxsmm_expect == NULL && libxsmm_rstdev == NULL)
  {
    if(bot_compute_engine[0] != engine)
    {
      if(in_prv_ptr == NULL)
      {
        int size = gp->batch_size * gp->nInput * (gp->iHeight + 2*gp->ipad_h) * (gp->iWidth + 2*gp->ipad_w);
        in_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(float), 2097152);

        inp[0]->setPrivBuffer(in_prv_ptr);
      }

      /* setup LIBXSMM buffers and filter */
      in_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input = libxsmm_dnn_link_tensor( in_buffer_layout, in_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
    }
    else
    {
      in_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input = libxsmm_dnn_link_tensor( in_buffer_layout, in_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
    }
    // Bind input buffer to handle
    CHKERR_LIBXSMM_DNN_BIND("input", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input, LIBXSMM_DNN_REGULAR_INPUT ) );

    // Tensor for Element-wise compute, if any
    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_LEFT, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_left = libxsmm_dnn_link_tensor(libxsmm_layout, in_res_ptr, &status);
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      CHKERR_LIBXSMM_DNN_BIND("input_res", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input_left, LIBXSMM_DNN_REGULAR_INPUT_LEFT ) );
    }

    // Assume that weights are in KCRS format
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_FILTER, &status );
    CHKERR_LIBXSMM_DNN( status );

    int welem = gp->nInput * gp->nOutput * gp->kw * gp->kh;
    if(gp->in_data_type == DT_FLOAT)
    {
      wt_prv_ptr = (void*)libxsmm_aligned_malloc(welem*sizeof(float), 2097152);
      libxsmm_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyin_tensor( libxsmm_filter, (void*)wt_ptr, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
      memcpy(wt_ptr, wt_prv_ptr, welem*sizeof(float));

      libxsmm_free(wt_prv_ptr);
      wt_prv_ptr = NULL;
      weightp->setPrivBuffer(NULL);
    }

    libxsmm_filter = libxsmm_dnn_link_tensor( libxsmm_layout, wt_ptr, &status );

    CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    CHKERR_LIBXSMM_DNN_BIND("wt", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_filter, LIBXSMM_DNN_REGULAR_FILTER ) );

    if(gp->bn_fwd || gp->bn_relu_fwd)
    {
      // Gamma
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_GAMMA, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_gamma  = libxsmm_dnn_link_tensor( libxsmm_layout,  gamma_ptr, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN_BIND("gamma", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_gamma, LIBXSMM_DNN_REGULAR_GAMMA ) );

      // Beta
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_BETA, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_beta  = libxsmm_dnn_link_tensor( libxsmm_layout,  beta_ptr, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN_BIND("beta", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_beta, LIBXSMM_DNN_REGULAR_BETA ) );

      // Expect
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_EXPECT, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_expect  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)expect, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN_BIND("expect", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_expect, LIBXSMM_DNN_REGULAR_EXPECT ) );

      // Rstdev
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_STDDEV, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_rstdev  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)rstdev, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN_BIND("rstdev", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_rstdev, LIBXSMM_DNN_REGULAR_STDDEV ) );
    }

    // Conv Output
    if(top_compute_engine[0] != engine)
    {
      if(out_prv_ptr == NULL)
      {

        int size = gp->batch_size * gp->nOutput * (gp->oHeight + 2*conv_desc.pad_h_out) * (gp->oWidth + 2*conv_desc.pad_w_out);
        out_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(float), 2097152);
        outp[0]->setPrivBuffer(out_prv_ptr);
      }

      out_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN(      status );
      libxsmm_output = libxsmm_dnn_link_tensor( out_buffer_layout, out_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
    }
    else
    {
      out_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN(      status );
      libxsmm_output = libxsmm_dnn_link_tensor( out_buffer_layout, out_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
    }

    CHKERR_LIBXSMM_DNN_BIND("out", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_output, LIBXSMM_DNN_REGULAR_OUTPUT ) );

    if(gp->bn_fwd || gp->bn_relu_fwd)
    {
      // Saved input buffer for BWD
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ST, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_st = libxsmm_dnn_link_tensor( libxsmm_layout, sin_ptr, &status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( status );
      CHKERR_LIBXSMM_DNN_BIND("input_st", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input_st, LIBXSMM_DNN_REGULAR_INPUT_ST ) );
    }

    if(gp->bstats_fwd)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_BATCH_STATS, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_batchstats  = libxsmm_dnn_link_tensor( libxsmm_layout, out_stats_ptr, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_batchstats, LIBXSMM_DNN_BATCH_STATS ) );
    }

    /* let's allocate (if required) and bind scratch */
    if(scratch == NULL)
    {
      long long int mysize = libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );
      CHKERR_LIBXSMM_DNN( status );
      scratch = (void*)libxsmm_aligned_malloc(mysize , 2097152);
      scratchp->setBuffer(scratch);
      scratchp->setBufferSize(mysize);

#ifdef USE_MLSL
      if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
        printf("%s allocated %lld bytes for scratch @ %p\n",nname.c_str(), mysize, scratch);
    }
    else
    {
      long long int ssize = scratchp->getBufferSize();
      long long int mysize = libxsmm_dnn_get_scratch_size( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, &status );

      CHKERR_LIBXSMM_DNN( status );

      if(ssize < mysize)
      {
        libxsmm_free(scratch);
        scratch = (void*)libxsmm_aligned_malloc(mysize, 2097152);
        scratchp->setBuffer(scratch);
        scratchp->setBufferSize(mysize);
#ifdef USE_MLSL
        if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
          printf("%s allocated %lld bytes for scratch @ %p, prev size was %lld bytes\n",nname.c_str(), mysize, scratch, ssize);
      }
    }
  }

  if(!updated_scratch)
  {
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
    updated_scratch = true;
  }

  if(bot_compute_engine[0] != engine)
  {
    assert(gp->in_data_type == DT_FLOAT);
    assert(in_buffer_layout->num_dims == 5);
    assert(in_prv_ptr != NULL);

    /* copy input data to LIBXSMM format */
    int i1, i2, i3, i4, i5;
    int N = in_buffer_layout->dim_size[4];
    int fmb = in_buffer_layout->dim_size[3];
    int bfm = in_buffer_layout->dim_size[0];
    int H = in_buffer_layout->dim_size[2];
    int W = in_buffer_layout->dim_size[1];

    LIBXSMM_VLA_DECL(4, const float, user_data, (const float*)in_ptr, fmb * bfm, H, W);
    LIBXSMM_VLA_DECL(5, float, handle_data_1, (float*)in_prv_ptr, fmb, H, W, bfm);

#ifdef _OPENMP
#pragma omp parallel for collapse(4) private(i1, i2, i3, i4, i5)
#endif
    for (i1 = 0; i1 < N; ++i1) {
      for (i2 = 0; i2 < fmb; ++i2) {
        for (i3 = 0; i3 < H; ++i3) {
          for (i4 = 0; i4 < W; ++i4) {
            for (i5 = 0; i5 < bfm; ++i5) {
              LIBXSMM_VLA_ACCESS(5, handle_data_1, i1, i2, i3, i4, i5, fmb, H, W, bfm) =
                LIBXSMM_VLA_ACCESS(4, user_data, i1, (i2*bfm) + i5, i3, i4, fmb * bfm, H, W);
            }
          }
        }
      }
    }
  }
  else
  {
    if(!destroyed_in_)
    {
      libxsmm_dnn_destroy_tensor_datalayout( in_buffer_layout );
      destroyed_in_ = true;
    }
  }

#ifdef TIMING_OV
  gettimeofday(&tve, NULL);

  double fpo_time = (tve.tv_sec + tve.tv_usec*1e-6) - (tvs.tv_sec + tvs.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    printf("Conv FP Overhead time = %g s\n",fpo_time);
  }
#endif

  if(conv_desc.options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE == false)
  {
    int nImg = gp->batch_size;
    int nOfm = gp->nOutput;
    int ofh = gp->oHeight;
    int ofw = gp->oWidth;
    float* out = (out_prv_ptr != NULL) ? (float*)out_prv_ptr : (float*)out_ptr;

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<nImg*nOfm*ofh*ofw; i++)
      out[i] = 0.0;
  }

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->opad_h > 0 || gp->opad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->opad_h == 0 || gp->opad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm forward is partially padded which cannot be :-(\n", nname.c_str());
  }

    check_physical_pad( nname.c_str(), (float*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

    check_physical_pad( nname.c_str(), (float*)out_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );

#endif

#ifdef USE_XSMM_TIMING
  struct timeval tvsc, tvec;
  gettimeofday(&tvsc, NULL);
#endif
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_FWD, 0, tid ) );
    }

#ifdef USE_XSMM_TIMING
  gettimeofday(&tvec, NULL);
  double fp_time = (tvec.tv_sec + tvec.tv_usec*1e-6) - (tvsc.tv_sec + tvsc.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    double gf = (double)gp->batch_size * (double)gp->nInput * (double)gp->nOutput * (double)gp->oHeight * (double)gp->oWidth * (double)gp->kh * (double)gp->kw * 2;
    if(gp->stride_h == 1 && gp->pad_h == 0)
      printf("XSMM-CONV-FP mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,fp_time*1000.0, gf/fp_time/1e9);
    else if(gp->stride_h == 2)
      printf("XSMM-CONV-FP mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->stride_h,fp_time*1000.0, gf/fp_time/1e9);
    else if(gp->pad_h == 1)
      printf("XSMM-CONV-FP mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->pad_h,fp_time*1000.0, gf/fp_time/1e9);
  }
#endif

#ifdef TIMING_PO
  struct timeval tvsp, tvep;

  gettimeofday(&tvsp, NULL);
#endif

  if(top_compute_engine[0] != engine)
  {
    assert(out_buffer_layout->num_dims == 5);
    assert(out_prv_ptr != NULL);

    /* copy input data to LIBXSMM format */
    int o1, o2, o3, o4, o5;
    int N = out_buffer_layout->dim_size[4];
    int fmb = out_buffer_layout->dim_size[3];
    int bfm = out_buffer_layout->dim_size[0];
    int H = out_buffer_layout->dim_size[2];
    int W = out_buffer_layout->dim_size[1];

    LIBXSMM_VLA_DECL(4, float, out_user_data, (float*)out_ptr, fmb * bfm, H, W);
    LIBXSMM_VLA_DECL(5, const float, out_handle_data_1, (const float*)out_prv_ptr, fmb, H, W, bfm);

#ifdef _OPENMP
#pragma omp parallel for collapse(4) private(o1, o2, o3, o4, o5)
#endif
    for (o1 = 0; o1 < N; ++o1) {
      for (o2 = 0; o2 < fmb; ++o2) {
        for (o3 = 0; o3 < H; ++o3) {
          for (o4 = 0; o4 < W; ++o4) {
            for (o5 = 0; o5 < bfm; ++o5) {
              LIBXSMM_VLA_ACCESS(4, out_user_data, o1, (o2*bfm) + o5, o3, o4, fmb * bfm, H, W) =
                LIBXSMM_VLA_ACCESS(5, out_handle_data_1, o1, o2, o3, o4, o5, fmb, H, W, bfm);
            }
          }
        }
      }
    }
    top_layout_type = NCHW;
    outp[0]->setLayoutType(top_layout_type);
    outp[0]->setLayout(NULL);
  }
  else
  {
    if(!destroyed_out_)
    {
      libxsmm_dnn_destroy_tensor_datalayout(out_buffer_layout);
      destroyed_out_ = true;
    }

    top_layout_type = LIBXSMM_CUSTOM_LAYOUT;
    outp[0]->setLayoutType(top_layout_type);
    outp[0]->setLayout(libxsmm_handle);
  }
#ifdef TIMING_PO
  gettimeofday(&tvep, NULL);

  double fpp_time = (tvep.tv_sec + tvep.tv_usec*1e-6) - (tvsp.tv_sec + tvsp.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    printf("Conv FP post compute time = %g s\n",fpp_time);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
    check_physical_pad( nname.c_str(), (float*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

    check_physical_pad( nname.c_str(), (float*)out_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
#endif
}

void FusedConvBNXSMM::reduce_delgamma_delbeta(float *dgbp, float *dgp, float *dbp, int nFM)
{
  int nBfm = nFM/VLEN;
  int nImg = gp->batch_size;

  __assume_aligned(dgbp, 64);
  __assume_aligned(dgp, 64);
  __assume_aligned(dbp, 64);

  float *del_gamma_imgp = dgbp;
  float *del_beta_imgp = dgbp + nImg*nFM;

  float (* __restrict del_gamma_img)[nImg][VLEN] = (float (*)[nImg][VLEN])del_gamma_imgp;
  float (* __restrict del_beta_img)[nImg][VLEN] = (float (*)[nImg][VLEN])del_beta_imgp;
  float (* __restrict del_gamma)[VLEN] = (float (*)[VLEN])dgp;
  float (* __restrict del_beta)[VLEN] = (float (*)[VLEN])dbp;

  for(int fm=0; fm < nBfm; fm++) {
    for(int img=0; img < nImg; img++) {
#pragma omp simd
#pragma vector aligned
      for(int v=0; v < VLEN; v++) {
        del_gamma[fm][v] += del_gamma_img[fm][img][v];
        del_beta[fm][v] += del_beta_img[fm][img][v];
      }
    }
  }
}

void FusedConvBNXSMM::backPropagate(TensorBuf *outp, vector<TensorBuf *>& deloutp, TensorBuf* weightp, TensorBuf *gammap, TensorBuf *delgammap, TensorBuf *delbetap, vector<TensorBuf*>& delinp, int tid)
{
#ifdef TIMING_OV
  struct timeval tvs, tve;

  gettimeofday(&tvs, NULL);
#endif

  assert(bot_compute_engine[0] != -1);
  assert(top_compute_engine[0] != -1);

  int nImg = conv_desc.N;
  int nBfm = conv_desc.K/VLEN;
  int nFM = conv_desc.K;
  int fh = gp->oHeight;
  int fw = gp->oWidth;

  out_ptr = outp->getBuffer();

  //Gamma
  gamma_ptr = gammap->getBuffer();

  //Stats of output appended to output buffer
  int offset = conv_desc.N * conv_desc.K * (gp->oHeight + 2*gp->opad_h) * (gp->oWidth + 2*gp->opad_w);
  void *out_stats_ptr = out_ptr + offset * sizeof(float);
  bmean1 = (float*)(out_stats_ptr + 2 * conv_desc.N * conv_desc.K * sizeof(float));
  brstd1 = bmean1 + conv_desc.K;

  // Saved output of FWD pass
  sout_ptr = (void*)(brstd1 + conv_desc.K);

    // Unreduced delgamma/delbeta of previous layer to be computed by BWD
  if(gp->bstats_bwd || gp->bstats_relu_bwd)
  {
    int inp_off = conv_desc.N * conv_desc.C * (gp->iHeight + 2*gp->ipad_h) * (gp->iWidth + 2*gp->ipad_w);
    dgamma_dbeta = sin_ptr + inp_off*sizeof(float);
memset(dgamma_dbeta, 0, 2*conv_desc.N * conv_desc.C*sizeof(float));    
  }

  delgamma_ptr = delgammap->getBuffer();
  delbeta_ptr = delbetap->getBuffer();
  
  if(gp->bn_bwd)
  {
    float *dgb_ptr = (float*)(sout_ptr + offset*sizeof(float));
    reduce_delgamma_delbeta(dgb_ptr, (float*)delgamma_ptr, (float*)delbeta_ptr, conv_desc.K);
  }

  // deloutput
  dout_ptr = deloutp[0]->getBuffer();
  dout_prv_ptr = deloutp[0]->getPrivBuffer();

  if(gp->split)
    dout_res_ptr = deloutp[1]->getBuffer();

  //delinput
  din_ptr = delinp[0]->getBuffer();
  din_prv_ptr = delinp[0]->getPrivBuffer();

  if(gp->eltwise)
    din_res_ptr = delinp[1]->getBuffer();

  dout_converted_in_BP = false;

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
  }

  if(libxsmm_delinput == NULL && libxsmm_delinput_left == NULL && libxsmm_deloutput == NULL && 
      libxsmm_deloutput_left == NULL && libxsmm_input_st_bwd == NULL && 
      libxsmm_input_st_bwd2 == NULL && libxsmm_gamma_bwd == NULL && libxsmm_dgamma_dbeta == NULL && 
      libxsmm_delgamma == NULL && libxsmm_delbeta == NULL && libxsmm_bmean1 == NULL && libxsmm_brstd1 == NULL
      && libxsmm_bmean2 == NULL && libxsmm_brstd2 == NULL)
  {

    // del Input
    if(bot_compute_engine[0] != engine)
    {
      if(din_prv_ptr == NULL)
      {
        int size = gp->batch_size * gp->nInput * (gp->iHeight + 2*conv_desc.pad_h_in) * (gp->iWidth + 2*conv_desc.pad_w_in);
        din_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(float), 2097152);
        delinp[0]->setPrivBuffer(din_prv_ptr);
      }

      /* setup LIBXSMM buffers and filter */
      din_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status );
      CHKERR_LIBXSMM_DNN(      status );
      libxsmm_delinput = libxsmm_dnn_link_tensor(din_buffer_layout, din_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN( status );
    }
    else
    {
      din_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT, &status );
      CHKERR_LIBXSMM_DNN_CREATE("delin", status );
      libxsmm_delinput = libxsmm_dnn_link_tensor(din_buffer_layout, din_ptr, &status );
      CHKERR_LIBXSMM_DNN_LINK("link", status );
    }
    CHKERR_LIBXSMM_DNN_BIND( "delin", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_delinput, LIBXSMM_DNN_GRADIENT_INPUT ) );

    // Element-wise BWD
    if(gp->eltwise)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_GRADIENT_INPUT_LEFT, &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_delinput_left = libxsmm_dnn_link_tensor(libxsmm_layout, din_res_ptr, &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      CHKERR_LIBXSMM_DNN_BIND("delin_res", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_delinput_left, LIBXSMM_DNN_GRADIENT_INPUT_LEFT));
    }

    // del Output
    if(top_compute_engine[0] != engine)
    {
      if(dout_prv_ptr == NULL)
      {
        int size = gp->batch_size * gp->nOutput * (gp->oHeight + 2*conv_desc.pad_h_out) * (gp->oWidth + 2*conv_desc.pad_w_out);
        dout_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(float), 2097152);
        deloutp[0]->setPrivBuffer(dout_prv_ptr);
      }

      dout_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN_CREATE("delout", status );
      libxsmm_deloutput = libxsmm_dnn_link_tensor( dout_buffer_layout, dout_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN_LINK("delout", status );
    }
    else
    {
      dout_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN_CREATE("delout", status );

      libxsmm_deloutput = libxsmm_dnn_link_tensor( dout_buffer_layout, dout_ptr, &status );

      CHKERR_LIBXSMM_DNN_LINK("delout", status );
    }

    CHKERR_LIBXSMM_DNN_BIND("delout", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );

    if(gp->split)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout(libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_SPLIT, &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_deloutput_left = libxsmm_dnn_link_tensor(libxsmm_layout, dout_res_ptr, &status);
      CHKERR_LIBXSMM_DNN(status);
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

      CHKERR_LIBXSMM_DNN_BIND("delout_res", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_deloutput_left, LIBXSMM_DNN_REGULAR_INPUT_SPLIT ) );
    }

    if(gp->bn_bwd)
    {
      // Conv output of this layer saved by next layer for this layer's BWD
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ST_BWD, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_st_bwd  = libxsmm_dnn_link_tensor( libxsmm_layout, sout_ptr, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN_BIND("saved_out", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input_st_bwd, LIBXSMM_DNN_REGULAR_INPUT_ST_BWD ) );
      
      // Gamma from FWD
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_GAMMA_BWD, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_gamma_bwd  = libxsmm_dnn_link_tensor( libxsmm_layout, gamma_ptr, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN_BIND("gamma", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_gamma_bwd, LIBXSMM_DNN_REGULAR_GAMMA_BWD) );
      
      // Reduced delgamma for BWD
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_GAMMA, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delgamma  = libxsmm_dnn_link_tensor( libxsmm_layout,  delgamma_ptr, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN_BIND("dgamma", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_delgamma, LIBXSMM_DNN_GRADIENT_GAMMA) );

      // Reduced delbeta for BWD
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_BETA, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_delbeta  = libxsmm_dnn_link_tensor( libxsmm_layout,  delbeta_ptr, &status ); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN_BIND("dbeta", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_delbeta, LIBXSMM_DNN_GRADIENT_BETA) );
    }

    if(gp->bstats_bwd || gp->bstats_relu_bwd)
    {
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_INPUT_ST_BWD2, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_input_st_bwd2  = libxsmm_dnn_link_tensor( libxsmm_layout, sin_ptr, &status); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN_BIND("saved_in", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_input_st_bwd2, LIBXSMM_DNN_REGULAR_INPUT_ST_BWD2 ) );

      // Unreduced delgamma and delbeta for previous layer
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_LCL_GAMMA_BETA, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dgamma_dbeta  = libxsmm_dnn_link_tensor( libxsmm_layout,  dgamma_dbeta, &status ); 
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN_BIND("dgamma_dbeta", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_dgamma_dbeta, LIBXSMM_DNN_REGULAR_LCL_GAMMA_BETA ) );


      // Reduced batch stats of this layer computed by next layer
      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_BMEAN2, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_bmean2 = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)expect, &status); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN_BIND("bmean2", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_bmean2, LIBXSMM_DNN_REGULAR_BMEAN2) );

      libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_BRSTD2, &status );
      CHKERR_LIBXSMM_DNN( status );
      libxsmm_brstd2 = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)rstdev, &status); CHKERR_LIBXSMM_DNN( status );
      libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
      CHKERR_LIBXSMM_DNN_BIND("brstd2", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_brstd2, LIBXSMM_DNN_REGULAR_BRSTD2 ) );
    }

    // Bmean1
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_BMEAN1, &status ); 
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_bmean1  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)bmean1, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN_BIND("bmean1", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_bmean1, LIBXSMM_DNN_REGULAR_BMEAN1 ) );

    // Rstdev1
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_REGULAR_BRSTD1, &status ); 
    CHKERR_LIBXSMM_DNN( status );
    libxsmm_brstd1  = libxsmm_dnn_link_tensor( libxsmm_layout, (void*)brstd1, &status ); CHKERR_LIBXSMM_DNN( status );
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );
    CHKERR_LIBXSMM_DNN_BIND("brstd1", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_brstd1, LIBXSMM_DNN_REGULAR_BRSTD1 ) );
  }

  if(top_compute_engine[0] != engine)
  {
    assert(dout_prv_ptr != NULL);

    /* copy input data to LIBXSMM format */
    int o1, o2, o3, o4, o5;
    int N = out_buffer_layout->dim_size[4];
    int fmb = out_buffer_layout->dim_size[3];
    int bfm = out_buffer_layout->dim_size[0];
    int H = out_buffer_layout->dim_size[2];
    int W = out_buffer_layout->dim_size[1];

    LIBXSMM_VLA_DECL(4, const float, dout_user_data, (const float*)dout_ptr, fmb * bfm, H, W);
    LIBXSMM_VLA_DECL(5, float, dout_handle_data_1, (float*)dout_prv_ptr, fmb, H, W, bfm);
#ifdef _OPENMP
#pragma omp parallel for collapse(4) private(o1, o2, o3, o4, o5)
#endif
    for (o1 = 0; o1 < N; ++o1) {
      for (o2 = 0; o2 < fmb; ++o2) {
        for (o3 = 0; o3 < H; ++o3) {
          for (o4 = 0; o4 < W; ++o4) {
            for (o5 = 0; o5 < bfm; ++o5) {
              LIBXSMM_VLA_ACCESS(5, dout_handle_data_1, o1, o2, o3, o4, o5, fmb, H, W, bfm) =
                LIBXSMM_VLA_ACCESS(4, dout_user_data, o1, (o2*bfm) + o5, o3, o4, fmb * bfm, H, W);
            }
          }
        }
      }
    }
    dout_converted_in_BP = true;
  }
  else
  {
    if(!destroyed_dout_)
    {
      libxsmm_dnn_destroy_tensor_datalayout( dout_buffer_layout );
      destroyed_dout_ = true;
    }
  }

  float* dinp = (din_prv_ptr != NULL) ? (float*)din_prv_ptr : (float*)din_ptr;

  if(conv_desc.options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE == false)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
      for(int i=0; i<gp->batch_size*gp->nInput*(gp->iHeight+2*gp->ipad_h)*(gp->iWidth+2*gp->ipad_w); i++)
        dinp[i] = 0.0;
  }

#ifdef TIMING_OV
  gettimeofday(&tve, NULL);

  double bpo_time = (tve.tv_sec + tve.tv_usec*1e-6) - (tvs.tv_sec + tvs.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    printf("Conv BP Overhead time = %g s\n",bpo_time);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->opad_h > 0 || gp->opad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->opad_h == 0 || gp->opad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
    check_physical_pad( nname.c_str(), (float*)din_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

    check_physical_pad( nname.c_str(), (float*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
#endif

#ifdef USE_XSMM_TIMING
  struct timeval tvsc, tvec;
  gettimeofday(&tvsc, NULL);
#endif

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_BWD, 0, tid ) );
    }

#ifdef USE_XSMM_TIMING
  gettimeofday(&tvec, NULL);
  double bp_time = (tvec.tv_sec + tvec.tv_usec*1e-6) - (tvsc.tv_sec + tvsc.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    double gf = (double)gp->batch_size * (double)gp->nInput * (double)gp->nOutput * (double)gp->oHeight * (double)gp->oWidth * (double)gp->kh * (double)gp->kw * 2;
    if(gp->stride_h == 1 && gp->pad_h == 0)
      printf("XSMM-CONV-BP mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,bp_time*1000.0, gf/bp_time/1e9);
    else if(gp->stride_h == 2)
      printf("XSMM-CONV-BP mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->stride_h,bp_time*1000.0, gf/bp_time/1e9);
    else if(gp->pad_h == 1)
      printf("XSMM-CONV-BP mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->pad_h,bp_time*1000.0, gf/bp_time/1e9);
  }
#endif


#ifdef TIMING_PO
  struct timeval tvsp, tvep;

  gettimeofday(&tvsp, NULL);
#endif

  if(bot_compute_engine[0] != engine)
  {
    assert(din_buffer_layout->num_dims == 5);
    assert(din_prv_ptr != NULL);

    /* copy input data to LIBXSMM format */
    int i1, i2, i3, i4, i5;
    int N = din_buffer_layout->dim_size[4];
    int fmb = din_buffer_layout->dim_size[3];
    int bfm = din_buffer_layout->dim_size[0];
    int H = din_buffer_layout->dim_size[2];
    int W = din_buffer_layout->dim_size[1];

    LIBXSMM_VLA_DECL(4, float, user_data, (float*)din_ptr, fmb * bfm, H, W);
    LIBXSMM_VLA_DECL(5, const float, handle_data_1, (const float*)din_prv_ptr, fmb, H, W, bfm);

#ifdef _OPENMP
#pragma omp parallel for collapse(4) private(i1, i2, i3, i4, i5)
#endif
    for (i1 = 0; i1 < N; ++i1) {
      for (i2 = 0; i2 < fmb; ++i2) {
        for (i3 = 0; i3 < H; ++i3) {
          for (i4 = 0; i4 < W; ++i4) {
            for (i5 = 0; i5 < bfm; ++i5) {
              LIBXSMM_VLA_ACCESS(4, user_data, i1, (i2*bfm) + i5, i3, i4, fmb * bfm, H, W) =
                LIBXSMM_VLA_ACCESS(5, handle_data_1, i1, i2, i3, i4, i5, fmb, H, W, bfm);
            }
          }
        }
      }
    }
    gbot_layout_type= NCHW;
    delinp[0]->setLayoutType(gbot_layout_type);
    delinp[0]->setLayout(NULL);
  }
  else
  {
    if(!destroyed_din_)
    {
      libxsmm_dnn_destroy_tensor_datalayout(din_buffer_layout);
      destroyed_din_ = true;
    }
    gbot_layout_type = LIBXSMM_CUSTOM_LAYOUT;
    delinp[0]->setLayoutType(gbot_layout_type);
    delinp[0]->setLayout(libxsmm_handle);
  }

#ifdef TIMING_PO
  gettimeofday(&tvep, NULL);

  double bpp_time = (tvep.tv_sec + tvep.tv_usec*1e-6) - (tvsp.tv_sec + tvsp.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    printf("Conv BP post compute time = %g s\n",bpp_time);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
    check_physical_pad( nname.c_str(), (float*)din_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

    check_physical_pad( nname.c_str(), (float*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
#endif
}

void FusedConvBNXSMM::weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf* delweightp, int tid)
{
#ifdef TIMING_OV
  struct timeval tvs, tve;

  gettimeofday(&tvs, NULL);
#endif

  if(libxsmm_deloutput == NULL)
  {
    dout_ptr = deloutp->getBuffer();
    dout_prv_ptr = deloutp->getPrivBuffer();
  }

  void *dwt_ptr = delweightp->getBuffer();

  if(scratch != scratchp->getBuffer())
  {
    scratch = scratchp->getBuffer();
    CHKERR_LIBXSMM_DNN( libxsmm_dnn_bind_scratch( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_ALL, scratch ) );
  }

  assert(bot_compute_engine[0] != -1);
  assert(top_compute_engine[0] != -1);

  if(libxsmm_delfilter == NULL)
  {
    libxsmm_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_FILTER, &status );
    CHKERR_LIBXSMM_DNN_CREATE("delwt",status );
    libxsmm_delfilter = libxsmm_dnn_link_tensor( libxsmm_layout, dwt_ptr, &status );
    CHKERR_LIBXSMM_DNN_LINK("delwt", status);
    libxsmm_dnn_destroy_tensor_datalayout( libxsmm_layout );

    /* bind buffers and filter to handle */
    CHKERR_LIBXSMM_DNN_BIND("delwt", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_delfilter, LIBXSMM_DNN_GRADIENT_FILTER ) );
  }

  if(libxsmm_deloutput == NULL)
  {
    if((top_compute_engine[0] != engine) && dout_converted_in_BP == false)
    {
      if(dout_prv_ptr == NULL)
      {
        int size = gp->batch_size * gp->nOutput * (gp->oHeight + 2*gp->opad_h) * (gp->oWidth + 2*gp->opad_w);
        dout_prv_ptr = (void*)libxsmm_aligned_malloc(size*sizeof(float), 2097152);
        deloutp->setPrivBuffer(dout_prv_ptr);
      }
      dout_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN_CREATE("delout",  status );
      libxsmm_deloutput = libxsmm_dnn_link_tensor( dout_buffer_layout, dout_prv_ptr, &status );
      CHKERR_LIBXSMM_DNN_LINK("delout", status);
    }
    else
    {
      dout_buffer_layout = libxsmm_dnn_create_tensor_datalayout( libxsmm_handle, LIBXSMM_DNN_GRADIENT_OUTPUT, &status );
      CHKERR_LIBXSMM_DNN(      status );

      libxsmm_deloutput = libxsmm_dnn_link_tensor(dout_buffer_layout, dout_ptr, &status );

      CHKERR_LIBXSMM_DNN_LINK("delout", status);
    }
    CHKERR_LIBXSMM_DNN_BIND("delout", libxsmm_dnn_bind_tensor( libxsmm_handle, libxsmm_deloutput, LIBXSMM_DNN_GRADIENT_OUTPUT ) );
  }

  if(top_compute_engine[0] != engine && dout_converted_in_BP == false)
  {
    assert(dout_prv_ptr != NULL);

    assert(dout_buffer_layout->num_dims == 5);

    /* copy input data to LIBXSMM format */
    int o1, o2, o3, o4, o5;
    int N = dout_buffer_layout->dim_size[4];
    int fmb = dout_buffer_layout->dim_size[3];
    int bfm = dout_buffer_layout->dim_size[0];
    int H = dout_buffer_layout->dim_size[2];
    int W = dout_buffer_layout->dim_size[1];
    LIBXSMM_VLA_DECL(4, const float, dout_user_data, (const float*)dout_ptr, fmb * bfm, H, W);
    LIBXSMM_VLA_DECL(5, float, dout_handle_data_1, (float*)dout_prv_ptr, fmb, H, W, bfm);

#ifdef _OPENMP
#pragma omp parallel for collapse(4) private(o1, o2, o3, o4, o5)
#endif
    for (o1 = 0; o1 < N; ++o1) {
      for (o2 = 0; o2 < fmb; ++o2) {
        for (o3 = 0; o3 < H; ++o3) {
          for (o4 = 0; o4 < W; ++o4) {
            for (o5 = 0; o5 < bfm; ++o5) {
              LIBXSMM_VLA_ACCESS(5, dout_handle_data_1, o1, o2, o3, o4, o5, fmb, H, W, bfm) =
                LIBXSMM_VLA_ACCESS(4, dout_user_data, o1, (o2*bfm) + o5, o3, o4, fmb * bfm, H, W);
            }
          }
        }
      }
    }
  }
  else
  {
    if(!destroyed_dout_)
    {
      libxsmm_dnn_destroy_tensor_datalayout( dout_buffer_layout );
      destroyed_dout_ = true;
    }
  }
  int wsize = gp->nInput * gp->nOutput * gp->kh * gp->kw;
  float *dwt = (float*)dwt_ptr;

  if(conv_desc.options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE == false)
  {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<wsize; i++)
      dwt[i] = 0.0;
  }

#ifdef TIMING_OV
  gettimeofday(&tve, NULL);

  double wuo_time = (tve.tv_sec + tve.tv_usec*1e-6) - (tvs.tv_sec + tvs.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    printf("Conv WU Overhead time = %g s\n",wuo_time);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
  if ( (gp->ipad_h > 0 || gp->ipad_w > 0) && (gp->opad_h > 0 || gp->opad_w > 0) ) {
  } else if ( (gp->ipad_h == 0 || gp->ipad_w == 0) && (gp->opad_h == 0 || gp->opad_w == 0) ) {
  } else {
    printf("node %s: conv xsmm backward is partially padded which cannot be :-(\n", nname.c_str());
  }
    check_physical_pad( nname.c_str(), (float*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

    check_physical_pad( nname.c_str(), (float*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
#endif

#ifdef USE_XSMM_TIMING
  struct timeval tvsc, tvec;
  gettimeofday(&tvsc, NULL);
#endif
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif

      CHKERR_LIBXSMM_DNN( libxsmm_dnn_execute_st( libxsmm_handle, LIBXSMM_DNN_COMPUTE_KIND_UPD, 0, tid ) );
    }

#ifdef USE_XSMM_TIMING
  gettimeofday(&tvec, NULL);
  double wu_time = (tvec.tv_sec + tvec.tv_usec*1e-6) - (tvsc.tv_sec + tvsc.tv_usec*1e-6);

#ifdef USE_MLSL
  if(MLSL::Environment::GetEnv().GetProcessIdx() == 0)
#endif
  {
    double gf = (double)gp->batch_size * (double)gp->nInput * (double)gp->nOutput * (double)gp->oHeight * (double)gp->oWidth * (double)gp->kh * (double)gp->kw * 2;
    if(gp->stride_h == 1 && gp->pad_h == 0)
      printf("XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,wu_time*1000.0, gf/wu_time/1e9);
    else if(gp->stride_h == 2)
      printf("XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dsh%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->stride_h,wu_time*1000.0, gf/wu_time/1e9);
    else if(gp->pad_h == 1)
      printf("XSMM-CONV-WU mb%dic%dih%doc%doh%dkh%dph%dn time = %g ms, GFLOPS = %.1f\n",gp->batch_size,gp->nInput,gp->iHeight,gp->nOutput,gp->oHeight,gp->kh,gp->pad_h,wu_time*1000.0, gf/wu_time/1e9);
  }
#endif

#ifndef NDEBUG
  /* check physical padding */
    check_physical_pad( nname.c_str(), (float*)in_ptr, gp->batch_size, gp->nInput/16, gp->iHeight, gp->iWidth, 16, gp->ipad_h, gp->ipad_w );

    check_physical_pad( nname.c_str(), (float*)dout_ptr, gp->batch_size, gp->nOutput/16, gp->oHeight, gp->oWidth, 16, gp->opad_h, gp->opad_w );
#endif
}

void FusedConvBNXSMM::dumpBuffer(TensorBuf* tBuf, void* wtemp)
{
  CHKERR_LIBXSMM_DNN( libxsmm_dnn_copyout_tensor( libxsmm_filter, (void*)wtemp, LIBXSMM_DNN_TENSOR_FORMAT_KCRS ) );
}

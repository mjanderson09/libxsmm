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
/* Sasikanth Avancha, Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/


#pragma once
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "common.hpp"
#include "check.hpp"
#include "FusedConvBNImpl.hpp"
#include "libxsmm.h"

#define VLEN 16

#define CHKERR_LIBXSMM_DNN(A) if ( A != LIBXSMM_DNN_SUCCESS )\
{\
  fprintf(stdout, "%s, %s\n", gp->node_name.c_str(), libxsmm_dnn_get_error(A) );\
  fflush(stdout);\
}

#define CHKERR_LIBXSMM_DNN_CREATE(t, A) if ( A != LIBXSMM_DNN_SUCCESS )\
{\
  fprintf(stdout, "Creating tensor %s in %s, %s\n", t, gp->node_name.c_str(), libxsmm_dnn_get_error(A) );\
  fflush(stdout);\
}

#define CHKERR_LIBXSMM_DNN_LINK(t, A) if ( A != LIBXSMM_DNN_SUCCESS )\
{\
  fprintf(stdout, "Linking tensor %s in %s, %s\n", t, gp->node_name.c_str(), libxsmm_dnn_get_error(A) );\
  fflush(stdout);\
}

#define CHKERR_LIBXSMM_DNN_BIND(t, A) if ( A != LIBXSMM_DNN_SUCCESS )\
{\
  fprintf(stdout, "Binding tensor %s in %s, %s\n", t, gp->node_name.c_str(), libxsmm_dnn_get_error(A) );\
  fflush(stdout);\
}

class FusedConvBNXSMM : public FusedConvBNImpl
{
  protected:
    FusedConvBNImpl *gp_;
    libxsmm_dnn_conv_desc conv_desc;
    libxsmm_dnn_layer* libxsmm_handle = NULL;
    libxsmm_dnn_tensor* libxsmm_input = NULL;
    libxsmm_dnn_tensor* libxsmm_input_st = NULL;
    libxsmm_dnn_tensor* libxsmm_input_left = NULL;
    libxsmm_dnn_tensor* libxsmm_input_st_bwd = NULL;
    libxsmm_dnn_tensor* libxsmm_input_st_bwd2 = NULL;
    libxsmm_dnn_tensor* libxsmm_output = NULL;
    libxsmm_dnn_tensor* libxsmm_filter = NULL;
    libxsmm_dnn_tensor* libxsmm_delinput = NULL;
    libxsmm_dnn_tensor* libxsmm_delinput_left = NULL;
    libxsmm_dnn_tensor* libxsmm_deloutput = NULL;
    libxsmm_dnn_tensor* libxsmm_deloutput_left = NULL;
    libxsmm_dnn_tensor* libxsmm_delfilter = NULL;
    libxsmm_dnn_tensor* libxsmm_expect = NULL;
    libxsmm_dnn_tensor* libxsmm_rstdev = NULL;
    libxsmm_dnn_tensor* libxsmm_bmean1 = NULL;
    libxsmm_dnn_tensor* libxsmm_brstd1 = NULL;
    libxsmm_dnn_tensor* libxsmm_bmean2 = NULL;
    libxsmm_dnn_tensor* libxsmm_brstd2 = NULL;
    libxsmm_dnn_tensor* libxsmm_gamma = NULL;
    libxsmm_dnn_tensor* libxsmm_gamma_bwd = NULL;
    libxsmm_dnn_tensor* libxsmm_beta = NULL;
    libxsmm_dnn_tensor* libxsmm_delgamma = NULL;
    libxsmm_dnn_tensor* libxsmm_delbeta = NULL;
    libxsmm_dnn_tensor* libxsmm_dgamma_dbeta = NULL;
    libxsmm_dnn_tensor* libxsmm_batchstats = NULL;
    libxsmm_dnn_tensor_datalayout* libxsmm_layout;
    libxsmm_dnn_err_t status;

    FusedConvBNImplParams *cp;
    float *dinptr, *dwtptr;
    libxsmm_dnn_tensor_datalayout *in_buffer_layout, *out_buffer_layout;
    libxsmm_dnn_tensor_datalayout *din_buffer_layout, *dout_buffer_layout;
    bool dout_converted_in_BP = false;
    bool destroyed_in_=false, destroyed_out_=false, destroyed_din_=false, destroyed_dout_=false;
    bool updated_scratch=false;
    void *in_ptr=NULL, *in_prv_ptr=NULL, *wt_ptr=NULL, *out_ptr=NULL, *out_prv_ptr=NULL;
    void *in_res_ptr=NULL, *din_res_ptr=NULL, *dout_res_ptr=NULL;
    void *gamma_ptr=NULL, *beta_ptr=NULL, *dgamma_dbeta=NULL, *delgamma_ptr=NULL, *delbeta_ptr=NULL;
    void *sin_ptr=NULL, *sout_ptr=NULL, *in_ptr_left=NULL, *din_ptr_left;
    void *din_ptr=NULL, *din_prv_ptr=NULL, *dwt_ptr=NULL, *dwt_prv_ptr=NULL, *dout_ptr=NULL, *dout_prv_ptr=NULL;
    float *expect=NULL, *rstdev=NULL, *bmean1=NULL, *brstd1=NULL, *bmean2=NULL, *brstd2=NULL;
    void *scratch=NULL;

  public:
    FusedConvBNXSMM(FusedConvBNImplParams *gp, int engine);
    virtual ~FusedConvBNXSMM(void) {}
    void forwardPropagate(vector<TensorBuf *>& inp, TensorBuf* weightp, TensorBuf* gammap, TensorBuf* betap, TensorBuf* mygammap, TensorBuf* mybetap, TensorBuf *gmeanp, TensorBuf *grstdp, vector<TensorBuf *>& outp, int tid);
    void backPropagate(TensorBuf* outp, vector<TensorBuf *>& deloutp, TensorBuf* weightp, TensorBuf *gammap, TensorBuf* delgammap, TensorBuf* delbetap, vector<TensorBuf *>& delinp, int tid);
    void weightUpdate(TensorBuf *inp, TensorBuf *deloutp, TensorBuf *delweightp, int tid);
    void dumpBuffer(TensorBuf *wt, void* temp);
    void reduce_batch_stats(void *bstats_ip, float *bmeanp, float *brstdp, TensorBuf *gmeanpb, TensorBuf *grstdpb, int nFM, int fh, int fw);
    void reduce_delgamma_delbeta(float *dgbp, float *dgp, float *dbp, int nFM);
};

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
#include <string>
#include <stdio.h>
#include "assert.h"
#include "Node.hpp"
#include "Engine.hpp"
#include "Params.hpp"
#include "Tensor.hpp"
#include "Solver.hpp"
#include "proto/gxm.pb.h"
#include "FusedConvBNImpl.hpp"
#include "FusedConvBNXSMM.hpp"

using namespace std;
using namespace gxm;

class FusedConvBNParams : public NNParams
{
  public:
    FusedConvBNParams(void) {}

    virtual ~FusedConvBNParams(void) {}

    void set_kernel_dims(int kdims, int ksize)
    {
      for(int i=0; i<kdims; i++)
        this->kernel_dim_.push_back(ksize);
    }

    void set_kernel_dims(int kh, int kw, int kd)
    {
      this->kernel_dim_.push_back(kh);
      this->kernel_dim_.push_back(kw);
      this->kernel_dim_.push_back(kd);
    }

    vector<int>& get_kernel_dims() { return kernel_dim_; }

    void set_strides(int sdims, int stride)
    {
      for(int i=0; i<sdims; i++)
        this->strides_.push_back(stride);
    }

    void set_strides(int sh, int sw, int sd)
    {
      this->strides_.push_back(sh);
      this->strides_.push_back(sw);
      this->strides_.push_back(sd);
    }

    vector<int>& get_strides() { return strides_; }

    void set_pads(int pdims, int pad)
    {
      for(int i=0; i<pdims; i++)
        this->pads_.push_back(pad);
    }
    void set_pads(int ph, int pw, int pd)
    {
      this->pads_.push_back(ph);
      this->pads_.push_back(pw);
      this->pads_.push_back(pd);
    }
    vector<int>& get_pads() { return pads_; }

    void set_output_pads(int pdims, int pad)
    {
      for(int i=0; i<pdims; i++)
        this->opads_.push_back(pad);
    }

    void set_output_pads(int ph, int pw, int pd)
    {
      this->opads_.push_back(ph);
      this->opads_.push_back(pw);
      this->opads_.push_back(pd);
    }
    vector<int>& get_output_pads() { return opads_; }

    void set_group(int g) { this->group_ = g;}
    int get_group() { return this->group_; }

    void set_nOutput(int num_output) { this->nOutput_ = num_output; }
    int get_output() { return nOutput_; }

    void set_weight_filler_type(string ftype) { wfiller_type_ = ftype; }
    string get_weight_filler_type() { return wfiller_type_; }

    void set_std(float s) { std_ = s; }
    float get_std() { return std_; }

    void set_variance_norm(int v) { variance_norm_ = v; }
    int get_variance_norm() { return variance_norm_; }

    void set_eps(float eps) { eps_ = eps; }
    float get_eps() { return eps_; }

    void set_mmf(float mmf) { mmf_ = mmf; }
    float get_mmf() { return mmf_; }

    void set_global_stats_flag(bool s) { use_global_stats_ = s; }
    bool get_global_stats_flag() { return use_global_stats_; }

    void set_eltwise(bool e) { eltwise_ = e; }
    bool get_eltwise() { return eltwise_; }

    void set_split(bool s) { split_ = s; }
    bool get_split() { return split_; }

    void set_relu_fwd(bool relu_fwd) { relu_fwd_ = relu_fwd; }
    bool get_relu_fwd() { return relu_fwd_; }

    void set_relu_bwd(bool br) { relu_bwd_ = br; }
    bool get_relu_bwd() { return relu_bwd_; }

    void set_bn_fwd(bool bn_fwd) { bn_fwd_ = bn_fwd; }
    bool get_bn_fwd() { return bn_fwd_; }

    void set_bn_bwd(bool bn_bwd) { bn_bwd_ = bn_bwd; }
    bool get_bn_bwd() { return bn_bwd_; }

    void set_bn_relu_fwd(bool brf) { bn_relu_fwd_ = brf; }
    bool get_bn_relu_fwd() { return bn_relu_fwd_; }

    void set_bstats_fwd(bool s) { bstats_fwd_ = s; }
    bool get_bstats_fwd() { return bstats_fwd_; }

    void set_bstats_bwd(bool s) { bstats_bwd_ = s; }
    bool get_bstats_bwd() { return bstats_bwd_; }

    void set_bstats_relu_bwd(bool s) { bstats_relu_bwd_ = s; }
    bool get_bstats_relu_bwd() { return bstats_relu_bwd_; }

    void set_physical_padding(bool p) { phys_pad_ = p; }
    bool get_physical_padding() { return phys_pad_; }

    void set_compute_engine(int ce) { compute_engine_ = ce; }
    int get_compute_engine() { return compute_engine_; }

    void set_algo_type(int at) { algotype_ = at; }
    int get_algo_type() { return algotype_; }

    void set_global_params(vector<ParamSpec> psv)
    {
      for(int i=0; i<psv.size(); i++)
      {
        lr_mult_.push_back(psv[i].lr_mult());
        decay_mult_.push_back(psv[i].decay_mult());
      }
    }
    const vector<float>& get_lr_mult() { return lr_mult_; }
    const vector<float>& get_decay_mult() { return decay_mult_; }

    void set_data_type(int t) { data_type_ = t; }
    int get_data_type() { return data_type_; }

  protected:
    vector<int> kernel_dim_; // Order of dimensions is Height, Width, Depth (for 3D Conv)
    vector<int> strides_;    // Order follows kernel dimension
    vector<int> pads_, opads_;       // Order follows kernel dimension
    int nOutput_;            // Number of output feature maps
    string wfiller_type_; 
    float std_, eps_, mmf_;
    bool relu_fwd_, relu_bwd_, bn_fwd_, bn_bwd_;
    bool bn_relu_fwd_, bstats_fwd_, bstats_bwd_, bstats_relu_bwd_;
    bool split_, eltwise_;
    bool phys_pad_, use_global_stats_;
    int group_, compute_engine_, algotype_;
    int variance_norm_, data_type_;
    vector<float> lr_mult_, decay_mult_;
};

static MLParams* parseFusedConvBNParams(NodeParameter* np)
{

  FusedConvBNParams* fcbnp = new FusedConvBNParams();

  // Set name of node
  string str = np->name();
  assert(!str.empty());
  fcbnp->set_node_name(str);

  //Set node type (Convolution, FullyConnected, etc)
  str = np->type();
  assert(!str.empty());
  fcbnp->set_node_type(str);

  //Set tensor names
  for(int i=0; i<np->bottom_size(); i++)
  {
    assert(!np->bottom(i).empty());
    fcbnp->set_bottom_names(np->bottom(i));
  }

  for(int i=0; i<np->top_size(); i++)
  {
    assert(!np->top(i).empty());
    fcbnp->set_top_names(np->top(i));
  }

  //Set Mode for the node
  assert((np->mode() == TRAIN) || (np->mode() == TEST));
  fcbnp->set_mode(np->mode());

  //Set backprop needed/not needed flag for this node
  fcbnp->set_bprop_flag(np->propagate_down());

  vector<ParamSpec> psv;
  for(int i=0; i<np->param_size(); i++)
    psv.push_back(np->param(i));
  fcbnp->set_global_params(psv);

  FusedConvBNParameter pcp = np->fused_conv_bn_param();

  int kdims = pcp.kernel_size_size();

  switch(kdims)
  {
    int kh, kw, kd;
    case 0:
      kh = pcp.kernel_h();
      kw = pcp.kernel_w();
      if(pcp.ndims() == 3)
        kd = pcp.kernel_d();
      else
        kd = 0;

      assert((kh > 0) && (kw > 0));
      fcbnp->set_kernel_dims(kh, kw, kd);
      break;

    case 1:
      kh = pcp.kernel_size(0);
      if(pcp.ndims() == 2)
        fcbnp->set_kernel_dims(kh, kh, 0);
      else if(pcp.ndims() == 3)
        fcbnp->set_kernel_dims(kh, kh, kh);
      break;

    case 2:
      kh = pcp.kernel_size(0);
      kw = pcp.kernel_size(1);
      assert(pcp.ndims() == 2);
      fcbnp->set_kernel_dims(kh, kw, 0);
      break;

    case 3:
      kh = pcp.kernel_size(0);
      kw = pcp.kernel_size(1);
      kd = pcp.kernel_size(2);
      assert(pcp.ndims() == 3);
      fcbnp->set_kernel_dims(kh, kw, kd);
      break;
  }

  // strides
  int sdims = pcp.stride_size();
  switch(sdims)
  {
    int sh, sw, sd;

    case 0:
      sh = pcp.stride_h();
      sw = pcp.stride_w();
      if(pcp.ndims() == 3)
        sd = pcp.stride_d();
      else
        sd = 0;

      assert((sh > 0) && (sw > 0));
      fcbnp->set_strides(sh, sw, sd);
      break;

    case 1:
      sh = pcp.stride(0);
      if(pcp.ndims() == 2)
      fcbnp->set_strides(sh, sh, 0);
      else if(pcp.ndims() == 3)
      fcbnp->set_strides(sh, sh, sh);
      break;

    case 2:
      sh = pcp.stride(0);
      sw = pcp.stride(1);
      assert(pcp.ndims() == 2);
      fcbnp->set_strides(sh, sw, 0);
      break;

    case 3:
      sh = pcp.stride(0);
      sw = pcp.stride(1);
      sd = pcp.stride(2);
      assert(pcp.ndims() == 3);
      fcbnp->set_strides(sh, sw, sd);
      break;
  }

  // pads
  int pdims = pcp.pad_size();
  switch(pdims)
  {
    int ph, pw, pd;
    case 0:
      ph = pcp.pad_h();
      pw = pcp.pad_w();
      if(pcp.ndims() == 3)
        pd = pcp.pad_d();
      else
        pd = 0;

      fcbnp->set_pads(ph, pw, pd);
      break;

    case 1:
      ph = pcp.pad(0);
      if(pcp.ndims() == 2)
        fcbnp->set_pads(ph, ph, 0);
      else if(pcp.ndims() == 3)
        fcbnp->set_pads(ph, ph, ph);
      break;

    case 2:
      ph = pcp.pad(0);
      pw = pcp.pad(1);
      assert(pcp.ndims() == 2);
      fcbnp->set_pads(ph, pw, 0);
      break;

    case 3:
      ph = pcp.pad(0);
      pw = pcp.pad(1);
      pd = pcp.pad(2);
      assert(pcp.ndims() == 3);
      fcbnp->set_pads(ph, pw, pd);
      break;
  }

  // output pads
  int opdims = pcp.opad_size();
  switch(opdims)
  {
    int oph, opw, opd;
    case 0:
      oph = pcp.opad_h();
      opw = pcp.opad_w();
      if(pcp.ndims() == 3)
        opd = pcp.opad_d();
      else
        opd = 0;

      fcbnp->set_output_pads(oph, opw, opd);
      break;

    case 1:
      oph = pcp.opad(0);
      if(pcp.ndims() == 2)
        fcbnp->set_output_pads(oph, oph, 0);
      else if(pcp.ndims() == 3)
        fcbnp->set_output_pads(oph, oph, oph);
      break;

    case 2:
      oph = pcp.opad(0);
      opw = pcp.opad(1);
      assert(pcp.ndims() == 2);
      fcbnp->set_output_pads(oph, opw, 0);
      break;

    case 3:
      oph = pcp.opad(0);
      opw = pcp.opad(1);
      opd = pcp.opad(2);
      assert(pcp.ndims() == 3);
      fcbnp->set_output_pads(oph, opw, opd);
      break;
  }

  if(pcp.group() > 1)
    fcbnp->set_group(pcp.group());
  else
    fcbnp->set_group(1);

  int nOutput = pcp.num_output();
  fcbnp->set_nOutput(nOutput);

  fcbnp->set_mmf(pcp.mmf());
  fcbnp->set_eps(pcp.eps());
  fcbnp->set_global_stats_flag(pcp.use_global_stats());
  fcbnp->set_relu_fwd(pcp.relu_fwd());
  fcbnp->set_relu_bwd(pcp.relu_bwd());
  fcbnp->set_bn_fwd(pcp.bn_fwd());
  fcbnp->set_bn_bwd(pcp.bn_bwd());
  fcbnp->set_bn_relu_fwd(pcp.bn_relu_fwd());
  fcbnp->set_bstats_fwd(pcp.bstats_fwd());
  fcbnp->set_bstats_bwd(pcp.bstats_bwd());
  fcbnp->set_bstats_relu_bwd(pcp.bstats_relu_bwd());

  FillerParameter wp = pcp.weight_filler();
  fcbnp->set_weight_filler_type(wp.type());
  fcbnp->set_std(wp.std());
  fcbnp->set_variance_norm(wp.variance_norm());

  fcbnp->set_eltwise(pcp.eltwise());
  fcbnp->set_split(pcp.split());

  fcbnp->set_physical_padding(pcp.physical_padding());

  fcbnp->set_data_type(pcp.data_type());
  fcbnp->set_compute_engine(pcp.engine());
  fcbnp->set_algo_type(pcp.algotype());

  return fcbnp;
}

class FusedConvBNNode : public NNNode
{
  public:
    FusedConvBNNode(FusedConvBNParams* p, MLEngine* e);

    virtual ~FusedConvBNNode(void) {}

    string get_weight_filler_type() { return wfiller_type_; }
    float get_std() { return std_; }

    void fillWeightBuffers(TensorBuf* tBuf, int buftype, long long int size);
    void fillBuffer(TensorBuf* tBuf, int buftype, long long int size);
    void fillWeightMultipliers(float* lr_mult, float* decay_mult, long long int bytes);
    void fillBiasMultipliers(float* lr_mult, float* decay_mult, long long int bytes);
    void Checkpoint(TensorBuf* tBuf, string name, string format);
    TensorBuf* getScaleBuf() { return tenScaleData_; }
    TensorBuf* getShiftBuf() { return tenShiftData_; }

  protected:
    void forwardPropagate();
    void backPropagate();
    void weightUpdate();
    void solverStep();

    void configure(int engine);

    void shape_setzero(Shape* s)
    {
      for(int i=0; i<MAX_DIMS; i++)
        s->dims[i] = 0;
    }

    vector<Tensor *> tenTop_, tenBot_;
    Tensor *tenWeight_, *tenScale_, *tenShift_, *tenMean_, *tenRstdev_;

    FusedConvBNImplParams gparams_;
    vector<TensorBuf *> tenBotDiff_, tenBotData_; // Data & Gradients with respect to input
    vector<TensorBuf *> tenTopData_, tenTopDiff_; 
    TensorBuf *tenWeightDiff_, *tenWeightData_, *tenWeightInc_; // Weight gradients, data, increments
    TensorBuf *tenScaleData_, *tenScaleDiff_, *tenScaleInc_; // Gamma data, gradients, increments
    TensorBuf *tenShiftData_, *tenShiftDiff_, *tenShiftInc_; // Beta data, gradients, increments
    TensorBuf *tenMeanData_, *tenRstdevData_; // Mean, 1/stddev data
    TensorBuf *tenScratchData_;

    Shape ts_, ws_;
    string wfiller_type_;
    string weight_, scale_, shift_, mean_, rstdev_;
    int variance_norm_;
    float std_;
    int bot_cengine_;
    int count_;
    vector<float> lr_mult_, decay_mult_;
    bool first_fp = true, first_bp=true;
    bool bstats_;

    FusedConvBNImpl *impl=NULL;

    SolverNode *solver_;
    MLEngine* eptr_;
};




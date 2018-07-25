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

#include <string>
#include "FusedConvBN.hpp"
#include "fillers.hpp"

#ifdef USE_MLSL
#include "mpi.h"
#endif


using namespace std;
using namespace gxm;

FusedConvBNNode::FusedConvBNNode(FusedConvBNParams* p, MLEngine* e): NNNode(p, e)
{
  nname_ = p->get_node_name();
  ntype_ = p->get_node_type();
  bottom_ = p->get_bottom_names();
  top_ = p->get_top_names();
  bp_flag_ = p->get_bprop_flag();
  has_weights_ = true;
  bstats_ = p->get_bstats_fwd();
  bot_compute_engine_ = p->get_compute_engine();

  tenTop_.resize(top_.size());
  tenTopData_.resize(top_.size());

  for(int i=0; i < top_.size(); i++)
  {
    tenTop_[i] = new Tensor(top_[i]);
    assert(tenTop_[i] != NULL);
    tenTop_[i]->setOwner(this);
    tenTop_[i]->setType(ACT);
    tenTopData_[i] = tenTop_[i]->getBuf(DATA);
    tenTopData_[i]->setBufferType(DATA);
  }

  tenBot_.resize(bottom_.size());
  tenBotData_.resize(bottom_.size());

  for(int i=0; i < bottom_.size(); i++)
  {
#ifndef NDEBUG
    printf("bottom%d name %s\n",i,bottom_[i].c_str());
#endif

    if(bottom_[i] == "data")
      tenBot_[i] = e->get_tensor(bottom_[i], INPUT);
    else
      tenBot_[i] = e->get_tensor(bottom_[i], ACT);

    assert(tenBot_[i] != NULL);
    NNNode *pnn = (NNNode*)tenBot_[i]->getOwner();
    setPrevNode(pnn);
    mode_ = pnn->getMode();
    pnn->set_top_compute_engine(p->get_compute_engine());
    bot_cengine_ = pnn->get_bot_compute_engine();

    tenBotData_[i] = tenBot_[i]->getBuf(DATA);

    int out_dtype = p->get_data_type();

    tenTopData_[i]->setDataType(out_dtype);
  }

  // Get input tensor shape (bottom)
  Shape* bs = tenBot_[0]->getShape();
  assert(bs->ndims <= MAX_DIMS);

  // Create shape of output tensor (top)
  vector<int> vd = p->get_kernel_dims();
  vector<int> ovp = p->get_output_pads();
  vector<int> vp = p->get_pads();
  vector<int> vs = p->get_strides();

  assert((vd.size() == vp.size()) && (vd.size() == vs.size()) && (vs.size() == ovp.size()));

  shape_setzero(&ts_);

  ts_.ndims = bs->ndims; // Number of dimensions
  ts_.dims[0] = bs->dims[0]; // Minibatch size
  ts_.dims[1] = p->get_output(); // Num output feature maps

  gparams_.physical_padding = p->get_physical_padding();

  if(vd[0] == 1 && gparams_.physical_padding)
    ts_.dims[2] = (bs->dims[2] - vd[0])/vs[0] + 1; // Height
  else
    ts_.dims[2] = (bs->dims[2] - vd[0] + 2*vp[0])/vs[0] + 1; // Height

  if(ts_.ndims == 4)
    if(vd[1] == 1  && gparams_.physical_padding)
      ts_.dims[3] = (bs->dims[3] - vd[1])/vs[1] + 1; // Width
    else
      ts_.dims[3] = (bs->dims[3] - vd[1] + 2*vp[1])/vs[1] + 1; // Width
  else if(ts_.ndims == 5)
  {
    if(vd[1]==1 && gparams_.physical_padding)
      ts_.dims[3] = (bs->dims[3] - vd[1])/vs[1] + 1; // Width
    else
      ts_.dims[3] = (bs->dims[3] - vd[1] + 2*vp[1])/vs[1] + 1; // Width

    if(vd[2]==1 && gparams_.physical_padding)
      ts_.dims[4] = (bs->dims[4] - vd[2])/vs[2] + 1; // Width
    else
      ts_.dims[4] = (bs->dims[4] - vd[2] + 2*vp[2])/vs[2] + 1; // Depth (for 3D)
  }

  tenTop_[0]->setShape(&ts_);
  if(top_.size() > 1)
    tenTop_[1]->setShape(tenBot_[1]->getShape());

  long long int tsize;
  int telem;
  
  if(gparams_.physical_padding)
    telem = ts_.dims[0] * ts_.dims[1] * (ts_.dims[2] + 2*ovp[0]) * (ts_.dims[3] + 2*ovp[1]);
  else
    telem = ts_.dims[0] * ts_.dims[1] * ts_.dims[2] * ts_.dims[3];

  // Buffer space for sum and sum^2
  int tstats;
  if(bstats_)
    tstats = 2*ts_.dims[0]*ts_.dims[1];

  // Conv out + Conv saved out + unreduced batch stats + reduced batch stats + unreduced dgamma/dbeta + reduced dgamma/dbeta
  tsize = 2*telem*sizeof(float) + tstats*(sizeof(float) + sizeof(float)) + 4*ts_.dims[1]*sizeof(float); 

  tenTopData_[0]->setBufferSize(tsize);

  // Create FP weight tensor
  weight_ = top_[0] + "_wt";
  tenWeight_ = new Tensor(weight_);
  assert(tenWeight_ != NULL);
  tenWeight_->setOwner(this);
  tenWeight_->setType(CONVWEIGHT);

  shape_setzero(&ws_);

  ws_.ndims = ts_.ndims;      // Number of dimesions
  ws_.dims[0] = ts_.dims[1];  // Num output feature maps (from top tensor)
  ws_.dims[1] = bs->dims[1];  // Num input feature maps (from bottom tensor)
  ws_.dims[2] = vd[0];        // Kernel height

  if(ts_.ndims == 4)
  {
    ws_.dims[3] = vd[1]; // Kernel width
  }
  else if(ts_.ndims == 5)
  {
    ws_.dims[3] = vd[1];
    ws_.dims[4] = vd[2];
  }

  tenWeight_->setShape(&ws_);
  tenWeight_->setBufDataType(DATA, DT_FLOAT);
  tenWeightData_ = tenWeight_->getBuf(DATA);
  tenWeightData_->setBufferType(DATA);

  int welem = 1;
  long long int wsize;
  for(int i=0; i<ws_.ndims; i++)
    welem = welem*ws_.dims[i];

  int in_dtype = tenBotData_[0]->getDataType();
  int out_dtype = tenTopData_[0]->getDataType();

  // size of weights -- always in FP32.
  if((in_dtype == DT_FLOAT) && (out_dtype == DT_FLOAT))
    wsize = welem*sizeof(float);
  else if(in_dtype == DT_DFP16)
    wsize = welem*sizeof(float);

  tenWeightData_->setBufferSize(wsize);

  wfiller_type_ = p->get_weight_filler_type();
  variance_norm_ = p->get_variance_norm();
  std_ = p->get_std();

  lr_mult_ = p->get_lr_mult();
  decay_mult_ = p->get_decay_mult();

  Shape sss;
  shape_setzero(&sss);
  sss.ndims = 1;
  sss.dims[0] = ts_.dims[1];

  scale_ = top_[0] + "_scale";
  tenScale_ = new Tensor(scale_);
  assert(tenScale_ != NULL);
  tenScale_->setOwner(this);
  tenScale_->setType(BNORMSCALE);
  tenScale_->setShape(&sss);
  tenScaleData_ = tenScale_->getBuf(DATA);
  tenScaleData_->setDataType(DT_FLOAT); //TODO: Eventually move to 16-bit? Currently it is FP32
  tenScaleData_->setBufferType(DATA);

  telem = sss.dims[0];
  tsize = telem*sizeof(float);
  tenScaleData_->setBufferSize(tsize);

  shift_ = top_[0] + "_shift";
  tenShift_ = new Tensor(shift_);
  assert(tenShift_ != NULL);
  tenShift_->setOwner(this);
  tenShift_->setType(BNORMSHIFT);
  tenShift_->setShape(&sss);
  tenShiftData_ = tenShift_->getBuf(DATA);
  tenShiftData_->setDataType(DT_FLOAT); //TODO: Eventually move to dfp16 beta. Currently it is FP32
  tenShiftData_->setBufferType(DATA);

  tenShiftData_->setBufferSize(tsize);

  mean_ = top_[0] + "_mean";
  tenMean_ = new Tensor(mean_);
  assert(tenMean_ != NULL);
  tenMean_->setOwner(this);
  tenMean_->setType(BNORMMEAN);
  tenMean_->setShape(&sss);
  tenMeanData_ = tenMean_->getBuf(DATA);
  tenMeanData_->setDataType(DT_FLOAT);
  tenMeanData_->setBufferType(DATA);
  tenMeanData_->setBufferSize(tsize);

  rstdev_ = top_[0] + "_rstdev";
  tenRstdev_ = new Tensor(rstdev_);
  assert(tenRstdev_ != NULL);
  tenRstdev_->setOwner(this);
  tenRstdev_->setType(BNORMRSTDEV);
  tenRstdev_->setShape(&sss);
  tenRstdevData_ = tenRstdev_->getBuf(DATA);
  tenRstdevData_->setDataType(DT_FLOAT);
  tenRstdevData_->setBufferType(DATA);
  tenRstdevData_->setBufferSize(tsize);

  if(!e->is_inference_only()) {
    tenBotDiff_.resize(bottom_.size());
    if(bp_flag_)
    {
      for(int i=0; i<bottom_.size(); i++)
      {
        tenBotDiff_[i] = tenBot_[i]->addBuf(); // DIFF type and index
        tenBotDiff_[i]->setDataType(in_dtype);
        tenBotDiff_[i]->setBufferType(DIFF);
      }

      long long int bsize;
      
      if(gparams_.physical_padding)
        bsize = bs->dims[0] * bs->dims[1] * (bs->dims[2] + 2*vp[0]) * (bs->dims[3] + 2*vp[1]);
      else
        bsize = bs->dims[0] * bs->dims[1] * bs->dims[2] * bs->dims[3];

      if((in_dtype == DT_FLOAT && out_dtype == DT_FLOAT) ||
          (in_dtype == DT_DFP16 && out_dtype == DT_FLOAT))
        bsize = bsize*sizeof(float);
      else if(in_dtype == DT_DFP16 && out_dtype == DT_DFP16)
        bsize = bsize*sizeof(short);

      // Set the size of the input-gradient buffer
      for(int i=0; i<bottom_.size(); i++)
        tenBotDiff_[i]->setBufferSize(bsize);
    }

    if(has_weights_)
    {
      tenWeightDiff_ = tenWeight_->addBuf(); // DIFF type and index
      tenWeightDiff_->setDataType(DT_FLOAT);
      tenWeightDiff_->setBufferType(DIFF);

      tenWeightInc_ = tenWeight_->addBuf(); // SHARED type and index
      tenWeightInc_->setDataType(DT_FLOAT);
      tenWeightInc_->setBufferType(HISTORY);

      // Set the size of the weight-gradient buffer and the weight-increment buffer
      tenWeightDiff_->setBufferSize(welem*sizeof(float));
      tenWeightInc_->setBufferSize(welem*sizeof(float));

      tenScaleDiff_ = tenScale_->addBuf();
      tenScaleDiff_->setDataType(DT_FLOAT);
      tenScaleDiff_->setBufferType(DIFF);
      tenScaleDiff_->setBufferSize(tsize);

      tenScaleInc_ = tenScale_->addBuf();
      tenScaleInc_->setDataType(DT_FLOAT);
      tenScaleInc_->setBufferType(HISTORY);
      tenScaleInc_->setBufferSize(tsize);

      tenShiftDiff_ = tenShift_->addBuf();
      tenShiftDiff_->setDataType(DT_FLOAT);
      tenShiftDiff_->setBufferType(DIFF);
      tenShiftDiff_->setBufferSize(tsize);

      tenShiftInc_ = tenShift_->addBuf();
      tenShiftInc_->setDataType(DT_FLOAT);
      tenShiftInc_->setBufferType(HISTORY);
      tenShiftInc_->setBufferSize(tsize);
    }
  }
  else {
    tenBotDiff_.resize(bottom_.size());
    for(int i=0; i<bottom_.size(); i++)
      tenBotDiff_[i] = NULL;
    tenWeightDiff_ = NULL;
    tenWeightInc_ = NULL;
    tenScaleDiff_ = NULL;
    tenShiftDiff_ = NULL;
    tenScaleInc_ = NULL;
    tenShiftInc_ = NULL;
  }

  // Register output tensor in tensor map
  bool inserted;
  for(int i=0; i<top_.size(); i++)
  {
    inserted = e->register_tensor(top_[i], ACT, tenTop_[i]);
    if(!inserted)
      printf("Warning: Tensor %s already registered\n",top_[i].c_str());
  }

  // Register weight tensor in weight tensor map
  inserted = e->register_tensor(weight_, CONVWEIGHT, tenWeight_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",weight_.c_str());

  inserted = e->register_tensor(scale_, BNORMSCALE, tenScale_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",scale_.c_str());

  inserted = e->register_tensor(shift_, BNORMSHIFT, tenShift_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",shift_.c_str());

  inserted = e->register_tensor(mean_, BNORMMEAN, tenMean_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",mean_.c_str());

  inserted = e->register_tensor(rstdev_, BNORMRSTDEV, tenRstdev_);
  if(!inserted)
    printf("Warning: Tensor %s already registered\n",rstdev_.c_str());

  // Setup parameter structure for convolution computation in library
  gparams_.bdims = bs->ndims;
  gparams_.tdims = ts_.ndims;
  gparams_.wdims = ws_.ndims;

  gparams_.node_name = nname_;
  gparams_.node_type = ntype_;
  gparams_.nInput = bs->dims[1];
  gparams_.nOutput = ts_.dims[1];
  gparams_.batch_size = bs->dims[0];
  gparams_.iHeight = bs->dims[2];
  gparams_.iWidth = bs->dims[3];
  gparams_.iDepth = bs->dims[4];
  gparams_.oHeight = ts_.dims[2];
  gparams_.oWidth = ts_.dims[3];
  gparams_.oDepth = ts_.dims[4];
  gparams_.pad_h = vp[0];
  gparams_.pad_w = vp[1];
  gparams_.pad_d = vp[2];

  if(gparams_.physical_padding)
  {
    gparams_.ipad_h = (nname_ == "conv1") ? 0 : vp[0];
    gparams_.ipad_w = (nname_ == "conv1") ? 0 : vp[1];
    gparams_.ipad_d = (nname_ == "conv1") ? 0 : vp[2];
  }
  else
  {
    gparams_.ipad_h = 0;
    gparams_.ipad_w = 0;
    gparams_.ipad_d = 0;
  }

  if(gparams_.physical_padding)
  {
    gparams_.opad_h = (nname_ == "conv1") ? 0 : ovp[0];
    gparams_.opad_w = (nname_ == "conv1") ? 0 : ovp[1];
  }
  else
  {
    gparams_.opad_h = 0;
    gparams_.opad_w = 0;
    gparams_.opad_d = 0;
  }

  gparams_.group = p->get_group();
  gparams_.stride_h = vs[0];
  gparams_.stride_w = vs[1];
  gparams_.stride_d = vs[2];
  gparams_.kh = ws_.dims[2];
  gparams_.kw = ws_.dims[3];
  gparams_.kd = ws_.dims[4];

  gparams_.relu_fwd = p->get_relu_fwd();
  gparams_.relu_bwd = p->get_relu_bwd();
  gparams_.bn_fwd = p->get_bn_fwd();
  gparams_.bn_bwd = p->get_bn_bwd();
  gparams_.bn_relu_fwd = p->get_bn_relu_fwd();
  gparams_.bstats_fwd = bstats_;
  gparams_.bstats_bwd = p->get_bstats_bwd();
  gparams_.bstats_relu_bwd = p->get_bstats_relu_bwd();
  gparams_.split = p->get_split();

  gparams_.mmf = p->get_mmf();
  gparams_.eps = p->get_eps();
  gparams_.use_global_stats = p->get_global_stats_flag();
  gparams_.eltwise = p->get_eltwise();

  gparams_.in_data_type = in_dtype;
  gparams_.out_data_type = out_dtype;
  gparams_.algType = p->get_algo_type();
  gparams_.num_threads = e->get_num_threads();

  // get solver
  solver_ = e->getSolver();

  //get global scratch tensor buffer
  tenScratchData_ = e->getScratchBuffer();

  // get engine
  eptr_ = e;

#ifdef USE_MLSL
  MLSL::DataType dt = MLSL::DT_FLOAT;
  MLSL::OperationRegInfo *myRegInfo;
  MLSL::Session *s = eptr_->get_session();
  myRegInfo = s->CreateOperationRegInfo(MLSL::OT_CC);
  myRegInfo->SetName(nname_.c_str());
  myRegInfo->AddParameterSet(gparams_.nInput*gparams_.nOutput/gparams_.group, gparams_.kw*gparams_.kh, dt, false);
  myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);
  myRegInfo->AddParameterSet(gparams_.nOutput, 1, dt, false);

  myRegInfo->Validate();
  size_t opIdx = s->AddOperation(myRegInfo, e->get_distribution());
  this->op_ = s->GetOperation(opIdx);
  s->DeleteOperationRegInfo(myRegInfo);
#endif

  configure(p->get_compute_engine());
}

void FusedConvBNNode::configure(int engine)
{
  switch(engine)
  {
    case XSMM:
      impl = new FusedConvBNXSMM(&gparams_, engine);
  }
}

void FusedConvBNNode::fillWeightBuffers(TensorBuf* tBuf, int buftype, long long int size)
{
  int dtype = DT_FLOAT;
  void *ptr = tBuf->getBuffer();

#ifdef USE_MLSL
    unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
#else
    unsigned int node_id = 0;
#endif

  if(buftype == DATA)
  {
    int n = gparams_.batch_size;
    int ic = gparams_.nInput;
    int oc = gparams_.nOutput;
    int kh = gparams_.kh;
    int kw = gparams_.kw;
    int g = gparams_.group;
    int fanin = (ic * kh * kw)/g;
    int fanout = (oc * kh * kw)/g;
    int welem = ic * oc * kh * kw;

    initBuffer(ptr, dtype, variance_norm_, fanin, fanout, welem*sizeof(float), wfiller_type_, (unsigned int)(node_id+PRIME_SEED), std_);

#ifdef USE_MLSL
    if(dtype == DT_FLOAT)
      MPI_Bcast(ptr, welem, MPI_FLOAT, 0, MPI_COMM_WORLD);
#endif

    int in_dtype = tenBotData_[0]->getDataType();
    int out_dtype = tenTopData_[0]->getDataType();
  }
  else
    memset(ptr, 0, size);
}

void FusedConvBNNode::fillWeightMultipliers(float* lr, float* decay, long long int size)
{
  for(int i=0; i < size; i++)
  {
    lr[i] = lr_mult_[0];
    decay[i] = decay_mult_[0];
  }
}

void FusedConvBNNode::fillBiasMultipliers(float* lr, float* decay, long long int size)
{
  for(int i=0; i < size; i++)
  {
    lr[i] = lr_mult_[1];
    decay[i] = decay_mult_[1];
  }
}

void FusedConvBNNode::fillBuffer(TensorBuf* tBuf, int buftype, long long int size)
{
  int ttype = tBuf->getTensor()->getType();
  int dtype = DT_FLOAT;
  void *ptr = tBuf->getBuffer();

  float value;
  if(ttype==BNORMSCALE && buftype == DATA)
  {
    if(nname_.find("bn3") == nname_.npos)
      value = 1;
    else
      value = 0.;
  }
  else
    value = 0.;

  initConstantBuffer(ptr, dtype, size, "CONSTANT", value);
}

void FusedConvBNNode::Checkpoint(TensorBuf *tBuf, string name, string format)
{
  long long int bytes = tBuf->getBufferSize();
  int dtype = tBuf->getDataType();

  FILE* f;
  void* ptr;
  size_t pos;

  while((pos = name.find("/", 10)) != name.npos)
    name.replace(pos, 1, 1, '_');

  float* p = (float*)tBuf->getBuffer();
  bool no_checkpt = false;
  for(int i=0; i<16; i++)
  {
    if(isnan(p[i]) || isinf(p[i]))
    {
      no_checkpt = true;
      printf("Warning! %s Did not checkpoint! Weights are NaNs or Inf\n", nname_.c_str());
      break;
    }
  }

  if(!no_checkpt)
  {
    if(format.compare("binary") == 0)
    {
      f = fopen(name.c_str(), "wb");
      if(f != NULL)
      {
        if(name.find("wt") != name.npos)
        {
          ptr = _mm_malloc(bytes, 64);
          assert(ptr != NULL);
          impl->dumpBuffer(tBuf, ptr);
        }
        else
          ptr = tBuf->getBuffer();

        size_t b = fwrite(ptr, 1, bytes, f);
        assert((long long int)b == bytes);

        if(name.find("wt") != name.npos)
          _mm_free(ptr);
      }
      else
        printf("Warning: could not checkpoint to file %s\n",name.c_str());
    }
    else
    {
      f = fopen(name.c_str(), "w");
      if(f != NULL)
      {
        if(name.find("wt") != name.npos)
        {
          ptr = _mm_malloc(bytes, 64);
          assert(ptr != NULL);
          impl->dumpBuffer(tBuf, ptr);
        }
        else
          ptr = tBuf->getBuffer();

        for(int i=0; i<bytes/sizeof(float); i++)
          fprintf(f, "%f\n", *((float*)ptr + i));

        if(name.find("wt") != name.npos)
          _mm_free(ptr);
      }
      else
        printf("Warning: could not checkpoint to file %s\n",name.c_str());
    }
    if(f != NULL)
    {
      fflush(f);
      fclose(f);
    }
  }
}


void FusedConvBNNode::forwardPropagate()
{
  int nImg = gparams_.batch_size;
  int ifm = gparams_.nInput;
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifhp = ifh + 2*gparams_.ipad_h;
  int ifw = gparams_.iWidth;
  int ifwp = ifw + 2*gparams_.ipad_w;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;
  int ofhp = ofh + 2*gparams_.opad_h;
  int ofwp = ofw + 2*gparams_.opad_w;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

#ifndef NDEBUG
  // printf("Executing FP %s: input %p, weights %p, output %p\n",NNNode::nname_.c_str(), bot, wt, top);
  printf("Executing FP %s\n",NNNode::nname_.c_str());
  printf("Inputs: %d x %d x %d\n",ifm, ifh, ifw);
  printf("Outputs: %d x %d x %d\n",ofm, ofh, ofw);
  printf("Weights: %d x %d x %d x %d\n", ifm, ofm, kh, kw);
  printf("Bias: %d\n", ofm);

  if (gparams_.relu_fwd) printf("Fused relu\n");
#endif

  impl->set_top_compute_engine(top_compute_engine_);
  impl->set_bot_compute_engine(bot_cengine_);
  impl->set_node_name(nname_);
  impl->set_scratch_buffer(tenScratchData_);

  FusedConvBNNode *pnn = (FusedConvBNNode*)tenBot_[0]->getOwner();
  TensorBuf *pgammab = pnn->getScaleBuf();
  TensorBuf *pbetab = pnn->getShiftBuf();

  if(first_fp)
  {
    float* ptr = (float*)tenTopData_[0]->getBuffer();
    int size = tenTopData_[0]->getBufferSize()/sizeof(float);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<size; i++)
      ptr[i] = 0;

    float *gmean = (float*)tenMeanData_->getBuffer();
    float *grstd = (float*)tenRstdevData_->getBuffer();

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<ifm; i++)
    {
      gmean[i] = 0;
      grstd[i] = 0;
    }

    first_fp = false;
  }

#if 0
  float* ptr = (float*)tenTopData_->getBuffer();
  float* sptr = ptr + size;

  /* @TODO move this into LIBXSMM */
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<2*nImg*ofm; i++)
    sptr[i] = 0;
#endif

  impl->forwardPropagate(tenBotData_, tenWeightData_, pgammab, pbetab, tenScaleData_, tenShiftData_, tenMeanData_, tenRstdevData_, tenTopData_);

#ifdef CHECK_BLOWUP_FP32
  float *cbptr = (float*)tenTopData_[0]->getBuffer();
  for(int i=0; i<16; i++)
  {
    if(isnan(cbptr[i]) || isinf(cbptr[i]))
    {
      printf("Warning! %s layer FP activations are NaN or Inf\n", nname_.c_str());
      exit(-1);
    }
  }
#endif

#ifdef GETSTATS
#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id == 0)
#endif
  {
    float *ptr, *pptr, *p;

    if(eptr_->get_current_batch() % STATFREQ == 0)
    {
      string s = nname_ + "_Inp0";
      ptr = (float*)tenBotData_[0]->getBuffer();
      pptr = (float*)tenBotData_[0]->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, nImg*ifm*ifhp*ifwp);

      if(gparams_.eltwise)
      {
        string s = nname_ + "_Inp1";
        ptr = (float*)tenBotData_[1]->getBuffer();
        MeanOfLayer((char*)s.c_str(), ptr, nImg*ifm*ifhp*ifwp);
      }

      if(gparams_.bn_fwd || gparams_.bn_relu_fwd)
      {
        string s = nname_ + "_savedInp";
        ptr = (float*)tenBotData_[0]->getBuffer();
        p = ptr + 2*nImg*ifm + 2*ifm + nImg*ifm*ifhp*ifwp;
        MeanOfLayer((char*)s.c_str(), p, nImg*ifm*ifhp*ifwp);
      }

      s = nname_ + "_Wt";
      ptr = (float*)tenWeightData_->getBuffer();
      pptr = (float*)tenWeightData_->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, ifm*ofm*kh*kw);

      s = nname_ + "_Outp0";
      ptr = (float*)tenTopData_[0]->getBuffer();
      pptr = (float*)tenTopData_[0]->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, nImg*ofm*ofhp*ofwp);

      if(gparams_.split)
      {
        string s = nname_ + "_Outp1";
        ptr = (float*)tenTopData_[1]->getBuffer();
        MeanOfLayer((char*)s.c_str(), ptr, nImg*ifm*ifhp*ifwp);
      }

      s = nname_ + "_sump";
      int offset = nImg*ofm*ofhp*ofwp;
      p = ptr + offset;
      MeanOfLayer((char*)s.c_str(), p, nImg*ofm);

      s = nname_ + "_sum2p";
      MeanOfLayer((char*)s.c_str(), p+nImg*ofm, nImg*ofm);

      if(gparams_.bn_fwd || gparams_.bn_relu_fwd)
      {
        s = nname_ + "_expect";
        ptr = (float*)tenBotData_[0]->getBuffer();
        p = ptr + nImg*ifm*ifhp*ifwp + 2*nImg*ifm;
        MeanOfLayer((char*)s.c_str(), p, ifm);

        s = nname_ + "_rstdev";
        p = ptr + nImg*ifm*ifhp*ifwp + 2*nImg*ifm + ifm;
        MeanOfLayer((char*)s.c_str(), p, ifm);

        s = nname_ + "_gexpect";
        ptr = (float*)tenMeanData_->getBuffer();
        MeanOfLayer((char*)s.c_str(), ptr, ifm);

        s = nname_ + "_grstdev";
        ptr = (float*)tenRstdevData_->getBuffer();
        MeanOfLayer((char*)s.c_str(), ptr, ifm);

        s = nname_ + "_prev_gammap";
        float* gamma = (float*)pgammab->getBuffer();
        MeanOfLayer((char*)s.c_str(), gamma, ifm);

        s = nname_ + "_prev_betap";
        float* beta = (float*)pbetab->getBuffer();
        MeanOfLayer((char*)s.c_str(), beta, ifm);
      }
    }
  }
#endif
}

void FusedConvBNNode::backPropagate()
{

  int nImg = gparams_.batch_size;
  int ifm = gparams_.nInput;
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifhp = ifh + 2*gparams_.ipad_h;
  int ifw = gparams_.iWidth;
  int ifwp = ifw + 2*gparams_.ipad_w;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;
  int ofhp = ofh + 2*gparams_.opad_h;
  int ofwp = ofw + 2*gparams_.opad_w;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

#ifdef DEBUG
  printf("Executing BP %s\n",NNNode::nname_.c_str());
  printf("Grad Outputs: %d x %d x %d\n", ofm, ofh, ofw);
  printf("Grad Inputs: %d x %d x %d\n", ifm, ifh, ifw);
  printf("Weights: %d x %d x %d x %d\n", ofm, ifm, kh, kw);
#endif

  tenTopDiff_.resize(top_.size());
  for(int i=0; i<top_.size(); i++)
    tenTopDiff_[i] = tenTop_[i]->getBuf(DIFF);

  if(first_bp)
  {
    long long int size = nImg * ifm * ifhp *ifwp;

    float* ptr = (float*)tenBotDiff_[0]->getBuffer();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<size; i++)
      ptr[i] = 0;

    ptr = (float*)tenScaleDiff_->getBuffer();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<ofm; i++)
      ptr[i] = 0;

    ptr = (float*)tenShiftDiff_->getBuffer();
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<ofm; i++)
      ptr[i] = 0;

    first_bp = false;
  }

  impl->backPropagate(tenTopData_[0], tenTopDiff_, tenWeightData_, tenScaleData_, tenScaleDiff_, tenShiftDiff_, tenBotDiff_);

#ifdef CHECK_BLOWUP_FP32
  float* cbptr = (float*)tenTopDiff_[0]->getBuffer();
  for(int i=0; i<16; i++)
  {
    if(isnan(cbptr[i]) || isinf(cbptr[i]))
    {
      printf("Warning! %s layer BP activations are NaN or Inf\n", nname_.c_str());
      exit(-1);
    }
  }
#endif

#ifdef GETSTATS
  float *ptr, *pptr, *p, *bias;
#ifdef USE_MLSL
  unsigned int node_id_ = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id_ == 0)
#endif
  {
    if(eptr_->get_current_batch() % STATFREQ == 0)// && gparams_.ipad_h)
    {
      string s = nname_ + "_delOutp0";
      ptr = (float*)tenTopDiff_[0]->getBuffer();
      pptr = (float*)tenTopDiff_[0]->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, nImg*ofm*ofhp*ofwp);

      if(gparams_.split)
      {
        string s = nname_ + "_delOutp1";
        ptr = (float*)tenTopDiff_[1]->getBuffer();
        MeanOfLayer((char*)s.c_str(), ptr, nImg*ifm*ifhp*ifwp);
      }

      s = nname_ + "_Wt";
      ptr = (float*)tenWeightData_->getBuffer();
      pptr = (float*)tenWeightData_->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, ifm*ofm*kh*kw);

      if(gparams_.bstats_bwd || gparams_.bstats_relu_bwd)
      {
        s = nname_ + "_delgammap";
        p = (float*)tenScaleDiff_->getBuffer();
        MeanOfLayer((char*)s.c_str(), p, gparams_.nOutput);

        s = nname_ + "_delbetap";
        p = (float*)tenShiftDiff_->getBuffer();
        MeanOfLayer((char*)s.c_str(), p, gparams_.nOutput);

        s = nname_ + "_bmean2";
        ptr = (float*)tenBotData_[0]->getBuffer();
        p = ptr + nImg*ifm*ifhp*ifwp + 2*nImg*ifm;
        MeanOfLayer((char*)s.c_str(), p, ifm);

        s = nname_ + "_brstd2";
        p += ifm;
        MeanOfLayer((char*)s.c_str(), p, ifm);
      }

      if(gparams_.bn_bwd)
      {
        s = nname_ + "_bmean";
        ptr = (float*)tenTopData_[0]->getBuffer();
        p = ptr + nImg*ofm*ofhp*ofwp + 2*nImg*ofm;
        MeanOfLayer((char*)s.c_str(), p, ofm);

        s = nname_ + "_brstd";
        p += ofm;
        MeanOfLayer((char*)s.c_str(), p, ofm);
      }

      s = nname_ + "_delInp0";
      ptr = (float*)tenBotDiff_[0]->getBuffer();
      pptr = (float*)tenBotDiff_[0]->getPrivBuffer();
      p = (pptr == NULL) ? ptr : pptr;
      MeanOfLayer((char*)s.c_str(), p, nImg*ifm*ifhp*ifwp);

      if(gparams_.eltwise)
      {
        s = nname_ + "_delInp1";
        ptr = (float*)tenBotDiff_[1]->getBuffer();
        MeanOfLayer((char*)s.c_str(), ptr, nImg*ifm*ifhp*ifwp);
      }

      s = nname_ + "_savedOutp";
      ptr = (float*)tenTopData_[0]->getBuffer();
      p = ptr + nImg*ofm*ofhp*ofwp + 2*nImg*ofm + 2*ofm;
      MeanOfLayer((char*)s.c_str(), p, nImg*ofm*ofhp*ofwp);
    }
  }
#endif
}

void FusedConvBNNode::weightUpdate()
{
  int nImg = gparams_.batch_size;
  int ifm = gparams_.nInput;
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifw = gparams_.iWidth;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;
  int ofhp = ofh + 2*gparams_.opad_h;
  int ofwp = ofw + 2*gparams_.opad_w;
  int ifhp = ifh + 2*gparams_.ipad_h;
  int ifwp = ifw + 2*gparams_.ipad_w;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

#ifdef DEBUG
  // printf("Executing WU %s: grad_output %p, grad_weights %p, input %p\n",NNNode::nname_.c_str(), gtop, gwt, bot);
  printf("Executing WU %s\n",NNNode::nname_.c_str());
  printf("Grad Outputs: %d x %d x %d\n",ofm, ofh,ofw);
  printf("Inputs: %d x %d x %d\n",ifm, ifh, ifw);
  printf("del-Weights: %d x %d x %d x %d\n", ofm, ifm, kh, kw);
  printf("del-Biases: %d\n", ofm);
#endif

#ifdef GETSTATS
{
  float *ptr, *pptr, *p;

#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    string s = nname_ + "_delWt_Bef";
    ptr = (float*)tenWeightDiff_->getBuffer();
    pptr = (float*)tenWeightDiff_->getPrivBuffer();
    p = (pptr == NULL) ? ptr : pptr;
    MeanOfLayer((char*)s.c_str(), p, ifm*ofm*kh*kw);
  }
}
#endif

  tenTopDiff_[0] = tenTop_[0]->getBuf(DIFF);

  impl->weightUpdate(tenBotData_[0], tenTopDiff_[0], tenWeightDiff_);

#ifdef CHECK_BLOWUP_FP32
  float* cbptr = (float*)tenWeightDiff_->getBuffer();
  for(int i=0; i<16; i++)
  {
    if(isnan(cbptr[i]) || isinf(cbptr[i]))
    {
      printf("Warning! %s layer WU gradients are NaN or Inf\n", nname_.c_str());
      exit(-1);
    }
  }
#endif

#ifdef USE_MLSL
  void *mptr = tenWeightDiff_->getBuffer();
  void *mpptr = tenWeightDiff_->getPrivBuffer();
  void *mp = (mpptr == NULL) ? mptr : mpptr;

  op_->GetParameterSet(0)->StartGradientComm(mp);
  op_->GetParameterSet(1)->StartGradientComm(tenScaleDiff_->getBuffer());
  op_->GetParameterSet(2)->StartGradientComm(tenShiftDiff_->getBuffer());
#endif

#ifdef GETSTATS
  float *ptr, *pptr, *p;

#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    string s = nname_ + "_Inp";
    ptr = (float*)tenBotData_[0]->getBuffer();
    pptr = (float*)tenBotData_[0]->getPrivBuffer();
    p = (pptr == NULL) ? ptr : pptr;
    MeanOfLayer((char*)s.c_str(), p, nImg*ifm*ifhp*ifwp);

    s = nname_ + "_delOutp";

    ptr = (float*)tenTopDiff_[0]->getBuffer();
    pptr = (float*)tenTopDiff_[0]->getPrivBuffer();
    p = (pptr == NULL) ? ptr : pptr;
    MeanOfLayer((char*)s.c_str(), p, nImg*ofm*ofhp*ofwp);

    s = nname_ + "_delWt_Aft";
    ptr = (float*)tenWeightDiff_->getBuffer();
    pptr = (float*)tenWeightDiff_->getPrivBuffer();
    p = (pptr == NULL) ? ptr : pptr;
    MeanOfLayer((char*)s.c_str(), p, ifm*ofm*kh*kw);
  }
#endif
}

void FusedConvBNNode::solverStep()
{
#ifdef RETURNALL
  return;
#endif

  int nImg = gparams_.batch_size;
  int ifm = gparams_.nInput;
  int ofm = gparams_.nOutput;
  int ifh = gparams_.iHeight;
  int ifw = gparams_.iWidth;
  int ofh = gparams_.oHeight;
  int ofw = gparams_.oWidth;
  int kh = gparams_.kh;
  int kw = gparams_.kw;

  float *wt_prv_ptr = (float*)tenWeightData_->getPrivBuffer();
  float *wt_ptr = (float*)(tenWeightData_->getBuffer());

  float *gwt_prv_ptr = (float*)(tenWeightDiff_->getPrivBuffer());
  float *gwt_ptr = (float*)(tenWeightDiff_->getBuffer());

  float *wt = (wt_prv_ptr == NULL) ? wt_ptr : wt_prv_ptr;
  float *gwt = (gwt_prv_ptr == NULL) ? gwt_ptr : gwt_prv_ptr;

  float *iwt = (float*)(tenWeightInc_->getBuffer());

  float *gscale = (float*)tenScaleDiff_->getBuffer();
  float *gshift = (float*)tenShiftDiff_->getBuffer();

  int wsize = ifm*ofm*kh*kw;

#ifdef GETSTATS
#ifdef USE_MLSL
  unsigned int node_id = MLSL::Environment::GetEnv().GetProcessIdx();
  if(node_id == 0 && eptr_->get_current_batch() % STATFREQ == 0)
#else
  if(eptr_->get_current_batch() % STATFREQ == 0)
#endif
  {
    string s = nname_ + "_OldWt";
    MeanOfLayer((char*)s.c_str(), wt, ifm*ofm*kh*kw);
  }
#endif

  int num_nodes = 1;

#ifdef USE_MLSL
  void *mptr = op_->GetParameterSet(0)->WaitGradientComm();
  if(mptr != NULL && mptr != gwt)
    memcpy((void*)gwt, mptr, wsize*sizeof(float));

  mptr = op_->GetParameterSet(1)->WaitGradientComm();
  if(mptr != NULL && mptr != gscale)
      memcpy((void*)gscale, mptr, ofm*sizeof(float));

  mptr = op_->GetParameterSet(2)->WaitGradientComm();
  if(mptr != NULL && mptr != gshift)
      memcpy((void*)gshift, mptr, ofm*sizeof(float));

  num_nodes = MLSL::Environment::GetEnv().GetProcessCount();
#endif

#ifdef CHECK_BLOWUP_FP32
  float* ptr = (float*)tenWeightDiff_->getBuffer();
  for(int i=0; i<16; i++)
  {
    if(isnan(ptr[i]) || isinf(ptr[i]))
    {
      printf("Warning! %s layer Solver gradients are NaN or Inf\n", nname_.c_str());
      exit(-1);
    }
  }
#endif

  if(solver_->getGlobalFlag())
    return;
}

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
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/
#define IMG_LOOP_INIT 0
#define OFM_LOOP_INIT 1
#define OFM_LOOP_CLOSE 2
#define CONVOLUTION_KERNEL 3
#define IFM_LOOP_CLOSE_S 4
#define IFM_LOOP_FIRST_TOUCH 5
#define IMG_LOOP_CLOSE 6

#define LOCAL_ENTRIES_PER_CONV 3

#define MIXED 0
#define KHWC 1
#define HWKC 2
#define CHWK 3
#define HWCK 4

int IMG_INIT_BATCH_NORM = ((handle->fuse_ops & LIBXSMM_DNN_CONV_BN_FUSE_LEVEL_NAIVE) > 0) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_BN_FUSE_LEVEL_FOURPASS) > 0);
int IFM_INIT_BATCH_NORM = ((handle->fuse_ops & LIBXSMM_DNN_CONV_BN_FUSE_LEVEL_IFM) > 0);
int INTERLEAVED_BATCH_NORM = ((handle->fuse_ops & LIBXSMM_DNN_CONV_BN_FUSE_LEVEL_KERNEL) > 0);

/* Interleaved is only for 1x1 */
if(handle->desc.R != 1 || handle->desc.S != 1) 
{
  assert(INTERLEAVED_BATCH_NORM != 1);
}

const char * get_segment_type(int segment_type)
{
  if(segment_type == IMG_LOOP_INIT) return "IMG_LOOP_INIT";
  if(segment_type == OFM_LOOP_INIT) return "OFM_LOOP_INIT";
  if(segment_type == OFM_LOOP_CLOSE) return "OFM_LOOP_CLOSE";
  if(segment_type == CONVOLUTION_KERNEL) return "CONVOLUTION_KERNEL";
  if(segment_type == IFM_LOOP_CLOSE_S) return "IFM_LOOP_CLOSE_S";
  if(segment_type == IFM_LOOP_FIRST_TOUCH) return "IFM_LOOP_FIRST_TOUCH";
  if(segment_type == IMG_LOOP_CLOSE) return "IMG_LOOP_CLOSE";
  return "";
}

const char * get_loop_order(int loop_order)
{
  if(loop_order == MIXED) return "MIXED";
  if(loop_order == KHWC) return "KHWC";
  if(loop_order == HWKC) return "HWKC";
  if(loop_order == CHWK) return "CHWK";
  if(loop_order == HWCK) return "HWCK";
  return "";
}



typedef struct EXPANDED_SEGMENT {
  int loop_order;
  int segment_type;
  char kernel_variant;
  int ofmb;
  int ojb;
  int oj;
  int oi;
  int ofm1;
  int ifmb;
  int ifm1;
  int ci0;
  int ci1;
  int ci2;
  int ci3;
  int fwd_ofh_rb;
  int fwd_ofw_rb;
  int bwd_ofh_rb;
  int bwd_ofw_rb;
} expanded_segment_t;

#if !defined(_OPENMP)
int ltid;
#endif

int BLOCKSIFM_BLOCKING = handle->blocksifm_blocking;
int BLOCKSIFM = handle->blocksifm_lp;
int BLOCKSOFM = handle->blocksofm;
int loop_order = handle->loop_order;

#if defined(_OPENMP)
# pragma omp parallel num_threads(handle->desc.threads)
#else
for (ltid = 0; ltid < handle->desc.threads; ltid++)
#endif
{
#if defined(_OPENMP)
  int ltid = omp_get_thread_num();
#endif
  int img, ofm1, ifm1, oj, oi, ij, ii, local_entries = 0, ojb, ifmb, ofmb;
  int ii_use, ij_use, oi_use, oj_use;
  int padded_h = 0, padded_w = 0;

  /* Threading related variables */
  int imgpt = (handle->desc.N + handle->desc.threads - 1)/handle->desc.threads;
  int threads_per_image = handle->desc.threads / handle->desc.N;
  int my_img_start = LIBXSMM_MIN( ltid * imgpt, handle->desc.N);
  int my_img_end = LIBXSMM_MIN( (ltid+1) * imgpt, handle->desc.N);
  int my_ofm_start = 0;
  int my_ofm_end = BLOCKSOFM;
  int myOfmId;
  int nOfmBlocks;
  int total_calls;
  int n_code_segments;
  int mark_ofm_init, mark_ofm_close, mark_img_init, mark_img_close, mark_ifm_close, mark_ifm_init;
  expanded_segment_t * dbg_expanded_stream;
  int *tmp_expanded_stream, tmp_stream_index;
  segment_t *encoded_code_segments = NULL;
  int expanded_size;
  int stretch_of_convs;
  int encoded_stream_index;
  int lookahead_index;

  /* Arrays of stream indices */
  int *compute_indices, *bn_indices = 0;
  char *kernel_variant;

  if (handle->padding_flag == 1) {
    padded_h = handle->ifhp + 2 * handle->desc.pad_h;
    padded_w = handle->ifwp + 2 * handle->desc.pad_w;
  }

  n_code_segments = 0;
  tmp_stream_index = 0;

  if ( imgpt <= 1 ) {
    my_img_start = LIBXSMM_MIN( ltid / threads_per_image, handle->desc.N);
    my_img_end = LIBXSMM_MIN( my_img_start + 1, handle->desc.N);
    myOfmId = ltid % threads_per_image;
    nOfmBlocks = (BLOCKSOFM + threads_per_image -1) / threads_per_image;
    my_ofm_start = LIBXSMM_MIN(myOfmId * nOfmBlocks, BLOCKSOFM);
    my_ofm_end = LIBXSMM_MIN((myOfmId+1) * nOfmBlocks, BLOCKSOFM);
  }

  mark_ifm_init = (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_FWD) > 0) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_RELU_FWD) > 0) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_NORM_BWD) > 0) ) ? 1 : 0;
  mark_ofm_init =  ((((handle->options & LIBXSMM_DNN_CONV_OPTION_OVERWRITE) > 0) && (handle->use_nts_fwd == 0) ) || ( (handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BIAS) > 0) ) ? 1 : 0;
  mark_ofm_close = (((((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0)  ) && (handle->use_fwd_for_bwd == 0) && (handle->use_nts_fwd == 0) ) || 
                    ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_BWD) > 0) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_RELU_BWD)) || 
                    (( ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_BWD) > 0) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS_RELU_BWD) > 0) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_RELU_BWD) > 0) || ((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_MAX_STATS) > 0)) && (handle->use_fwd_for_bwd == 1) && (handle->use_nts_bwd == 0) ) ) ? 1 : 0;

  mark_ifm_close = 0;
  mark_img_init = ( (IMG_INIT_BATCH_NORM == 1) || (handle->padding_flag == 1) || (mark_ofm_close == 1) || (mark_ifm_close == 1) || (mark_ifm_init) ) ? 1 : 0;
  mark_img_close = (handle->padding_flag == 1) ? 1 : 0;

  /* Perform a dryrun to compute the memory requirements of the stream of indices */
  if (loop_order == MIXED) {
    if (handle->use_lp_kernel == 0) { /* Well, in this case leave loop as is...  */
      for (img = my_img_start; img < my_img_end; img++) {
        if (mark_img_init== 1) {
          n_code_segments++;
        }
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
          for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
            for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {
                  for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                    for (oi = 0; oi < handle->ofw ; oi += handle->fwd_ofw_rb) {
                      local_entries += LOCAL_ENTRIES_PER_CONV;

                      if (mark_ifm_init == 1) {
		                if((INTERLEAVED_BATCH_NORM == 1 && ofmb == my_ofm_start && ofm1 == ofmb) || 
						   (IFM_INIT_BATCH_NORM == 1 && ofmb == my_ofm_start && ojb == 0 && ofm1 == ofmb && oj == ojb && oi == 0))
			            {
                          n_code_segments++;
			            }
		              }

                      if (mark_ofm_init == 1) {
                        if (ifm1 == 0 && oj == 0 && oi == 0) {
                          n_code_segments++;
                        }
                      }

                      if (mark_ofm_close == 1) {
                        if (ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING  && oj >= handle->ofh - handle->fwd_ofh_rb && oi == handle->ofw - handle->fwd_ofw_rb) {
                          n_code_segments++;
                        }
                      }

                    }
                  }
                }
              }
            }
          }
        }
        if (mark_img_close== 1) {
          n_code_segments++;
        }
      }
    } else {
      for (img = my_img_start; img < my_img_end; img++) {
        if (mark_img_init== 1) {
          n_code_segments++;
        }
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
              for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                  for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
                    for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {
                      local_entries += LOCAL_ENTRIES_PER_CONV;
                      if (mark_ifm_init == 1) {
		                if ((INTERLEAVED_BATCH_NORM == 1 && ofmb == my_ofm_start && ofm1 == ofmb) || 
						    (IFM_INIT_BATCH_NORM == 1 && ofmb == my_ofm_start && ojb == 0 && ofm1 == ofmb && oj == ojb && oi == 0)) {
                          n_code_segments++;
			            }
                      }

                      if (mark_ofm_init == 1) {
                        if (ifm1 == 0 && oj == 0 && oi == 0) {
                          n_code_segments++;
                        }
                      }

                      if (mark_ofm_close == 1) {
                        if (ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING  && oj >= handle->ofh - handle->fwd_ofh_rb && oi == handle->ofw - handle->fwd_ofw_rb) {
                          n_code_segments++;
                        }
                      }


                      if (mark_ifm_close == 1) {
                        if ( ifm1 >= BLOCKSIFM-BLOCKSIFM_BLOCKING ) {
                          n_code_segments++;
                        }
                      }

                    }
                  }
                }
              }
            }
          }
        }
        if (mark_img_close== 1) {
          n_code_segments++;
        }
      }
    }
  }

  if (loop_order == HWKC) {
    for (img = my_img_start; img < my_img_end; img++) {
      if (mark_img_init== 1) {
        n_code_segments++;
      }
      for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
        for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
          for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
            for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
                for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
                  for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {

                    local_entries += LOCAL_ENTRIES_PER_CONV;

                    if (mark_ifm_init == 1) {
	                  if((INTERLEAVED_BATCH_NORM == 1 && ofmb == my_ofm_start && ofm1 == ofmb) || 
					     (IFM_INIT_BATCH_NORM == 1 && ofmb == my_ofm_start && ojb == 0 && oj == ojb && oi == 0 && ofm1 == ofmb))
		              {
                        n_code_segments++;
		              }
                    }
                    if (mark_ofm_init == 1) {
                      if (ifm1 == 0 && oj == 0 && oi == 0) {
                        n_code_segments++;
                      }
                    }

                    if (mark_ofm_close == 1) {
                      if (ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING  && oj >= handle->ofh - handle->fwd_ofh_rb && oi == handle->ofw - handle->fwd_ofw_rb) {
                        n_code_segments++;
                      }
                    }


                    if (mark_ifm_close == 1) {
                      if ( ifm1 >= BLOCKSIFM-BLOCKSIFM_BLOCKING ) {
                        n_code_segments++;
                      }
                    }
                  }

                }
              }
            }
          }
        }
      }
      if (mark_img_close== 1) {
        n_code_segments++;
      }
    }
  }


  handle->n_entries_fwd[ltid] = local_entries/LOCAL_ENTRIES_PER_CONV;

  /* Alocate auxiliary data structures for index jitting  */
  compute_indices = (int*) libxsmm_aligned_malloc( (local_entries+LOCAL_ENTRIES_PER_CONV) * sizeof(int), 64);
  handle->compute_fwd_indices_ptrs[ltid] = compute_indices;

  /* BN offsets...  */
  if  (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) && (handle->use_fwd_for_bwd == 0) && (handle->use_nts_fwd == 1) && 0 != local_entries) {
    bn_indices = (int*) libxsmm_aligned_malloc( (local_entries/3) * sizeof(int), 64);
    handle->bn_indices_ptrs[ltid] = bn_indices;
  }

  kernel_variant = (char*)(LOCAL_ENTRIES_PER_CONV <= local_entries ? libxsmm_aligned_malloc((local_entries / LOCAL_ENTRIES_PER_CONV) * sizeof(char), 64) : NULL);
  handle->kernel_fwd_variant_ptrs[ltid] = kernel_variant;
  handle->n_fwd_code_segments[ltid] = n_code_segments;
  expanded_size = local_entries/LOCAL_ENTRIES_PER_CONV + n_code_segments;
  tmp_expanded_stream = (int*)(0 < expanded_size ? malloc(expanded_size * sizeof(int)) : 0);
  dbg_expanded_stream = (expanded_segment_t*)(0 < expanded_size ? malloc(expanded_size * sizeof(expanded_segment_t)) : 0);
  memset(dbg_expanded_stream, 0, expanded_size*sizeof(expanded_segment_t));
  tmp_stream_index = 0;
  if (n_code_segments) {
    encoded_code_segments = (segment_t*) libxsmm_aligned_malloc(n_code_segments * sizeof(segment_t), 64);
    handle->fwd_code_segments[ltid] = encoded_code_segments;
  }
  local_entries = 0;

  /* Second run to compute actual indices */
  if (loop_order == MIXED) {
    if (handle->use_lp_kernel == 0) { /* Well, in this case leave loop as is...  */
      for (img = my_img_start; img < my_img_end; img++) {
        if (mark_img_init== 1 && 0 != tmp_expanded_stream && 0 != dbg_expanded_stream) {
          tmp_expanded_stream[tmp_stream_index] = IMG_LOOP_INIT;
          dbg_expanded_stream[tmp_stream_index].segment_type = IMG_LOOP_INIT;
          dbg_expanded_stream[tmp_stream_index].loop_order= MIXED;
          dbg_expanded_stream[tmp_stream_index].ofmb=0;
          dbg_expanded_stream[tmp_stream_index].ojb=0;
          dbg_expanded_stream[tmp_stream_index].oj=0;
          dbg_expanded_stream[tmp_stream_index].oi=0;
          dbg_expanded_stream[tmp_stream_index].ofm1=0;
          dbg_expanded_stream[tmp_stream_index].ifmb=0;
          dbg_expanded_stream[tmp_stream_index].ifm1=0;
          tmp_stream_index++;
        }
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
          for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
            for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
                for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {
                  for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                    for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {

                      if ( handle->use_fwd_for_bwd == 0 ) {
                        ij_use = oj * handle->desc.u;
                        ii_use = oi * handle->desc.v;
                        oi_use = oi;
                        oj_use = oj;
                      } else {
                        ij_use = oj;
                        ii_use = oi;
                        oi_use = oi * handle->desc.u;
                        oj_use = oj * handle->desc.v;
                      }

		      if(mark_ifm_init == 1)
		      {
		        if((INTERLEAVED_BATCH_NORM == 1 && ofmb == my_ofm_start && ofm1 == ofmb) || 
				   (IFM_INIT_BATCH_NORM == 1 && ofmb == my_ofm_start && ojb == 0 && ofm1 == ofmb && oj == ojb && oi == 0)) {
                          tmp_expanded_stream[tmp_stream_index] = IFM_LOOP_FIRST_TOUCH;
                          dbg_expanded_stream[tmp_stream_index].segment_type = IFM_LOOP_FIRST_TOUCH;
                          dbg_expanded_stream[tmp_stream_index].loop_order= MIXED;
                          dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                          dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                          dbg_expanded_stream[tmp_stream_index].oj=oj;
                          dbg_expanded_stream[tmp_stream_index].oi=oi;
                          dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                          dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                          dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                          tmp_stream_index++;
			}
		      }
                      if (0 != tmp_expanded_stream && mark_ofm_init == 1 && ifm1 == 0 && oj == 0 && oi == 0) {
                        tmp_expanded_stream[tmp_stream_index] = OFM_LOOP_INIT;
                        dbg_expanded_stream[tmp_stream_index].segment_type = OFM_LOOP_INIT;
                        dbg_expanded_stream[tmp_stream_index].loop_order= MIXED;
                        dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                        dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                        dbg_expanded_stream[tmp_stream_index].oj=oj;
                        dbg_expanded_stream[tmp_stream_index].oi=oi;
                        dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                        dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                        dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                        tmp_stream_index++;
                      }

                      if (handle->padding_flag == 1) {
                        compute_indices[local_entries] =  ( ( ( ifm1 *  padded_h  +  ij_use) * padded_w)  +  ii_use) *  handle->ifmblock * handle->fm_lp_block;
                      } else {
                        compute_indices[local_entries] =  ( ( ( ( ( (img *  BLOCKSIFM) +  ifm1) *  handle->ifhp )  +  ij_use) * handle->ifwp)  +  ii_use  ) *  handle->ifmblock * handle->fm_lp_block;
                      }
                      compute_indices[local_entries+1] = ( (ofm1 *  BLOCKSIFM )  +  ifm1 ) * handle->desc.R * handle->desc.S *  handle->ifmblock *  handle->ofmblock *  handle->fm_lp_block;
                      compute_indices[local_entries+2] = ( ( ( ( ( (img *  BLOCKSOFM) +  ofm1) *  handle->ofhp )  +  oj_use) * handle->ofwp)  +  oi_use) *  handle->ofmblock;

                      /* Initialize kernel variant with the one that prefetches everything */
                      if (handle->n_variants == 2) {
                        if (handle->h_variants) {
                          if (oj + handle->fwd_ofh_rb <= handle->ofh) {
                            kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] =  0;
                          } else {
                            kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] =  1;
                          }
                        } else {
                          if (oi + handle->fwd_ofw_rb <= handle->ofw) {
                            kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] =  0;
                          } else {
                            kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] = 1;
                          }
                        }
                      } else {
                        kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] =  0;
		      }

                      if (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) && (handle->use_fwd_for_bwd == 0) && (handle->use_nts_fwd == 1) ) {
                        bn_indices[local_entries/LOCAL_ENTRIES_PER_CONV] =  img * handle->ofmblock + ofm1 * handle->ofmblock * handle->desc.N;
                      }


                      if (0 != tmp_expanded_stream) {
                        tmp_expanded_stream[tmp_stream_index] = CONVOLUTION_KERNEL;
                        dbg_expanded_stream[tmp_stream_index].segment_type = CONVOLUTION_KERNEL;
                        dbg_expanded_stream[tmp_stream_index].loop_order= MIXED;
                        dbg_expanded_stream[tmp_stream_index].kernel_variant=kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV];
                        dbg_expanded_stream[tmp_stream_index].ci0 = compute_indices[local_entries + 0];
                        dbg_expanded_stream[tmp_stream_index].ci1 = compute_indices[local_entries + 1];
                        dbg_expanded_stream[tmp_stream_index].ci2 = compute_indices[local_entries + 2];
			dbg_expanded_stream[tmp_stream_index].fwd_ofh_rb = handle->fwd_ofh_rb;
			dbg_expanded_stream[tmp_stream_index].fwd_ofw_rb = handle->fwd_ofw_rb;
			dbg_expanded_stream[tmp_stream_index].bwd_ofh_rb = handle->bwd_ofh_rb;
			dbg_expanded_stream[tmp_stream_index].bwd_ofw_rb = handle->bwd_ofw_rb;
                        dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                        dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                        dbg_expanded_stream[tmp_stream_index].oj=oj;
                        dbg_expanded_stream[tmp_stream_index].oi=oi;
                        dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                        dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                        dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                        tmp_stream_index++;
	              }
                      local_entries += LOCAL_ENTRIES_PER_CONV;
                      if (0 != tmp_expanded_stream) {
                        if (mark_ofm_close == 1 && ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING && oj >= (handle->ofh - handle->fwd_ofh_rb) && oi == (handle->ofw - handle->fwd_ofw_rb)) {
                          tmp_expanded_stream[tmp_stream_index] = OFM_LOOP_CLOSE;
                          dbg_expanded_stream[tmp_stream_index].segment_type = OFM_LOOP_CLOSE;
                          dbg_expanded_stream[tmp_stream_index].loop_order= MIXED;
                          dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                          dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                          dbg_expanded_stream[tmp_stream_index].oj=oj;
                          dbg_expanded_stream[tmp_stream_index].oi=oi;
                          dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                          dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                          dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                          tmp_stream_index++;
                        }
                      }

                    }
                  }
                }
              }
            }
          }
        }
        if (mark_img_close== 1) {
          tmp_expanded_stream[tmp_stream_index] = IMG_LOOP_CLOSE;
          tmp_stream_index++;
        }
      }
    } else { /* Bring in all ifms to introduce IFM close tag  */
      for (img = my_img_start; img < my_img_end; img++) {
        if (0 != tmp_expanded_stream && mark_img_init== 1) {
          tmp_expanded_stream[tmp_stream_index] = IMG_LOOP_INIT;
          dbg_expanded_stream[tmp_stream_index].segment_type = IMG_LOOP_INIT;
          dbg_expanded_stream[tmp_stream_index].loop_order= MIXED;
          dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
          dbg_expanded_stream[tmp_stream_index].ojb=ojb;
          dbg_expanded_stream[tmp_stream_index].oj=oj;
          dbg_expanded_stream[tmp_stream_index].oi=oi;
          dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
          dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
          dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
          tmp_stream_index++;
        }
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
            for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
              for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                  for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
                    for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {

                      if ( handle->use_fwd_for_bwd == 0 ) {
                        ij_use = oj * handle->desc.u;
                        ii_use = oi * handle->desc.v;
                        oi_use = oi;
                        oj_use = oj;
                      } else {
                        ij_use = oj;
                        ii_use = oi;
                        oi_use = oi * handle->desc.u;
                        oj_use = oj * handle->desc.v;
                      }

		      if(mark_ifm_init == 1)
		      {
		        if((INTERLEAVED_BATCH_NORM == 1 && ofmb == my_ofm_start && ofm1 == ofmb) || 
				   (IFM_INIT_BATCH_NORM == 1 && ofmb == my_ofm_start && ojb == 0 && ofm1 == ofmb && oj == ojb && oi == 0)) {
                          tmp_expanded_stream[tmp_stream_index] = IFM_LOOP_FIRST_TOUCH;
                          dbg_expanded_stream[tmp_stream_index].segment_type = IFM_LOOP_FIRST_TOUCH;
                          dbg_expanded_stream[tmp_stream_index].loop_order= MIXED;
                          dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                          dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                          dbg_expanded_stream[tmp_stream_index].oj=oj;
                          dbg_expanded_stream[tmp_stream_index].oi=oi;
                          dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                          dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                          dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                          tmp_stream_index++;
			            }
		              }

                      if (0 != tmp_expanded_stream && mark_ofm_init == 1 && ifm1 == 0 && oj == 0 && oi == 0) {
                        tmp_expanded_stream[tmp_stream_index] = OFM_LOOP_INIT;
                        dbg_expanded_stream[tmp_stream_index].segment_type = OFM_LOOP_INIT;
                        dbg_expanded_stream[tmp_stream_index].loop_order= MIXED;
                        dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                        dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                        dbg_expanded_stream[tmp_stream_index].oj=oj;
                        dbg_expanded_stream[tmp_stream_index].oi=oi;
                        dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                        dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                        dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                        tmp_stream_index++;
                      }

                      if (handle->padding_flag == 1) {
                        compute_indices[local_entries] =  ( ( ( ifm1 *  padded_h  +  ij_use) * padded_w)  +  ii_use) *  handle->ifmblock * handle->fm_lp_block;
                      } else {
                        compute_indices[local_entries] =  ( ( ( ( ( (img *  BLOCKSIFM) +  ifm1) *  handle->ifhp )  +  ij_use) * handle->ifwp)  +  ii_use  ) *  handle->ifmblock * handle->fm_lp_block;
                      }
                      compute_indices[local_entries+1] = ( (ofm1 *  BLOCKSIFM )  +  ifm1 ) * handle->desc.R * handle->desc.S *  handle->ifmblock *  handle->ofmblock *  handle->fm_lp_block;
                      compute_indices[local_entries+2] = ( ( ( ( ( (img *  BLOCKSOFM) +  ofm1) *  handle->ofhp )  +  oj_use) * handle->ofwp)  +  oi_use) *  handle->ofmblock;

                      /* Initialize kernel variant with the one that prefetches everything */
                      if (handle->n_variants == 2) {
                        if (handle->h_variants) {
                          if (oj + handle->fwd_ofh_rb <= handle->ofh) {
                            kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] =  0;
                          } else {
                            kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] =  1;
                          }
                        } else {
                          if (oi + handle->fwd_ofw_rb <= handle->ofw) {
                            kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] = 0;
                          } else {
                            kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] = 1;
                          }
                        }
                      } else {
                        kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] =  0;
		      }


                      if (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) && (handle->use_fwd_for_bwd == 0) && (handle->use_nts_fwd == 1) ) {
                        bn_indices[local_entries/LOCAL_ENTRIES_PER_CONV] =  img * handle->ofmblock + ofm1 * handle->ofmblock * handle->desc.N;
                      }


                      if (0 != tmp_expanded_stream) {
                        tmp_expanded_stream[tmp_stream_index] = CONVOLUTION_KERNEL;
                        dbg_expanded_stream[tmp_stream_index].segment_type = CONVOLUTION_KERNEL;
                        dbg_expanded_stream[tmp_stream_index].loop_order= MIXED;
                        dbg_expanded_stream[tmp_stream_index].kernel_variant=kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV];
                        dbg_expanded_stream[tmp_stream_index].ci0 = compute_indices[local_entries + 0];
                        dbg_expanded_stream[tmp_stream_index].ci1 = compute_indices[local_entries + 1];
                        dbg_expanded_stream[tmp_stream_index].ci2 = compute_indices[local_entries + 2];
	                dbg_expanded_stream[tmp_stream_index].fwd_ofh_rb = handle->fwd_ofh_rb;
	                dbg_expanded_stream[tmp_stream_index].fwd_ofw_rb = handle->fwd_ofw_rb;
	                dbg_expanded_stream[tmp_stream_index].bwd_ofh_rb = handle->bwd_ofh_rb;
	                dbg_expanded_stream[tmp_stream_index].bwd_ofw_rb = handle->bwd_ofw_rb;
                        dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                        dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                        dbg_expanded_stream[tmp_stream_index].oj=oj;
                        dbg_expanded_stream[tmp_stream_index].oi=oi;
                        dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                        dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                        dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                        tmp_stream_index++;
                        if (mark_ifm_close == 1 && ifm1 >= BLOCKSIFM-BLOCKSIFM_BLOCKING) {
                          tmp_expanded_stream[tmp_stream_index] = IFM_LOOP_CLOSE_S;
                          dbg_expanded_stream[tmp_stream_index].segment_type = IFM_LOOP_CLOSE_S;
                          dbg_expanded_stream[tmp_stream_index].loop_order= MIXED;
                          dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                          dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                          dbg_expanded_stream[tmp_stream_index].oj=oj;
                          dbg_expanded_stream[tmp_stream_index].oi=oi;
                          dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                          dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                          dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                          tmp_stream_index++;
                        }
                        if (mark_ofm_close == 1 && ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING && oj >= (handle->ofh - handle->fwd_ofh_rb) && oi == (handle->ofw - handle->fwd_ofw_rb)) {
                          tmp_expanded_stream[tmp_stream_index] = OFM_LOOP_CLOSE;
                          dbg_expanded_stream[tmp_stream_index].segment_type = OFM_LOOP_CLOSE;
                          dbg_expanded_stream[tmp_stream_index].loop_order= MIXED;
                          dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                          dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                          dbg_expanded_stream[tmp_stream_index].oj=oj;
                          dbg_expanded_stream[tmp_stream_index].oi=oi;
                          dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                          dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                          dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                          tmp_stream_index++;
                        }
                      }
                      local_entries += LOCAL_ENTRIES_PER_CONV;
                    }
                  }
                }
              }
            }
          }
        }
        if (mark_img_close== 1) {
          tmp_expanded_stream[tmp_stream_index] = IMG_LOOP_CLOSE;
          dbg_expanded_stream[tmp_stream_index].segment_type = IMG_LOOP_CLOSE;
          dbg_expanded_stream[tmp_stream_index].loop_order= MIXED;
          tmp_stream_index++;
        }
      }
    }
  }


  if (loop_order == HWKC) {
    for (img = my_img_start; img < my_img_end; img++) {
      if (0 != tmp_expanded_stream && mark_img_init== 1) {
        tmp_expanded_stream[tmp_stream_index] = IMG_LOOP_INIT;
        dbg_expanded_stream[tmp_stream_index].segment_type = IMG_LOOP_INIT;
        dbg_expanded_stream[tmp_stream_index].loop_order= HWKC;
        tmp_stream_index++;
      }

      for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
        for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
          for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
            for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
                for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
                  for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {

                    if ( handle->use_fwd_for_bwd == 0 ) {
                      ij_use = oj * handle->desc.u;
                      ii_use = oi * handle->desc.v;
                      oi_use = oi;
                      oj_use = oj;
                    } else {
                      ij_use = oj;
                      ii_use = oi;
                      oi_use = oi * handle->desc.u;
                      oj_use = oj * handle->desc.v;
                    }

		    if(mark_ifm_init == 1)
		    {
		      if((INTERLEAVED_BATCH_NORM == 1 && ofmb == my_ofm_start && ofm1 == ofmb) || 
			     (IFM_INIT_BATCH_NORM == 1 && ofmb == my_ofm_start && ojb == 0 && oj == ojb && oi == 0 && ofm1 == ofmb)) {
                        tmp_expanded_stream[tmp_stream_index] = IFM_LOOP_FIRST_TOUCH;
                        dbg_expanded_stream[tmp_stream_index].segment_type = IFM_LOOP_FIRST_TOUCH;
                        dbg_expanded_stream[tmp_stream_index].loop_order= HWKC;
                        dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                        dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                        dbg_expanded_stream[tmp_stream_index].oj=oj;
                        dbg_expanded_stream[tmp_stream_index].oi=oi;
                        dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                        dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                        dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                        tmp_stream_index++;
	              }
		    }
                    if (0 != tmp_expanded_stream && mark_ofm_init == 1 && ifm1 == 0 && oj == 0 && oi == 0) {
                      tmp_expanded_stream[tmp_stream_index] = OFM_LOOP_INIT;
                      dbg_expanded_stream[tmp_stream_index].segment_type = OFM_LOOP_INIT;
                      dbg_expanded_stream[tmp_stream_index].loop_order= HWKC;
                      dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                      dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                      dbg_expanded_stream[tmp_stream_index].oj=oj;
                      dbg_expanded_stream[tmp_stream_index].oi=oi;
                      dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                      dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                      dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                      tmp_stream_index++;
                    }

                    if (handle->padding_flag == 1) {
                      compute_indices[local_entries] =  ( ( ( ifm1 *  padded_h  +  ij_use) * padded_w)  +  ii_use) *  handle->ifmblock * handle->fm_lp_block;
                    } else {
                      compute_indices[local_entries] =  ( ( ( ( ( (img *  BLOCKSIFM) +  ifm1) *  handle->ifhp )  +  ij_use) * handle->ifwp)  +  ii_use  ) *  handle->ifmblock * handle->fm_lp_block;
                    }
                    compute_indices[local_entries+1] = ( (ofm1 *  BLOCKSIFM )  +  ifm1 ) * handle->desc.R * handle->desc.S *  handle->ifmblock *  handle->ofmblock *  handle->fm_lp_block;
                    compute_indices[local_entries+2] = ( ( ( ( ( (img *  BLOCKSOFM) +  ofm1) *  handle->ofhp )  +  oj_use) * handle->ofwp)  +  oi_use  ) *  handle->ofmblock;

                    /* Initialize kernel variant with the one that prefetches everything */
                    if (handle->n_variants == 2) {
                      if (handle->h_variants) {
                        if (oj + handle->fwd_ofh_rb <= handle->ofh) {
                          kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] = 0;
                        } else {
                          kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] = 1;
                        }
                      } else {
                        if (oi + handle->fwd_ofw_rb <= handle->ofw) {
                          kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] = 0;
                        } else {
                          kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] = 1;
                        }
                      }
                    } else {
                      kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV] = 0;
		    }

                    if (((handle->fuse_ops & LIBXSMM_DNN_CONV_FUSE_BATCH_STATS) > 0) && (handle->use_fwd_for_bwd == 0) && (handle->use_nts_fwd == 1) ) {
                      bn_indices[local_entries/LOCAL_ENTRIES_PER_CONV] = img * handle->ofmblock + ofm1 * handle->ofmblock * handle->desc.N;
                    }


                    if (0 != tmp_expanded_stream) {
                      tmp_expanded_stream[tmp_stream_index] = CONVOLUTION_KERNEL;
                      dbg_expanded_stream[tmp_stream_index].segment_type = CONVOLUTION_KERNEL;
                      dbg_expanded_stream[tmp_stream_index].loop_order= HWKC;
                      dbg_expanded_stream[tmp_stream_index].kernel_variant=kernel_variant[local_entries/LOCAL_ENTRIES_PER_CONV];
                      dbg_expanded_stream[tmp_stream_index].ci0 = compute_indices[local_entries + 0];
                      dbg_expanded_stream[tmp_stream_index].ci1 = compute_indices[local_entries + 1];
                      dbg_expanded_stream[tmp_stream_index].ci2 = compute_indices[local_entries + 2];
	              dbg_expanded_stream[tmp_stream_index].fwd_ofh_rb = handle->fwd_ofh_rb;
	              dbg_expanded_stream[tmp_stream_index].fwd_ofw_rb = handle->fwd_ofw_rb;
	              dbg_expanded_stream[tmp_stream_index].bwd_ofh_rb = handle->bwd_ofh_rb;
	              dbg_expanded_stream[tmp_stream_index].bwd_ofw_rb = handle->bwd_ofw_rb;
                      dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                      dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                      dbg_expanded_stream[tmp_stream_index].oj=oj;
                      dbg_expanded_stream[tmp_stream_index].oi=oi;
                      dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                      dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                      dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                      tmp_stream_index++;

                      if (mark_ifm_close == 1 && ifm1 >= BLOCKSIFM-BLOCKSIFM_BLOCKING) {
                        tmp_expanded_stream[tmp_stream_index] = IFM_LOOP_CLOSE_S;
                        dbg_expanded_stream[tmp_stream_index].segment_type = IFM_LOOP_CLOSE_S;
                        dbg_expanded_stream[tmp_stream_index].loop_order= HWKC;
                        dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                        dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                        dbg_expanded_stream[tmp_stream_index].oj=oj;
                        dbg_expanded_stream[tmp_stream_index].oi=oi;
                        dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                        dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                        dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                        tmp_stream_index++;
                      }

                      if (mark_ofm_close == 1 && ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING && oj >= handle->ofh - handle->fwd_ofh_rb && oi == handle->ofw - handle->fwd_ofw_rb) {
                        tmp_expanded_stream[tmp_stream_index] = OFM_LOOP_CLOSE;
                        dbg_expanded_stream[tmp_stream_index].segment_type = OFM_LOOP_CLOSE;
                        dbg_expanded_stream[tmp_stream_index].loop_order= HWKC;
                        dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
                        dbg_expanded_stream[tmp_stream_index].ojb=ojb;
                        dbg_expanded_stream[tmp_stream_index].oj=oj;
                        dbg_expanded_stream[tmp_stream_index].oi=oi;
                        dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
                        dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
                        dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
                        tmp_stream_index++;
                      }
                    }
                    local_entries += LOCAL_ENTRIES_PER_CONV;
                  }
                }
              }
            }
          }
        }
      }
      if (mark_img_close== 1) {
        tmp_expanded_stream[tmp_stream_index] = IMG_LOOP_CLOSE;
        dbg_expanded_stream[tmp_stream_index].segment_type = IMG_LOOP_CLOSE;
        dbg_expanded_stream[tmp_stream_index].loop_order= HWKC;
        dbg_expanded_stream[tmp_stream_index].ofmb=ofmb;
        dbg_expanded_stream[tmp_stream_index].ojb=ojb;
        dbg_expanded_stream[tmp_stream_index].oj=oj;
        dbg_expanded_stream[tmp_stream_index].oi=oi;
        dbg_expanded_stream[tmp_stream_index].ofm1=ofm1;
        dbg_expanded_stream[tmp_stream_index].ifmb=ifmb;
        dbg_expanded_stream[tmp_stream_index].ifm1=ifm1;
        tmp_stream_index++;
      }
    }
  }


  /* Process the expanded stream and encode the segments via run length encoding */
  if (n_code_segments) {
    stretch_of_convs = 0;
    encoded_stream_index = 0;
    tmp_stream_index = 0;
    lookahead_index = 1;

    if (0 != tmp_expanded_stream) {
      while ( lookahead_index < expanded_size ) {
        while  ( tmp_expanded_stream[lookahead_index] == CONVOLUTION_KERNEL) {
          stretch_of_convs++;
          lookahead_index++;
          if ( lookahead_index >= expanded_size ) break;
        }
        encoded_code_segments[encoded_stream_index].segment_type = tmp_expanded_stream[tmp_stream_index];
        encoded_code_segments[encoded_stream_index].n_convs = stretch_of_convs;
        encoded_stream_index++;
        stretch_of_convs = 0;
        tmp_stream_index = lookahead_index;
        lookahead_index++;
      }

      /* Check if we have not written last segment entry -- in this case the stream ends with an action point */
      if ( encoded_stream_index < n_code_segments ) {
        encoded_code_segments[encoded_stream_index].segment_type = tmp_expanded_stream[tmp_stream_index];
        encoded_code_segments[encoded_stream_index].n_convs = stretch_of_convs;
      }
    }

    /* Final pass over the segments to fill-in auxiliary indices...  */
    encoded_stream_index = 0;

    if (loop_order == MIXED) {
      if (handle->use_lp_kernel == 0) { /* Well, in this case leave loop as is...  */
        for (img = my_img_start; img < my_img_end; img++) {
          if (mark_img_init== 1) {
            encoded_code_segments[encoded_stream_index].aux_index = img;
			if(IMG_INIT_BATCH_NORM)
			{
              encoded_code_segments[encoded_stream_index].aux_index0 = 0;
              encoded_code_segments[encoded_stream_index].aux_index1 = 0;
              encoded_code_segments[encoded_stream_index].aux_index2 = handle->desc.H;
              encoded_code_segments[encoded_stream_index].aux_index3 = handle->desc.W;
			}
			else
			{
              encoded_code_segments[encoded_stream_index].aux_index0 = 0;
              encoded_code_segments[encoded_stream_index].aux_index1 = 0;
              encoded_code_segments[encoded_stream_index].aux_index2 = 0;
              encoded_code_segments[encoded_stream_index].aux_index3 = 0;
			}
            encoded_stream_index++;
          }
          for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
            for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
              for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
                for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
                  for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {
                    for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                      for (oi = 0; oi < handle->ofw ; oi += handle->fwd_ofw_rb) {

                        ij = oj * handle->desc.u;
                        ii = oi * handle->desc.v;

                        if (mark_ifm_init == 1) {
                            if(INTERLEAVED_BATCH_NORM == 1 && ofmb == my_ofm_start && ofm1 == ofmb) {
                              encoded_code_segments[encoded_stream_index].aux_index = ifm1;
                              encoded_code_segments[encoded_stream_index].aux_index0 = ij;
                              encoded_code_segments[encoded_stream_index].aux_index1 = ii;
							  if(oj + handle->fwd_ofh_rb <= handle->ofh) {
                                encoded_code_segments[encoded_stream_index].aux_index2 = ij + handle->fwd_ofh_rb * handle->desc.u;
                                encoded_code_segments[encoded_stream_index].aux_index3 = ii + handle->fwd_ofw_rb * handle->desc.v;
							  } else {
                                encoded_code_segments[encoded_stream_index].aux_index2 = ij + handle->fwd_ofh_rb_2 * handle->desc.u;
                                encoded_code_segments[encoded_stream_index].aux_index3 = ii + handle->fwd_ofw_rb_2 * handle->desc.v;
							  }
                              encoded_stream_index++; // Assume BLOCKSIFM 
                            } else if(IFM_INIT_BATCH_NORM == 1 && ofmb == my_ofm_start && ojb == 0 && ofm1 == ofmb && oj == ojb && oi == 0) {
                              encoded_code_segments[encoded_stream_index].aux_index = ifm1;
                              encoded_code_segments[encoded_stream_index].aux_index0 = 0;
                              encoded_code_segments[encoded_stream_index].aux_index1 = 0;
                              encoded_code_segments[encoded_stream_index].aux_index2 = handle->desc.H;
                              encoded_code_segments[encoded_stream_index].aux_index3 = handle->desc.W;
                              encoded_stream_index++; // Assume BLOCKSIFM 
                           }
		                }

                        if (mark_ofm_init == 1) {
                          if (ifm1 == 0 && oj == 0 && oi == 0) {
                            encoded_code_segments[encoded_stream_index].aux_index = ofm1;
                            encoded_stream_index++;
                          }
                        }

                        if (mark_ofm_close == 1) {
                          if (ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING && oj >= handle->ofh - handle->fwd_ofh_rb && oi == handle->ofw - handle->fwd_ofw_rb) {
                            encoded_code_segments[encoded_stream_index].aux_index = ofm1;
                            encoded_stream_index++;
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          if (mark_img_close== 1) {
            encoded_code_segments[encoded_stream_index].aux_index = img;
            encoded_stream_index++;
          }
        }
      } else {
        for (img = my_img_start; img < my_img_end; img++) {
          if (mark_img_init== 1) {
            encoded_code_segments[encoded_stream_index].aux_index = img;
			if(IMG_INIT_BATCH_NORM)
			{
              encoded_code_segments[encoded_stream_index].aux_index0 = 0;
              encoded_code_segments[encoded_stream_index].aux_index1 = 0;
              encoded_code_segments[encoded_stream_index].aux_index2 = handle->desc.H;
              encoded_code_segments[encoded_stream_index].aux_index3 = handle->desc.W;
			}
			else
			{
              encoded_code_segments[encoded_stream_index].aux_index0 = 0;
              encoded_code_segments[encoded_stream_index].aux_index1 = 0;
              encoded_code_segments[encoded_stream_index].aux_index2 = 0;
              encoded_code_segments[encoded_stream_index].aux_index3 = 0;
			}
            encoded_stream_index++;
          }
          for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
            for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
              for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
                for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
                  for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                    for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
                      for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {
                        ij = oj * handle->desc.u;
                        ii = oi * handle->desc.v;
                        if (mark_ifm_init == 1) {
                          if(INTERLEAVED_BATCH_NORM == 1 && ofmb == my_ofm_start && ofm1 == ofmb) {
                            encoded_code_segments[encoded_stream_index].aux_index = ifm1;
                            encoded_code_segments[encoded_stream_index].aux_index0 = ij;
                            encoded_code_segments[encoded_stream_index].aux_index1 = ii;
							if(oj + handle->fwd_ofh_rb <= handle->ofh) {
                              encoded_code_segments[encoded_stream_index].aux_index2 = ij + handle->fwd_ofh_rb * handle->desc.u;
                              encoded_code_segments[encoded_stream_index].aux_index3 = ii + handle->fwd_ofw_rb * handle->desc.v;
							} else {
                              encoded_code_segments[encoded_stream_index].aux_index2 = ij + handle->fwd_ofh_rb_2 * handle->desc.u;
                              encoded_code_segments[encoded_stream_index].aux_index3 = ii + handle->fwd_ofw_rb_2 * handle->desc.v;
							}
                            encoded_stream_index++;
                          } else if(IFM_INIT_BATCH_NORM ==1 && ofmb == my_ofm_start && ojb == 0 && ofm1 == ofmb && oj == ojb && oi == 0){
                            encoded_code_segments[encoded_stream_index].aux_index = ifm1;
                            encoded_code_segments[encoded_stream_index].aux_index0 = 0;
                            encoded_code_segments[encoded_stream_index].aux_index1 = 0;
                            encoded_code_segments[encoded_stream_index].aux_index2 = handle->desc.H;
                            encoded_code_segments[encoded_stream_index].aux_index3 = handle->desc.W;
                            encoded_stream_index++;
                          }
			            }

                        if (mark_ofm_init == 1) {
                          if (ifm1 == 0 && oj == 0 && oi == 0) {
                            encoded_code_segments[encoded_stream_index].aux_index = ofm1;
                            encoded_stream_index++;
                          }
                        }

                        if (mark_ofm_close == 1) {
                          if (ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING && oj >= handle->ofh - handle->fwd_ofh_rb && oi == handle->ofw - handle->fwd_ofw_rb) {
                            encoded_code_segments[encoded_stream_index].aux_index = ofm1;
                            encoded_stream_index++;
                          }
                        }

                        if (mark_ifm_close == 1) {
                          if ( ifm1 >= BLOCKSIFM-BLOCKSIFM_BLOCKING ) {
                            encoded_stream_index++;
                          }
                        }

                      }
                    }
                  }
                }
              }
            }
          }
          if (mark_img_close== 1) {
            encoded_code_segments[encoded_stream_index].aux_index = img;
            encoded_stream_index++;
          }
        }
      }
    }

    if (loop_order == HWKC) {
      for (img = my_img_start; img < my_img_end; img++) {
        if (mark_img_init== 1) {
          encoded_code_segments[encoded_stream_index].aux_index = img;
	      if(IMG_INIT_BATCH_NORM)
			{
              encoded_code_segments[encoded_stream_index].aux_index0 = 0;
              encoded_code_segments[encoded_stream_index].aux_index1 = 0;
              encoded_code_segments[encoded_stream_index].aux_index2 = handle->desc.H;
              encoded_code_segments[encoded_stream_index].aux_index3 = handle->desc.W;
			}
			else
			{
              encoded_code_segments[encoded_stream_index].aux_index0 = 0;
              encoded_code_segments[encoded_stream_index].aux_index1 = 0;
              encoded_code_segments[encoded_stream_index].aux_index2 = 0;
              encoded_code_segments[encoded_stream_index].aux_index3 = 0;
			}
          encoded_stream_index++;
        }
        for (ofmb = my_ofm_start; ofmb < my_ofm_end; ofmb += handle->block_fwd_ofm) {
          for (ojb = 0; ojb < handle->ofh; ojb += handle->block_fwd_oj) {
            for (oj = ojb; oj < LIBXSMM_MIN(ojb+handle->block_fwd_oj,handle->ofh); oj += handle->fwd_ofh_rb) {
              for (oi = 0; oi < handle->ofw; oi += handle->fwd_ofw_rb) {
                for (ofm1 = ofmb; ofm1 < LIBXSMM_MIN(ofmb+handle->block_fwd_ofm, my_ofm_end); ofm1++ ) {
                  for (ifmb = 0; ifmb < BLOCKSIFM; ifmb += handle->block_fwd_ifm) {
                    for (ifm1 = ifmb; ifm1 < LIBXSMM_MIN(ifmb+handle->block_fwd_ifm, BLOCKSIFM); ifm1 += BLOCKSIFM_BLOCKING) {
                      ij = oj * handle->desc.u;
                      ii = oi * handle->desc.v;
                      if (mark_ifm_init == 1) {
		                if((INTERLEAVED_BATCH_NORM == 1 && ofmb == my_ofm_start &&  ofm1 == ofmb))
						{
                          encoded_code_segments[encoded_stream_index].aux_index = ifm1;
                          encoded_code_segments[encoded_stream_index].aux_index0 = ij;
                          encoded_code_segments[encoded_stream_index].aux_index1 = ii;
					      if(oj + handle->fwd_ofh_rb <= handle->ofh) {
                            encoded_code_segments[encoded_stream_index].aux_index2 = ij + handle->fwd_ofh_rb * handle->desc.u;
                            encoded_code_segments[encoded_stream_index].aux_index3 = ii + handle->fwd_ofw_rb * handle->desc.v;
					   	  } else {
                            encoded_code_segments[encoded_stream_index].aux_index2 = ij + handle->fwd_ofh_rb_2 * handle->desc.u;
                            encoded_code_segments[encoded_stream_index].aux_index3 = ii + handle->fwd_ofw_rb_2 * handle->desc.v;
					      }	
                          encoded_stream_index++;
						}
						else if(IFM_INIT_BATCH_NORM == 1 && ofmb == my_ofm_start && ojb == 0 && oj == ojb && oi == 0 && ofm1 == ofmb) {
                          encoded_code_segments[encoded_stream_index].aux_index = ifm1;
                          encoded_code_segments[encoded_stream_index].aux_index0 = 0;
                          encoded_code_segments[encoded_stream_index].aux_index1 = 0;
                          encoded_code_segments[encoded_stream_index].aux_index2 = handle->desc.H;
                          encoded_code_segments[encoded_stream_index].aux_index3 = handle->desc.W;
                          encoded_stream_index++;
                        }
			          }

                      if (mark_ofm_init == 1) {
                        if (ifm1 == 0 && oj == 0 && oi == 0) {
                          encoded_code_segments[encoded_stream_index].aux_index = ofm1;
                          encoded_stream_index++;
                        }
                      }

                      if (mark_ofm_close == 1) {
                        if (ifm1 == BLOCKSIFM-BLOCKSIFM_BLOCKING && oj >= handle->ofh - handle->fwd_ofh_rb && oi == handle->ofw - handle->fwd_ofw_rb) {
                          encoded_code_segments[encoded_stream_index].aux_index = ofm1;
                          encoded_stream_index++;
                        }
                      }

                      if (mark_ifm_close == 1) {
                        if ( ifm1 >= BLOCKSIFM-BLOCKSIFM_BLOCKING ) {
                          encoded_stream_index++;
                        }

                      }
                    }
                  }
                }
              }
            }
          }
        }
        if (mark_img_close== 1) {
          encoded_code_segments[encoded_stream_index].aux_index = img;
          encoded_stream_index++;
        }
      }
    }
  }

  free(tmp_expanded_stream);

  /* At the end of stream do not prefetch garbage */
  compute_indices[local_entries] = 0;
  compute_indices[local_entries+1] = 0;
  compute_indices[local_entries+2] = 0;
  total_calls = local_entries/LOCAL_ENTRIES_PER_CONV;

}

#undef IMG_LOOP_INIT
#undef OFM_LOOP_INIT
#undef OFM_LOOP_CLOSE
#undef CONVOLUTION_KERNEL
#undef IFM_LOOP_CLOSE_S
#undef MIXED
#undef KHWC
#undef HWKC
#undef CHWK
#undef HWCK


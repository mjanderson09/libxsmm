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
/* Alexander Heinecke, Kunal Banerjee (Intel Corp.)
******************************************************************************/
#ifndef LIBXSMM_DNN_RNNCELL_H
#define LIBXSMM_DNN_RNNCELL_H

#include "libxsmm_macros.h"
#include "libxsmm_typedefs.h"
#include "libxsmm_dnn.h"

#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <stdlib.h>
#if !defined(NDEBUG)
# include <stdio.h>
#endif
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_rnncell {
  int N;
  int nThreads;
} libxsmm_dnn_rnncell;

LIBXSMM_EXTERN_C typedef struct LIBXSMM_RETARGETABLE libxsmm_dnn_rnncell_desc {
  int N;
  int nThreads;
} libxsmm_dnn_rnncell_desc;

LIBXSMM_API libxsmm_dnn_rnncell* libxsmm_dnn_create_rnncell(libxsmm_dnn_rnncell_desc rnncell_desc, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_destroy_rnncell(const libxsmm_dnn_rnncell* handle);

LIBXSMM_API libxsmm_dnn_tensor_datalayout* libxsmm_dnn_rnncell_create_tensor_datalayout(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);

LIBXSMM_API size_t libxsmm_dnn_rnncell_get_scratch_size(const libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_scratch(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind, const void* scratch);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_scratch(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_compute_kind kind);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_bind_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor* tensor, const libxsmm_dnn_tensor_type type);
LIBXSMM_API libxsmm_dnn_tensor* libxsmm_dnn_rnncell_get_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type, libxsmm_dnn_err_t* status);
LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_release_tensor(libxsmm_dnn_rnncell* handle, const libxsmm_dnn_tensor_type type);

LIBXSMM_API libxsmm_dnn_err_t libxsmm_dnn_rnncell_execute_st(libxsmm_dnn_rnncell* handle, libxsmm_dnn_compute_kind kind,
  /*unsigned*/int start_thread, /*unsigned*/int tid);

#endif /*LIBXSMM_DNN_RNNCELL_H*/


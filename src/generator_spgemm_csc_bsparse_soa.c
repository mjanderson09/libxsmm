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
/* Alexander Heinecke (Intel Corp.)
******************************************************************************/
#include "generator_spgemm_csc_bsparse_soa.h"
#include "generator_gemm_common.h"
#include "generator_x86_instructions.h"
#include "generator_common.h"
#include "libxsmm_main.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csc_bsparse_soa( libxsmm_generated_code*         io_generated_code,
                                               const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                               const char*                     i_arch,
                                               const unsigned int*             i_row_idx,
                                               const unsigned int*             i_column_idx,
                                               const void*                     i_values ) {
  if ( strcmp(i_arch, "knl") == 0 ||
       strcmp(i_arch, "knm") == 0 ||
       strcmp(i_arch, "skx") == 0 ||
       strcmp(i_arch, "hsw") == 0 ||
       strcmp(i_arch, "snb") == 0 ) {
    libxsmm_generator_spgemm_csc_bsparse_soa_avx256_512( io_generated_code,
                                                         i_xgemm_desc,
                                                         i_arch,
                                                         i_row_idx,
                                                         i_column_idx,
                                                         i_values );
  } else {
    fprintf( stderr, "CSC + SOA is only available for AVX/AVX2/AVX512 at this point\n" );
    exit(-1);
  }
}

LIBXSMM_API_INTERN
void libxsmm_generator_spgemm_csc_bsparse_soa_avx256_512( libxsmm_generated_code*         io_generated_code,
                                                          const libxsmm_gemm_descriptor*  i_xgemm_desc,
                                                          const char*                     i_arch,
                                                          const unsigned int*             i_row_idx,
                                                          const unsigned int*             i_column_idx,
                                                          const void*                     i_values ) {
  unsigned int l_n = 0;
  unsigned int l_k = 0;
  unsigned int l_soa_width = 0;
  unsigned int l_max_cols = 0;
  unsigned int l_n_processed = 0;
  unsigned int l_n_limit = 0;
  unsigned int l_n_chunks = 0;
  unsigned int l_n_chunksize = 0;
  unsigned int l_found_mul = 0;
  unsigned int l_max_reg_block = 0;

  libxsmm_micro_kernel_config l_micro_kernel_config;
  libxsmm_loop_label_tracker l_loop_label_tracker;
  libxsmm_gp_reg_mapping l_gp_reg_mapping;

  LIBXSMM_UNUSED(i_values);

  /* select soa width */
  if ( LIBXSMM_GEMM_PRECISION_F64 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype )  ) {
    if ( strcmp(i_arch, "knl") == 0 ||
         strcmp(i_arch, "knm") == 0 ||
         strcmp(i_arch, "skx") == 0 ) {
      l_soa_width = 8;
      l_max_reg_block = 28;
    } else {
      l_soa_width = 4;
      l_max_reg_block = 14;
    }
  } else {
    if ( strcmp(i_arch, "knl") == 0 ||
         strcmp(i_arch, "knm") == 0 ||
         strcmp(i_arch, "skx") == 0 ) {
      l_soa_width = 16;
      l_max_reg_block = 28;
    } else {
      l_soa_width = 8;
      l_max_reg_block = 14;
    }
  }

  /* define gp register mapping */
  libxsmm_reset_x86_gp_reg_mapping( &l_gp_reg_mapping );
  /* matching calling convention on Linux */
#if defined(_WIN32) || defined(__CYGWIN__)
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_R8;
  /* TODO: full support for Windows calling convention */
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_RSI;
#else /* match calling convention on Linux */
  l_gp_reg_mapping.gp_reg_a = LIBXSMM_X86_GP_REG_RDI;
  l_gp_reg_mapping.gp_reg_b = LIBXSMM_X86_GP_REG_RSI;
  l_gp_reg_mapping.gp_reg_c = LIBXSMM_X86_GP_REG_RDX;
  l_gp_reg_mapping.gp_reg_a_prefetch = LIBXSMM_X86_GP_REG_RCX;
  l_gp_reg_mapping.gp_reg_b_prefetch = LIBXSMM_X86_GP_REG_R8;
#endif
  l_gp_reg_mapping.gp_reg_c_prefetch = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_mloop = LIBXSMM_X86_GP_REG_R12;
  l_gp_reg_mapping.gp_reg_nloop = LIBXSMM_X86_GP_REG_R13;
  l_gp_reg_mapping.gp_reg_kloop = LIBXSMM_X86_GP_REG_R14;
  l_gp_reg_mapping.gp_reg_help_0 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_1 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_2 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_3 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_4 = LIBXSMM_X86_GP_REG_UNDEF;
  l_gp_reg_mapping.gp_reg_help_5 = LIBXSMM_X86_GP_REG_UNDEF;

  /* define loop_label_tracker */
  libxsmm_reset_loop_label_tracker( &l_loop_label_tracker );

  /* define the micro kernel code gen properties */
  libxsmm_generator_gemm_init_micro_kernel_config_fullvector( &l_micro_kernel_config, i_xgemm_desc, i_arch, 0 );

  /* get max column in C */
  l_max_cols = i_xgemm_desc->n;
  for ( l_n = 0; l_n < i_xgemm_desc->n; l_n++ ) {
    if ( i_column_idx[l_n] == i_column_idx[i_xgemm_desc->n] ) {
      l_max_cols = l_n+1;
    }
  }

  /* calculate the chunk size of current columns to work on */
  l_n_chunks = ( (l_max_cols % l_max_reg_block) == 0 ) ? (l_max_cols / l_max_reg_block) : (l_max_cols / l_max_reg_block) + 1;
  assert(0 != l_n_chunks); /* mute static analysis (division-by-zero); such invalid input must be caught upfront */
  l_n_chunksize = ( (l_max_cols % l_n_chunks) == 0 ) ? (l_max_cols / l_n_chunks) : (l_max_cols / l_n_chunks) + 1;

  /* open asm */
  libxsmm_x86_instruction_open_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );

  /* m loop */
  libxsmm_x86_instruction_register_jump_label( io_generated_code, &l_loop_label_tracker );
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_mloop, 1 );

  /* loop over n-blocks */
  l_n_processed = 0;
  l_n_limit = l_n_chunksize;
  while ( l_n_processed < l_max_cols ) {
#if 0
    printf("l_max_cols: %i, l_n_processed: %i, l_n_limit: %i\n", l_max_cols, l_n_processed, l_n_limit);
#endif
    /* load C accumulator */
    for ( l_n = 0; l_n < l_n_limit - l_n_processed; l_n++ ) {
      if ( i_xgemm_desc->beta == 0 ) {
        libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                 l_micro_kernel_config.instruction_set,
                                                 l_micro_kernel_config.vxor_instruction,
                                                 l_micro_kernel_config.vector_name,
                                                 l_n, l_n, l_n );
      } else {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          l_micro_kernel_config.instruction_set,
                                          l_micro_kernel_config.c_vmove_instruction,
                                          l_gp_reg_mapping.gp_reg_c,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          (l_n_processed + l_n)*l_soa_width*l_micro_kernel_config.datatype_size,
                                          l_micro_kernel_config.vector_name,
                                          l_n, 0, 0 );
      }
    }

    /* do dense soa times sparse multiplication */
    for ( l_k = 0; l_k < (unsigned int)i_xgemm_desc->k; l_k++ ) {
      unsigned int l_found_qmadd = 0;
      unsigned int l_col_k = 0;
      unsigned int l_column_active[28];
      int l_nnz_idx[28][4];

      /* reset helpers */
      for ( l_n = 0; l_n < l_n_limit - l_n_processed; l_n++ ) {
        l_column_active[l_n] = 0;
        l_nnz_idx[l_n][0] = -1; l_nnz_idx[l_n][1] = -1; l_nnz_idx[l_n][2] = -1; l_nnz_idx[l_n][3] = -1;
      }
      l_found_mul = 0;

      /* let's figure out if we can apply qmadd when being sin F32 setting and on KNM */
      if ( (l_k < ((unsigned int)i_xgemm_desc->k - 3))                       &&
           (l_micro_kernel_config.instruction_set == LIBXSMM_X86_AVX512_KNM) &&
           (LIBXSMM_GEMM_PRECISION_F32 == LIBXSMM_GETENUM_INP( i_xgemm_desc->datatype ) )               ) {
        /* loop over the columns of B/C */
        for ( l_n = 0; l_n < l_n_limit - l_n_processed; l_n++ ) {
          unsigned int l_found = 0;
          unsigned int l_acol_k = 0;
          unsigned int l_col_elements = i_column_idx[l_n_processed+l_n+1] - i_column_idx[l_n_processed+l_n];
          unsigned int l_cur_column = i_column_idx[l_n_processed+l_n];

          for ( l_col_k = 0; l_col_k < l_col_elements; l_col_k++ ) {
            for ( l_acol_k = l_found; l_acol_k < 4 ; l_acol_k++ ) {
              if ( (l_k + l_acol_k) == i_row_idx[l_cur_column + l_col_k] ) {
                l_nnz_idx[l_n][l_acol_k] = l_cur_column + l_col_k;
                l_found = l_acol_k+1;
              }
              if (l_found == 4) {
                l_col_k = l_col_elements;
              }
            }
          }
          /* let's check if we can apply qmadd in col l_n */
          if ( (l_nnz_idx[l_n][0] != -1) && (l_nnz_idx[l_n][1] != -1) && (l_nnz_idx[l_n][2] != -1) && (l_nnz_idx[l_n][3] != -1) ) {
            l_column_active[l_n] = 2;
            l_found_qmadd = 1;
            l_found_mul = 1;
          } else {
            /* let's check if we have at least one entry in the column that matches one of the four entries */
            if ( (l_nnz_idx[l_n][0] != -1) || (l_nnz_idx[l_n][1] != -1) || (l_nnz_idx[l_n][2] != -1) || (l_nnz_idx[l_n][3] != -1) ) {
              l_column_active[l_n] = 1;
              l_found_mul = 1;
            } else {
              l_column_active[l_n] = 0;
            }
          }
        }
      }

      if ( l_found_qmadd == 0 ) {
        /* loop over the columns of B/C */
        for ( l_n = 0; l_n < l_n_limit - l_n_processed; l_n++ ) {
          unsigned int l_col_elements = i_column_idx[l_n_processed+l_n+1] - i_column_idx[l_n_processed+l_n];
          unsigned int l_cur_column = i_column_idx[l_n_processed+l_n];
          /* search for entries matching that k */
          for ( l_col_k = 0; l_col_k < l_col_elements; l_col_k++ ) {
            if ( l_k == i_row_idx[l_cur_column + l_col_k] ) {
              l_nnz_idx[l_n][0] = l_cur_column + l_col_k;
              l_col_k = l_col_elements;
            }
          }
          /* let's check if we have an entry in the column that matches the k from A */
          if ( (l_nnz_idx[l_n][0] != -1) ) {
            l_column_active[l_n] = 1;
            l_found_mul = 1;
          } else {
            l_column_active[l_n] = 0;
          }
        }
      }

      /* First case: we can use qmadd */
      if ( l_found_qmadd != 0 ) {
        unsigned int l_lcl_k = 0;
        for ( l_lcl_k = 0; l_lcl_k < 4; l_lcl_k++ ) {
          libxsmm_x86_instruction_vec_move( io_generated_code,
                                            l_micro_kernel_config.instruction_set,
                                            l_micro_kernel_config.a_vmove_instruction,
                                            l_gp_reg_mapping.gp_reg_a,
                                            LIBXSMM_X86_GP_REG_UNDEF, 0,
                                            (l_k+l_lcl_k)*l_soa_width*l_micro_kernel_config.datatype_size,
                                            l_micro_kernel_config.vector_name,
                                            l_max_reg_block+l_lcl_k, 0, 0 );
        }

        /* loop over the columns of B/C */
        for ( l_n = 0; l_n < l_n_limit - l_n_processed; l_n++ ) {
          /* issue a qmadd */
          if ( l_column_active[l_n] == 2 ) {
            libxsmm_x86_instruction_vec_compute_qfma( io_generated_code,
                                                      l_micro_kernel_config.instruction_set,
                                                      LIBXSMM_X86_INSTR_V4FMADDPS,
                                                      l_gp_reg_mapping.gp_reg_b,
                                                      LIBXSMM_X86_GP_REG_UNDEF,
                                                      0,
                                                      l_nnz_idx[l_n][0] * l_micro_kernel_config.datatype_size,
                                                      l_micro_kernel_config.vector_name,
                                                      l_max_reg_block,
                                                      l_n );
          } else if ( l_column_active[l_n] == 1 ) {
            for ( l_lcl_k = 0; l_lcl_k < 4; l_lcl_k++ ) {
              if ( l_nnz_idx[l_n][l_lcl_k] != -1 ) {
                libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                         l_micro_kernel_config.instruction_set,
                                                         l_micro_kernel_config.vmul_instruction,
                                                         1,
                                                         l_gp_reg_mapping.gp_reg_b,
                                                         LIBXSMM_X86_GP_REG_UNDEF,
                                                         0,
                                                         l_nnz_idx[l_n][l_lcl_k] * l_micro_kernel_config.datatype_size,
                                                         l_micro_kernel_config.vector_name,
                                                         l_max_reg_block+l_lcl_k,
                                                         l_n );
              }
            }
          }
        }
        /* increment by additional 3 columns */
        l_k += 3;
      } else if ( l_found_mul != 0 ) {
        libxsmm_x86_instruction_vec_move( io_generated_code,
                                          l_micro_kernel_config.instruction_set,
                                          l_micro_kernel_config.a_vmove_instruction,
                                          l_gp_reg_mapping.gp_reg_a,
                                          LIBXSMM_X86_GP_REG_UNDEF, 0,
                                          l_k*l_soa_width*l_micro_kernel_config.datatype_size,
                                          l_micro_kernel_config.vector_name,
                                          l_max_reg_block, 0, 0 );
        /* loop over the columns of B/C */
        for ( l_n = 0; l_n < l_n_limit - l_n_processed; l_n++ ) {
          if ( l_nnz_idx[l_n][0] != -1 ) {
            if ( strcmp(i_arch, "knl") == 0 ||
                 strcmp(i_arch, "knm") == 0 ||
                 strcmp(i_arch, "skx") == 0 ) {
              libxsmm_x86_instruction_vec_compute_mem( io_generated_code,
                                                       l_micro_kernel_config.instruction_set,
                                                       l_micro_kernel_config.vmul_instruction,
                                                       1,
                                                       l_gp_reg_mapping.gp_reg_b,
                                                       LIBXSMM_X86_GP_REG_UNDEF,
                                                       0,
                                                       l_nnz_idx[l_n][0] * l_micro_kernel_config.datatype_size,
                                                       l_micro_kernel_config.vector_name,
                                                       l_max_reg_block,
                                                       l_n );
            } else if ( strcmp(i_arch, "hsw") == 0 ) {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                                                l_micro_kernel_config.instruction_set,
                                                l_micro_kernel_config.b_vmove_instruction,
                                                l_gp_reg_mapping.gp_reg_b,
                                                LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                l_nnz_idx[l_n][0] * l_micro_kernel_config.datatype_size,
                                                l_micro_kernel_config.vector_name,
                                                15, 0, 0 );
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                       l_micro_kernel_config.instruction_set,
                                                       l_micro_kernel_config.vmul_instruction,
                                                       l_micro_kernel_config.vector_name,
                                                       l_max_reg_block,
                                                       15,
                                                       l_n );
            } else {
              libxsmm_x86_instruction_vec_move( io_generated_code,
                                                l_micro_kernel_config.instruction_set,
                                                l_micro_kernel_config.b_vmove_instruction,
                                                l_gp_reg_mapping.gp_reg_b,
                                                LIBXSMM_X86_GP_REG_UNDEF, 0,
                                                l_nnz_idx[l_n][0] * l_micro_kernel_config.datatype_size,
                                                l_micro_kernel_config.vector_name,
                                                15, 0, 0 );
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                       l_micro_kernel_config.instruction_set,
                                                       l_micro_kernel_config.vmul_instruction,
                                                       l_micro_kernel_config.vector_name,
                                                       l_max_reg_block,
                                                       15,
                                                       15 );
              libxsmm_x86_instruction_vec_compute_reg( io_generated_code,
                                                       l_micro_kernel_config.instruction_set,
                                                       l_micro_kernel_config.vadd_instruction,
                                                       l_micro_kernel_config.vector_name,
                                                       15,
                                                       l_n,
                                                       l_n );
            }
          }
        }
      } else {
        /* shouldn't happen */
      }
    }

    /* store C accumulator */
    for ( l_n = 0; l_n < l_n_limit - l_n_processed; l_n++ ) {
      libxsmm_x86_instruction_vec_move( io_generated_code,
                                        l_micro_kernel_config.instruction_set,
                                        l_micro_kernel_config.c_vmove_instruction,
                                        l_gp_reg_mapping.gp_reg_c,
                                        LIBXSMM_X86_GP_REG_UNDEF, 0,
                                        (l_n_processed + l_n)*l_soa_width*l_micro_kernel_config.datatype_size,
                                        l_micro_kernel_config.vector_name,
                                        l_n, 0, 1 );
    }

    /* adjust n progression */
    l_n_processed += l_n_chunksize;
    l_n_limit = LIBXSMM_MIN(l_n_processed + l_n_chunksize, l_max_cols);
  }

  /* advance C pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_c,
                                     l_micro_kernel_config.datatype_size*l_soa_width*i_xgemm_desc->ldc);

  /* advance A pointer */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_add_instruction, l_gp_reg_mapping.gp_reg_a,
                                   l_micro_kernel_config.datatype_size*l_soa_width*i_xgemm_desc->lda);

  /* close m loop */
  libxsmm_x86_instruction_alu_imm( io_generated_code, l_micro_kernel_config.alu_cmp_instruction, l_gp_reg_mapping.gp_reg_mloop, i_xgemm_desc->m );
  libxsmm_x86_instruction_jump_back_to_label( io_generated_code, l_micro_kernel_config.alu_jmp_instruction, &l_loop_label_tracker );

  /* close asm */
  libxsmm_x86_instruction_close_stream( io_generated_code, &l_gp_reg_mapping, i_arch, i_xgemm_desc->prefetch );
}


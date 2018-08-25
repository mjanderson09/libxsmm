#!/bin/bash

echo $OMP_NUM_THREADS
echo $KMP_AFFINITY
source compile.sh

echo "" > bench_defines.h
echo "#define USE_FUSE_LEVEL_IFM" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
CHECK=1 srun run_resnet50.sh 56 100 1 f32 F L 1   > nobn_ifm_check.txt

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_FOURPASS" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
CHECK=1 srun run_resnet50_no_first_layer.sh 56 100 1 f32 F L 1   > bn_fourpass_check.txt

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_NAIVE" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
CHECK=1 srun run_resnet50_no_first_layer.sh 56 100 1 f32 F L 1   > bn_naive_check.txt

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_IFM" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
CHECK=1 srun run_resnet50_no_first_layer.sh 56 100 1 f32 F L 1   > bn_ifm_check.txt

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_KERNEL" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
CHECK=1 srun run_resnet50_1x1.sh 56 100 1 f32 F L 1  > bn_kernel_check.txt

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_ELEMENTWISE_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_FOURPASS" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
CHECK=1 srun run_resnet50_no_first_layer.sh 56 100 1 f32 F L 1    > bn_eltwise_fourpass_check.txt

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_ELEMENTWISE_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_NAIVE" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
CHECK=1 srun run_resnet50_no_first_layer.sh 56 100 1 f32 F L 1   > bn_eltwise_naive_check.txt

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_ELEMENTWISE_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_IFM" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
CHECK=1 srun run_resnet50_no_first_layer.sh 56 100 1 f32 F L 1   > bn_eltwise_ifm_check.txt

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_ELEMENTWISE_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_KERNEL" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
CHECK=1 srun run_resnet50_1x1.sh 56 100 1 f32 F L 1  > bn_eltwise_kernel_check.txt



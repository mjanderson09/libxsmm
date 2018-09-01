#!/bin/bash

export OMP_NUM_THREADS=28
export KMP_AFFINITY=granularity=fine,compact,1,0
source compile.sh

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_STATS" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_IFM" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
mv layer_example_f32 layer_example_f32_bn_ifm

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_STATS" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_KERNEL" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
mv layer_example_f32 layer_example_f32_bn_kernel

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_STATS" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_NAIVE" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
mv layer_example_f32 layer_example_f32_bn_naive

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_FOURPASS" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
mv layer_example_f32 layer_example_f32_bn_fourpass

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_STATS" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_ELEMENTWISE_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_IFM" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
mv layer_example_f32 layer_example_f32_bne_ifm

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_STATS" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_ELEMENTWISE_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_KERNEL" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
mv layer_example_f32 layer_example_f32_bne_kernel

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_ELEMENTWISE_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_FOURPASS" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
mv layer_example_f32 layer_example_f32_bne_fourpass

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_STATS" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_FWD" >> bench_defines.h
echo "#define USE_ELEMENTWISE_FWD" >> bench_defines.h
echo "#define USE_FUSED_BATCH_NORM_RELU_FWD" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_NAIVE" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
mv layer_example_f32 layer_example_f32_bne_naive

echo "" > bench_defines.h
echo "#define USE_FUSED_BATCH_STATS" >> bench_defines.h
echo "#define USE_FUSE_LEVEL_IFM" >> bench_defines.h
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j
mv layer_example_f32 layer_example_f32_nobn



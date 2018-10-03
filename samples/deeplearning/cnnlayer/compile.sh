cd ../../../
make realclean && make AVX=3 MIC=0 OMP=1 STATIC=1 -j  
cd samples/deeplearning/cnnlayer

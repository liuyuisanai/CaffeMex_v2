# CaffeMex 
v2.2: 

A ___multi-GPU___ & ___memory-reduced___ ___MAT___-Caffe on ___LINUX and WINDOWS___

For now it ~~is an unstable version~~ has worked well in plenty of works.

Bug report: liuyu@sensetime.com or liuyuisanai@gmail.com
## What's New in Version 2.3?

1.Reduce 30% GPU memory usage

2.More stable for detection task (fix 0-count issue in a few layers)

3.Add center loss layer

## Installation for LINUX

1.cp Makefile.config.example Makefile.config and customize your own config.

2.make -j && make matcaffe
## Installation for WINDOWS

1.Find 'windows/' and you will know.
## Feature

1.Support both windows and linux platforms

2.Reduce 30% GPU memory usage

3.Lastest Matlab interface

4.Compatible with original caffe

## Run on cluster

1.Copy your datas and codes to cluster's shared disk (such as /mnt/lustre)

2.Write following shell script and save on your management node:

```
#!/usr/bin/env sh
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun --gres=gpu:4 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 matlab -nodisplay -r "run /FULL/PATH/TO/YOUR/MATLAB/SCRIPT.m"

```


3.Run the shell script.

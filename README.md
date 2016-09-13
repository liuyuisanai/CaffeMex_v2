# CaffeMex 
v2.1: 

A multiple-GPU version of MATLAB Caffe on LINUX and WINDOWS

For now it is an unstable version

Bug report: liuyu@sensetime.com
## Installation for LINUX

1.cp Makefile.config.example Makefile.config and customize your own config.

2.make -j && make matcaffe
## Installation for WINDOWS

1.Find 'windows/' and you will know.
## Feature

1.Support both windows and linux platforms

2.Support R-FCN

3.Lastest Matlab interface

4.Compatible with original caffe

## Run on cluster

1.Copy your datas and codes to cluster's shared disk (such as /mnt/lustre)

2.Write following shell script and save on your management node:

```
#!/usr/bin/env sh
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun --gres=gpu:4 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 matlab -nodisplay -r "run /FULL/PATH/TO/YOUR/MATLAB/SCRIPT.m"
```

3. run the shell script.
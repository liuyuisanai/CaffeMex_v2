# CaffeMex 
v2.3: 

A ___multi-GPU___ & ___memory-reduced___ ___MAT___-Caffe on ___LINUX and WINDOWS___

For now it ~~is an unstable version~~ has worked well in plenty of works.

Contributors & Bug report: 
[Yu Liu] liuyuisanai@gmail.com
[Hongyang Li] yangli@ee.cuhk.edu.hk
## What's New in Version 2.3?

1.Reduce 30% GPU memory usage

2.More stable for detection task (fix 0-count issue in a few layers)

3.Add center loss layer

## Installation for LINUX

1.cp Makefile.config.example Makefile.config and customize your own config.

2.make -j && make matcaffe
## Installation for WINDOWS

1.Find 'windows/' and you will know. By merging R-FCN https://github.com/daijifeng001/R-FCN
## Feature

1.Support both windows and linux platforms

2.Reduce 30% GPU memory usage (by merging Yuanjun Xiong's caffe http://yjxiong.me/)

3.Lastest Matlab interface

4.Compatible with original caffe

## Running on single machine with single/multiple GPU(s)

1. Installation this caffe

2. Interfaces:
Different interfaces between our caffe and origin's:

```Matlab
% reset all solvers and nets
Caffe.reset_all() 

% init a solver and set it handle in my_solver
my_solver = Caffe.get_solver('solver_proto_path, gpu_id')

% make a snapshot for my_solver
my_solver.snapshot('snapshot_path_and_name')

% restore a snapshot
my_solver.use_caffemodel('snapshot_path_and_name')

% set phase. It is useful if you use batch norm layer
my_solver.set_phase('train/test')

% set input data, format: cellA{cellB{matrix}}
% data{i}{j} means the j-th input on card i
my_solver.set_input_data(data)

% forward after set input data, usually used for test
my_solver.forward_prefilled()

% you can also replace the two above by this:
my_solver.forward(data)

% train one or more step after you set input data
my_solver.step(1) or step(N)

% get outputs, usually used for get losses
my_solver.get_output()

% get specified net handle:
my_solver.nets{k}
```
The other interfaces are same as origin caffe.

## Running on cluster

1.Copy your datas and codes to cluster's shared disk (such as /mnt/lustre)

2.Write following shell script and save on your management node:

```
#!/usr/bin/env sh
MV2_USE_CUDA=1 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 srun --gres=gpu:4 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 matlab -nodisplay -r "run /FULL/PATH/TO/YOUR/MATLAB/SCRIPT.m"

```


3.Run the shell script.

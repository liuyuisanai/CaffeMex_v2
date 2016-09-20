caffe.reset_all;
s = caffe.get_solver('test_model/solver.prototxt', 0);
input{1}{1} = rand(4,4,2,1,'single');
input{1}{2} = [0 0 0 1; 0 1 0 0; 0 0 0 1; 1 0 0 0];


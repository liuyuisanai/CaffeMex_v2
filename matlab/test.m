caffe.init_log('log/log')
t = caffe.get_solver('solver_80k110k_lr1_3.prototxt', 0:3);
t.step(1);
fprintf('done\n');

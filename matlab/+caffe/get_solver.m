function solver = get_solver(solver_file, gpu_ids)
% solver = get_solver(solver_file)
%   Construct a Solver object from solver_file

    CHECK(ischar(solver_file), 'solver_file must be a string');
    CHECK_FILE_EXIST(solver_file);
    CHECK(isnumeric(gpu_ids), 'gpu_ids should be an nonnegative array.')
    pSolver = caffe_('get_solver', solver_file, gpu_ids);
    solver = caffe.Solver(pSolver, gpu_ids);

end

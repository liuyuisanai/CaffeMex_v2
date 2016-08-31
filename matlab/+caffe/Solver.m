classdef Solver < handle
  % Wrapper class of caffe::SGDSolver in matlab
  
  properties (Access = private)
    hSolver_self
    attributes
    gpu_ids
    % attribute fields
    %     hNet_net
    %     hNet_test_nets
  end
  properties (SetAccess = private)
    nets
    test_nets
  end
  
  methods
    function self = Solver(varargin)
      % decide whether to construct a solver from solver_file or handle
      if ~(nargin == 2 && isstruct(varargin{1}))
        % construct a solver from solver_file
        self = caffe.get_solver(varargin{1}, varargin{2});
        return
      end
      % construct a solver from handle
      hSolver_solver = varargin{1};
      self.gpu_ids = varargin{2};
      CHECK(is_valid_handle(hSolver_solver), 'invalid Solver handle');
      
      % setup self handle and attributes
      self.hSolver_self = hSolver_solver;
      self.attributes = caffe_('solver_get_attr', self.hSolver_self);
      
      % setup net and test_nets
      for i = 1 : length(self.gpu_ids)
        self.nets{i} = caffe.Net(self.attributes{i}.hNet_net);
        self.test_nets = caffe.Net.empty();
        for n = 1:length(self.attributes{i}.hNet_test_nets)
          self.test_nets{i}(n) = caffe.Net(self.attributes{i}.hNet_test_nets(n));
        end
      end
      
    end
    function iter = iter(self)
      iter = caffe_('solver_get_iter', self.hSolver_self);
    end
    % add new interfaces for solver
    function use_caffemodel(self, model_dir)
      for t = 1 : length(self.gpu_ids)
          self.nets{t}.copy_from(model_dir);
      end
    end
    
    function set_input_data(self, inputs)
      CHECK(iscell(inputs), 'inputs must be a cell array');
      CHECK(length(inputs) == length(self.gpu_ids), ...
        'input data cell length must match gpu number');
      for n = 1 : length(self.gpu_ids)
        input_data = inputs{n};
        CHECK(iscell(input_data), 'input_data for each net must be a cell array');
        CHECK(length(input_data) == length(self.nets{n}.inputs), ...
        'input data cell length must match input blob number');
        for i = 1 : length(self.nets{n}.inputs)
            CHECK(isnumeric(input_data{i}), 'data must be numeric types');
            if ~isa(input_data{i}, 'single')
                inputs{n}{i} = single(input_data{i});
            end
        end
      end
      caffe_('solver_set_input', self.hSolver_self, inputs);
    end
    
    function output = get_output(self)
      output = cell(length(self.gpu_ids), 1);
      for t = 1 : length(self.gpu_ids)
          output{t} = self.nets{t}.get_output();
      end
    end
    function set_phase(self, phase)
      for t = 1 : length(self.gpu_ids)
          self.nets{t}.set_phase(phase);
      end
    end
    function reshape_as_input(self, inputs)
      CHECK(iscell(inputs), 'inputs must be a cell array');
      CHECK(length(inputs) == length(self.gpu_ids), ...
        'input data cell length must match gpu number');
      % reshape input blobs
      for n = 1 : length(self.gpu_ids)
        input_data = inputs{n};
        CHECK(iscell(input_data), 'input_data for each net must be a cell array');
        CHECK(length(input_data) == length(self.nets{n}.inputs), ...
        'input data cell length must match input blob number');
        for i = 1 : length(self.nets{n}.inputs)
            input_data_size = size(input_data{i});
            inputs{n}{i} = [input_data_size, ones(1, 4 - length(input_data_size))];
        end
      end
      caffe_('solver_reshape_input', self.hSolver_self, inputs);
    end
    function forward(self, input_data)
      self.set_input_data(input_data);
      caffe_('solver_test');
    end
    function forward_prefilled(self)
      caffe_('solver_test');
    end
    function snapshot(self, path)
        self.nets{1}.save(path);
    end
    function savestate(self, snapshot_filename)
        CHECK(ischar(snapshot_filename), 'snapshot_filename must be a string');
        caffe_('solver_snapshot', self.hSolver_self, snapshot_filename);
    end
    % add done
    
    function max_iter = max_iter(self)
      max_iter = caffe_('solver_get_max_iter', self.hSolver_self);
    end
    function restore(self, snapshot_filename)
      CHECK(ischar(snapshot_filename), 'snapshot_filename must be a string');
      CHECK_FILE_EXIST(snapshot_filename);
      caffe_('solver_restore', self.hSolver_self, snapshot_filename);
    end
    function solve(self)
      caffe_('solver_solve', self.hSolver_self);
    end
    function step(self, iters)
      CHECK(isscalar(iters) && iters > 0, 'iters must be positive integer');
      iters = double(iters);
      caffe_('solver_step', self.hSolver_self, iters);
    end
  end
end

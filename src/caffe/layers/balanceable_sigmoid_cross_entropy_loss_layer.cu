#include <vector>

#include "caffe/layers/balanceable_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void BSCE_scale_GPU(const int nthreads,
	Dtype* bottom_diff, const Dtype* stat){
	CUDA_KERNEL_LOOP(i, nthreads) {
		bottom_diff[ i ] = bottom_diff[ i ] * stat[ (int)floorf(fabs(bottom_diff[ i ]) * 10) ];
	}
}

template <typename Dtype>
void BSCELossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
	  if ( bottom[ 0 ]->count() < 1 || valid_num_ < 1 ){
		  return;
	  }
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
	caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
	BSCE_statistics(bottom[ 0 ]->mutable_cpu_diff(), count);
    // Scale down gradient
	const Dtype loss_weight = top[ 0 ]->cpu_diff()[ 0 ];
	BSCE_scale_GPU<Dtype> << <CAFFE_GET_BLOCKS(count),
		CAFFE_CUDA_NUM_THREADS >> >( count, bottom[ 0 ]->mutable_gpu_diff(), statistics_.gpu_data() );
	CUDA_POST_KERNEL_CHECK;
	caffe_gpu_scal(count, loss_weight / valid_num_, bottom[ 0 ]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(BSCELossLayer);


}  // namespace caffe

#include <vector>

#include "caffe/layers/balanceable_sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
__global__ void BSCE_scale_GPU(const int nthreads,
	Dtype* bottom_diff, const Dtype* his_stat_, const Dtype* cls_grad_, const Dtype* target, int bin_num_, bool b_cls_, bool b_his_){
	CUDA_KERNEL_LOOP(i, nthreads) {
		if ( b_his_ )
			bottom_diff[ i ] = bottom_diff[ i ] * his_stat_[ (int)floorf(fabs(bottom_diff[ i ]) * bin_num_) ];
		if ( b_cls_ )
			bottom_diff[ i ] = bottom_diff[ i ] * cls_grad_[ (int)target[ i ] ];   //nthreads / 2.0 / (cls_stat_[ (int)target[ i ] ]);
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
	BSCE_statistics(bottom[ 0 ]->mutable_cpu_diff(), count, bottom[ 1 ]->cpu_data());
    // Scale down gradient
	const Dtype loss_weight = top[ 0 ]->cpu_diff()[ 0 ];
	BSCE_scale_GPU<Dtype> << <CAFFE_GET_BLOCKS(count),
		CAFFE_CUDA_NUM_THREADS >> >( count, bottom[ 0 ]->mutable_gpu_diff(), his_stat_.gpu_data(), cls_grad_.gpu_data(), target, bin_num_, b_cls_, b_his_ );
	CUDA_POST_KERNEL_CHECK;
	caffe_gpu_scal(count, loss_weight / valid_num_, bottom[ 0 ]->mutable_gpu_diff());
  }
}

INSTANTIATE_LAYER_GPU_BACKWARD(BSCELossLayer);


}  // namespace caffe

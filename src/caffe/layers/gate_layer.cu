#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/gate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	
	template <typename Dtype>
	void GateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		top[ 0 ]->ShareData(*bottom[ 0 ]);
	}

	
	template <typename Dtype>
	void GateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		if ( !propagate_down[ 0 ] ) { return; }
		if ( state_ == active_val_ ){
			caffe_copy(top[ 0 ]->count(), top[ 0 ]->gpu_diff(), bottom[ 0 ]->mutable_gpu_diff());
		}
		else{
			caffe_gpu_set(bottom[ 0 ]->count(), static_cast<Dtype>( 0. ), bottom[ 0 ]->mutable_gpu_diff());
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(GateLayer);

}  // namespace caffe

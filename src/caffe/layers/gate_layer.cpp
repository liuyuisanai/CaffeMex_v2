#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gate_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void GateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		active_val_ = this->layer_param().gate_param().active_val();
	}

	template <typename Dtype>
	void GateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK(bottom[ 1 ]->count() == 1)
			<< "The second botton should be a scatter state value between [0, MaxInt].";
		top[ 0 ]->ReshapeLike(*bottom[ 0 ]);
		state_ = bottom[ 1 ]->cpu_data()[0];
	}

	template <typename Dtype>
	void GateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		//top[ 0 ]->ShareData(*bottom[ 0 ]);
		caffe_copy(top[ 0 ]->count(), bottom[ 0 ]->cpu_data(), top[ 0 ]->mutable_cpu_data());
		LOG(INFO) << "Using cpu code.";
	}

	template <typename Dtype>
	void GateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		if ( !propagate_down[ 0 ] ) { return; }
		if ( state_ == active_val_ ){
			caffe_copy(top[ 0 ]->count(), top[ 0 ]->cpu_diff(), bottom[0]->mutable_cpu_diff());
		}
		else {
			caffe_set<Dtype>(bottom[ 0 ]->count(), Dtype(0.), bottom[ 0 ]->mutable_cpu_diff());
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(GateLayer);
#endif

	INSTANTIATE_CLASS(GateLayer);
	REGISTER_LAYER_CLASS(Gate);

}  // namespace caffe

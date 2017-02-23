#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gaussian_sample_layer.hpp"

namespace caffe {

template <typename Dtype>
void GaussianSampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  	channels_ = bottom[0]->channels();
	number_ = this->layer_param().sample_param().number();
	decay_ = this->layer_param().sample_param().decay();
	scale_ = this->layer_param().sample_param().scale();

	// Check if we need to set up the weights
	if ( this->blobs_.size() > 0 ) {
		LOG(INFO) << "Skipping parameter initialization";
	}
	else {
		this->blobs_.resize(2);
		// history mean
		this->blobs_[ 0 ].reset(new Blob<Dtype>(1, channels_, 1, 1));
		caffe_set(channels_, Dtype(0), this->blobs_[ 0 ]->mutable_cpu_data());
		// history variance
		this->blobs_[ 1 ].reset(new Blob<Dtype>(1, channels_, 1, 1));
		caffe_set(channels_, Dtype(1), this->blobs_[ 1 ]->mutable_cpu_data());
	}  // parameter initialization
	this->param_propagate_down_.resize(this->blobs_.size(), false);
	this->param_propagate_down_[ 0 ] = false;
	this->param_propagate_down_[ 1 ] = false;
}

template <typename Dtype>
void GaussianSampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[ 0 ]->num();
  channels_ = bottom[0]->channels();
  int height_ = bottom[0]->height();
  int width_ = bottom[0]->width();
  CHECK(height_ == 1 && width_ == 1) << "Inputs must be N C 1 1.";
  top[0]->Reshape(number_, channels_, 1, 1);
  vector<int> mult_dims(1, bottom[ 0 ]->num());
  sum_multiplier_n_.Reshape(mult_dims);
  sum_multiplier_num_.Reshape(1, number_, 1, 1);
  batch_ex_.Reshape(1, channels_, 1, 1);
  batch_dx_.Reshape(1, channels_, 1, 1);
  caffe_set(sum_multiplier_n_.count(), Dtype(1), sum_multiplier_n_.mutable_cpu_data());
  caffe_set(sum_multiplier_num_.count(), Dtype(1), sum_multiplier_num_.mutable_cpu_data());
}

template <typename Dtype>
void GaussianSampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[ 0 ]->cpu_data();
// Step1. statistics
	// EX
	caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1.0 / num_), bottom_data,
		sum_multiplier_n_.cpu_data(), Dtype(0), batch_ex_.mutable_cpu_data());
	// DX
	// DX.1: prepare average matrix into buffer_blob_
	Blob<Dtype> buffer_blob_;
	buffer_blob_.ReshapeLike(*bottom[ 0 ]);
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(-1),
		sum_multiplier_n_.cpu_data(), batch_ex_.cpu_data(), Dtype(0),
		buffer_blob_.mutable_cpu_data());
	// DX.2: buffer_blob_ = X-EX
	caffe_add(buffer_blob_.count(), bottom_data, buffer_blob_.cpu_data(), buffer_blob_.mutable_cpu_data());
	// DX.3: buffer_blob_ = (X-EX)^2
	caffe_powx(buffer_blob_.count(), buffer_blob_.cpu_data(), Dtype(2), buffer_blob_.mutable_cpu_data());
	// DX.4: DX = E((X-EX)^2)
	caffe_cpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1. / num_), buffer_blob_.cpu_data(),
		sum_multiplier_n_.cpu_data(), Dtype(0), batch_dx_.mutable_cpu_data());
	// save history mean & var
	if ( this->phase_ == TRAIN ) {
		caffe_add_scalar(batch_dx_.count(), std::numeric_limits<Dtype>::epsilon(), batch_dx_.mutable_cpu_data());
		caffe_cpu_axpby<Dtype>(batch_ex_.count(), Dtype(1) - decay_, batch_ex_.cpu_data(), decay_,
			this->blobs_[ 0 ]->mutable_cpu_data());
		caffe_cpu_axpby<Dtype>(batch_dx_.count(), Dtype(1) - decay_, batch_dx_.cpu_data(), decay_,
			this->blobs_[ 1 ]->mutable_cpu_data());
		caffe_add_scalar(batch_dx_.count(), std::numeric_limits<Dtype>::epsilon(), batch_dx_.mutable_cpu_data());
	}
	//if ( this->phase_ == TEST ) {
		caffe_copy(batch_ex_.count(), this->blobs_[ 0 ]->cpu_data(), batch_ex_.mutable_cpu_data());
		caffe_copy(batch_dx_.count(), this->blobs_[ 1 ]->cpu_data(), batch_dx_.mutable_cpu_data());
	//}
	// DX -> std for generation
	caffe_powx(batch_dx_.count(), batch_dx_.cpu_data(), Dtype(0.5),
		batch_dx_.mutable_cpu_data());
// Step2. generate
	// sum_multiplier_num_
	caffe_rng_gaussian(top[ 0 ]->count(), Dtype(0), Dtype(1)*scale_, top[0]->mutable_cpu_data());
	buffer_blob_.ReshapeLike(*top[0]);
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, number_, channels_, 1, Dtype(-1),
		sum_multiplier_num_.cpu_data(), batch_dx_.cpu_data(), Dtype(0),
		buffer_blob_.mutable_cpu_data());
	caffe_mul(top[ 0 ]->count(), top[ 0 ]->cpu_data(), buffer_blob_.cpu_data(), top[ 0 ]->mutable_cpu_data());
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, number_, channels_, 1, Dtype(1),
		sum_multiplier_num_.cpu_data(), batch_ex_.cpu_data(), Dtype(0),
		buffer_blob_.mutable_cpu_data());
	caffe_add(top[ 0 ]->count(), top[ 0 ]->cpu_data(), buffer_blob_.cpu_data(), top[ 0 ]->mutable_cpu_data());
}

template <typename Dtype>
void GaussianSampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

#ifdef CPU_ONLY
STUB_GPU(GaussianSampleLayer);
#endif

INSTANTIATE_CLASS(GaussianSampleLayer);
REGISTER_LAYER_CLASS(GaussianSample);

}  // namespace caffe

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gaussian_sample_layer.hpp"

namespace caffe {

template <typename Dtype>
void GaussianSampleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[ 0 ]->gpu_data();
	// Step1. statistics
	// EX
	caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1.0 / num_), bottom_data,
		sum_multiplier_n_.gpu_data(), Dtype(0), batch_ex_.mutable_gpu_data());
	// DX
	// DX.1: prepare average matrix into buffer_blob_
	Blob<Dtype> buffer_blob_;
	buffer_blob_.ReshapeLike(*bottom[ 0 ]);
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_, channels_, 1, Dtype(-1),
		sum_multiplier_n_.gpu_data(), batch_ex_.gpu_data(), Dtype(0),
		buffer_blob_.mutable_gpu_data());
	// DX.2: buffer_blob_ = X-EX
	caffe_gpu_add(buffer_blob_.count(), bottom_data, buffer_blob_.gpu_data(), buffer_blob_.mutable_gpu_data());
	// DX.3: buffer_blob_ = (X-EX)^2
	caffe_gpu_powx(buffer_blob_.count(), buffer_blob_.gpu_data(), Dtype(2), buffer_blob_.mutable_gpu_data());
	// DX.4: DX = E((X-EX)^2)
	caffe_gpu_gemv<Dtype>(CblasTrans, num_, channels_, Dtype(1. / num_), buffer_blob_.gpu_data(),
		sum_multiplier_n_.gpu_data(), Dtype(0), batch_dx_.mutable_gpu_data());
	// save history mean & var
	if ( this->phase_ == TRAIN ) {
		caffe_gpu_add_scalar(batch_dx_.count(), std::numeric_limits<Dtype>::epsilon(), batch_dx_.mutable_gpu_data());
		caffe_gpu_axpby<Dtype>(batch_ex_.count(), Dtype(1) - decay_, batch_ex_.gpu_data(), decay_,
			this->blobs_[ 0 ]->mutable_gpu_data());
		caffe_gpu_axpby<Dtype>(batch_dx_.count(), Dtype(1) - decay_, batch_dx_.gpu_data(), decay_,
			this->blobs_[ 1 ]->mutable_gpu_data());
		caffe_gpu_add_scalar(batch_dx_.count(), std::numeric_limits<Dtype>::epsilon(), batch_dx_.mutable_gpu_data());
	}
	//if ( this->phase_ == TEST ) {
	caffe_copy(batch_ex_.count(), this->blobs_[ 0 ]->gpu_data(), batch_ex_.mutable_gpu_data());
	caffe_copy(batch_dx_.count(), this->blobs_[ 1 ]->gpu_data(), batch_dx_.mutable_gpu_data());
	//}
	// DX -> std for generation
	caffe_gpu_powx(batch_dx_.count(), batch_dx_.gpu_data(), Dtype(0.5),
		batch_dx_.mutable_gpu_data());
	// Step2. generate
	// sum_multiplier_num_
	caffe_gpu_rng_gaussian(top[ 0 ]->count(), Dtype(0), Dtype(1), top[ 0 ]->mutable_gpu_data());
	buffer_blob_.ReshapeLike(*top[ 0 ]);
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, number_, channels_, 1, Dtype(-1),
		sum_multiplier_num_.gpu_data(), batch_dx_.gpu_data(), Dtype(0),
		buffer_blob_.mutable_gpu_data());
	caffe_gpu_mul(top[ 0 ]->count(), top[ 0 ]->gpu_data(), buffer_blob_.gpu_data(), top[ 0 ]->mutable_gpu_data());
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, number_, channels_, 1, Dtype(1),
		sum_multiplier_num_.gpu_data(), batch_ex_.gpu_data(), Dtype(0),
		buffer_blob_.mutable_gpu_data());
	caffe_gpu_add(top[ 0 ]->count(), top[ 0 ]->gpu_data(), buffer_blob_.gpu_data(), top[ 0 ]->mutable_gpu_data());
}

template <typename Dtype>
void GaussianSampleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(GaussianSampleLayer);

}  // namespace caffe

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gaussian_noise_layer.hpp"

namespace caffe {

template <typename Dtype>
void GaussianNoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_scale(count, Dtype(0.5), bottom[1]->gpu_data(), top[0]->mutable_gpu_data());
  caffe_gpu_exp(count, top[0]->gpu_data(), top[0]->mutable_gpu_data());
  caffe_gpu_rng_gaussian(count, Dtype(0), Dtype(1), noise_.mutable_gpu_data());
  caffe_gpu_mul(count, noise_.gpu_data(), top[0]->gpu_data(), noise_.mutable_gpu_data());
  caffe_gpu_add(count, noise_.gpu_data(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void GaussianNoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  caffe_copy(count, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
  caffe_gpu_mul(count, top[0]->gpu_diff(), noise_.gpu_data(), bottom[1]->mutable_gpu_diff());
  caffe_gpu_scal(count, Dtype(0.5), bottom[1]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(GaussianNoiseLayer);

}  // namespace caffe

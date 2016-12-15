#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/gaussian_noise_layer.hpp"

namespace caffe {

template <typename Dtype>
void GaussianNoiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void GaussianNoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK(bottom[0]->shape() == bottom[1]->shape())
      << "Inputs must have the same shape.";
  top[0]->ReshapeLike(*bottom[0]);
  noise_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GaussianNoiseLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_cpu_scale(count, Dtype(0.5), bottom[1]->cpu_data(), top[0]->mutable_cpu_data());
  caffe_exp(count, top[0]->cpu_data(), top[0]->mutable_cpu_data());
  caffe_rng_gaussian(count, Dtype(0), Dtype(1), noise_.mutable_cpu_data());
  caffe_mul(count, noise_.cpu_data(), top[0]->cpu_data(), noise_.mutable_cpu_data());
  caffe_add(count, noise_.cpu_data(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void GaussianNoiseLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  caffe_copy(count, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
  caffe_mul(count, top[0]->cpu_diff(), noise_.cpu_data(), bottom[1]->mutable_cpu_diff());
  caffe_scal(count, Dtype(0.5), bottom[1]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(GaussianNoiseLayer);
#endif

INSTANTIATE_CLASS(GaussianNoiseLayer);
REGISTER_LAYER_CLASS(GaussianNoise);

}  // namespace caffe

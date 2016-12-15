#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/divergence_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void DivergenceLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_gpu_exp(count, bottom[1]->gpu_data(), sigma_.mutable_gpu_data());
  Dtype sum_sigma;
  caffe_gpu_dot(count, sigma_.gpu_data(), multiplier_.gpu_data(), &sum_sigma);
  Dtype dot;
  caffe_gpu_dot(count, bottom[0]->gpu_data(), bottom[0]->gpu_data(), &dot);
  Dtype sum_log_sigma;
  caffe_gpu_dot(count, bottom[1]->gpu_data(), multiplier_.gpu_data(), &sum_log_sigma);
  Dtype loss = sum_sigma + dot - sum_log_sigma - count;
  loss = loss / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

}

template <typename Dtype>
void DivergenceLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();

  caffe_gpu_scale(count, alpha, bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());

  caffe_copy(count, sigma_.gpu_data(), bottom[1]->mutable_gpu_diff());
  caffe_gpu_add_scalar(count, Dtype(-1), bottom[1]->mutable_gpu_diff());
  caffe_gpu_scal(count, Dtype(0.5) * alpha, bottom[1]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(DivergenceLossLayer);

}  // namespace caffe

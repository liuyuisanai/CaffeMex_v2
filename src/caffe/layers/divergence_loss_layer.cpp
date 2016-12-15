#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/divergence_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void DivergenceLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK(bottom[0]->shape() == bottom[1]->shape())
      << "Inputs must have the same shape.";
  sigma_.ReshapeLike(*bottom[0]);
  multiplier_.ReshapeLike(*bottom[0]);
  caffe_set(multiplier_.count(), Dtype(1), multiplier_.mutable_cpu_data());
}

template <typename Dtype>
void DivergenceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const int count = bottom[0]->count();
  caffe_exp(count, bottom[1]->cpu_data(), sigma_.mutable_cpu_data());
  Dtype loss = caffe_cpu_dot(count, sigma_.cpu_data(), multiplier_.cpu_data())
      + caffe_cpu_dot(count, bottom[0]->cpu_data(), bottom[0]->cpu_data())
      - caffe_cpu_dot(count, bottom[1]->cpu_data(), multiplier_.cpu_data()) - count;
  loss = loss / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DivergenceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const int count = bottom[0]->count();
  const Dtype alpha = top[0]->cpu_diff()[0] / bottom[0]->num();

  caffe_cpu_scale(count, alpha, bottom[0]->cpu_data(), bottom[0]->mutable_cpu_diff());

  caffe_copy(count, sigma_.cpu_data(), bottom[1]->mutable_cpu_diff());
  caffe_add_scalar(count, Dtype(-1), bottom[1]->mutable_cpu_diff());
  caffe_scal(count, Dtype(0.5) * alpha, bottom[1]->mutable_cpu_diff());
}

#ifdef CPU_ONLY
STUB_GPU(DivergenceLossLayer);
#endif

INSTANTIATE_CLASS(DivergenceLossLayer);
REGISTER_LAYER_CLASS(DivergenceLoss);

}  // namespace caffe

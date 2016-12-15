#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/upsample_layer.hpp"

namespace caffe {

template <typename Dtype>
void UpsampleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  UpsampleParameter upsample_param = this->layer_param_.upsample_param();
  CHECK((!upsample_param.has_scale() && upsample_param.has_scale_h() && upsample_param.has_scale_w()) ||
    (upsample_param.has_scale() && !upsample_param.has_scale_h() && !upsample_param.has_scale_w()));

  if (upsample_param.has_scale()) {
    scale_h_ = scale_w_ = upsample_param.scale();
    CHECK_GT(scale_h_, 1);
  } else {
    scale_h_ = upsample_param.scale_h();
    scale_w_ = upsample_param.scale_w();
    CHECK_GT(scale_h_, 1);
    CHECK_GT(scale_w_, 1);
  }
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  upsample_h_ = bottom[0]->height() * scale_h_;
  upsample_w_ = bottom[0]->width() * scale_w_;

  upsample_index_.Reshape(1, 1, height_, width_);
  Dtype *upsample_index_data = upsample_index_.mutable_cpu_data();
  int offset_h = (scale_h_ - 1) / 2;
  int offset_w = (scale_w_ - 1) / 2;
  for (int i = 0; i < height_; ++i) {
    for (int j = 0; j < width_; ++j) {
      int up_i = i * scale_h_ + offset_h;
      int up_j = j * scale_w_ + offset_w;
      upsample_index_data[i * width_ + j] = up_i * upsample_w_ + up_j;
    }
  }

  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), upsample_h_,
      upsample_w_);
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* upsample_index_data = upsample_index_.cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // Initialize
  const int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_data);
  // The main loop
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int c = 0; c < bottom[0]->channels(); ++c) {
      for (int i = 0; i < height_ * width_; ++i) {
        const int idx = static_cast<int>(upsample_index_data[i]);
        top_data[idx] = bottom_data[i];
      }
      // compute offset
      bottom_data += bottom[0]->offset(0, 1);
      top_data += top[0]->offset(0, 1);
    }
  }
}

template <typename Dtype>
void UpsampleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* upsample_index_data = upsample_index_.cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    const int bottom_count = bottom[0]->count();
    caffe_set(bottom_count, Dtype(0), bottom_diff);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < bottom[0]->channels(); ++c) {
        for (int i = 0; i < height_ * width_; ++i) {
          const int idx = static_cast<int>(upsample_index_data[i]);
          bottom_diff[i] = top_diff[idx];
        }
        // compute offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(UpsampleLayer);
#endif

INSTANTIATE_CLASS(UpsampleLayer);
REGISTER_LAYER_CLASS(Upsample);

}  // namespace caffe

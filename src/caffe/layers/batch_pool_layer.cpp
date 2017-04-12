#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/batch_pool_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void BatchPoolLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_LE(bottom.size(), 2);
  has_weight = bottom.size() == 2 ? 1 : 0;
}

template <typename Dtype>
void BatchPoolLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  batch_size_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (this->layer_param_.batch_pool_param().pool() == BatchPoolParameter_PoolMethod_MAX)
  {
    maxpool_mask.clear();
    maxpool_mask.resize(channels_*height_*width_);
  }
  if (has_weight)
  {
    CHECK_EQ(bottom[0]->num(), bottom[1]->count()) << "Batch number and weight mismatch!";
  }
  std::vector<int> shape = bottom[0]->shape();
  shape[0] = 1;
  top[0]->Reshape(shape);
}

template <typename Dtype>
void BatchPoolLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int slice_cnt = top[0]->count();
  if (has_weight){
    const Dtype* weight = bottom[1]->cpu_data();
    caffe_set(slice_cnt, Dtype(0.0), top_data);
    for (int cnt_t = 0; cnt_t < slice_cnt; ++cnt_t)
    {
      for (int num_t = 0; num_t < batch_size_; ++num_t)
      {
        top_data[cnt_t] += weight[num_t] * bottom_data[num_t*slice_cnt+cnt_t];
      }
    }
  }else{
    switch (this->layer_param_.batch_pool_param().pool()) {
    case BatchPoolParameter_PoolMethod_MAX:
      caffe_set(slice_cnt, Dtype(-FLT_MAX), top_data);
      for (int cnt_t = 0; cnt_t < slice_cnt; ++cnt_t){
        for (int num_t = 0; num_t < batch_size_; ++num_t){
          if (top_data[cnt_t] < bottom_data[num_t*slice_cnt+cnt_t]){
            top_data[cnt_t] = bottom_data[num_t*slice_cnt+cnt_t];
            maxpool_mask[cnt_t] = num_t;
          }
        }
      }
      break;
    case BatchPoolParameter_PoolMethod_AVE:
      for (int cnt_t = 0; cnt_t < slice_cnt; ++cnt_t){
        Dtype sum = 0;
        for (int num_t = 0; num_t < batch_size_; ++num_t){
          sum += bottom_data[num_t*slice_cnt+cnt_t];
        }
        top_data[cnt_t] = sum / batch_size_;
      }
      break;
    case BatchPoolParameter_PoolMethod_STOCHASTIC:
      NOT_IMPLEMENTED;
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
    }
  }
}

template <typename Dtype>
void BatchPoolLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    /*LOG(INFO) << "Check point 0";
  if (!propagate_down[0]) {
    LOG(INFO) << "Check point ERR";
    for (int i = 0; i < bottom[0]->count(); ++i)
      LOG(INFO) << "Bottom_diff[0]:" << bottom[0]->cpu_diff()[i];
    for (int i = 0; i < bottom[1]->count(); ++i)
      LOG(INFO) << "Bottom_diff[1]:" << bottom[1]->cpu_diff()[i];
    return;
  }*/
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  int slice_cnt = top[0]->count();
  if (has_weight){
    // LOG(INFO) << "Check point 1";
    Dtype* weight_diff = bottom[1]->mutable_cpu_diff();
    // diff for input
    for (int num_t = 0; num_t < batch_size_ ; ++num_t)
    {
      caffe_copy<Dtype>(slice_cnt, top_diff, bottom_diff + num_t * slice_cnt);
      caffe_scal<Dtype>(slice_cnt, bottom[1]->cpu_data()[num_t], bottom_diff + num_t * slice_cnt);
      // LOG(INFO) << "bottom_diff[" << num_t << "] -> " << bottom_diff[num_t * slice_cnt];
    }
    // diff for weight
    for (int num_t = 0; num_t < batch_size_; ++num_t)
    {
      weight_diff[num_t] = caffe_cpu_dot(slice_cnt, bottom_data + num_t * slice_cnt, top_diff);
      // LOG(INFO) << "weight_diff[" << num_t << "] -> " << weight_diff[num_t];
    }
  }else{
    switch (this->layer_param_.batch_pool_param().pool()) {
    case BatchPoolParameter_PoolMethod_MAX:
      for (int cnt_t = 0; cnt_t < slice_cnt; ++cnt_t){
        bottom_diff[maxpool_mask[cnt_t]*slice_cnt+cnt_t] = top_diff[cnt_t];
      }
      break;
    case BatchPoolParameter_PoolMethod_AVE:
      for (int num_t = 0; num_t < batch_size_; ++num_t){
        caffe_copy<Dtype>(slice_cnt, top_diff, bottom_diff);
        caffe_scal<Dtype>(slice_cnt, batch_size_, bottom_diff);
        bottom_diff += slice_cnt;
      }
      break;
    case BatchPoolParameter_PoolMethod_STOCHASTIC:
      NOT_IMPLEMENTED;
      break;
    default:
      LOG(FATAL) << "Unknown pooling method.";
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(BatchPoolLayer);
#endif

INSTANTIATE_CLASS(BatchPoolLayer);
REGISTER_LAYER_CLASS(BatchPool);

}  // namespace caffe

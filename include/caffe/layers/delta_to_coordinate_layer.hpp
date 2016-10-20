#ifndef CAFFE_DELTA_TO_COORDINATE_LAYER_HPP_
#define CAFFE_DELTA_TO_COORDINATE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <math.h>

namespace caffe {

template <typename Dtype>
class DeltaToCoordinateLayer : public Layer<Dtype> {
 public:
  explicit DeltaToCoordinateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DeltaToCoordinate"; }
  virtual inline int ExactNumBottomBlobs() const { return 6; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
//  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
//  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
//     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  float thresh_, anchor_bias_;
  int stride_, valid_num_, w_, h_;
};

}  // namespace caffe

#endif  // CAFFE_DELTA_TO_COORDINATE_LAYER_HPP_

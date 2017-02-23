// ------------------------------------------------------------------
// R-FCN
// Copyright (c) 2016 Microsoft
// Licensed under The MIT License [see r-fcn/LICENSE for details]
// Written by Yi Li
// ------------------------------------------------------------------

#include <cfloat>

#include <string>
#include <utility>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/cls_ohem_layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

  template <typename Dtype>
  void ClsOHEMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    topk_ = this->layer_param_.cls_ohem_param().topk();
    CHECK_GT(topk_, 0);
	ignore_label_ = this->layer_param_.cls_ohem_param().ignore_label();
  }

  template <typename Dtype>
  void ClsOHEMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    num_ = bottom[0]->num();
	CHECK_EQ(bottom[ 1 ]->channels(), 1);
	CHECK_EQ(bottom[ 0 ]->channels(), 1);
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	top[0]->Reshape(num_, 1, height_, width_);
	top[1]->Reshape(num_, 1, height_, width_);
  }

  template <typename Dtype>
  void ClsOHEMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    NOT_IMPLEMENTED;
  }

  template <typename Dtype>
  void ClsOHEMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    NOT_IMPLEMENTED;
  }


#ifdef CPU_ONLY
  STUB_GPU(ClsOHEMLayer);
#endif

  INSTANTIATE_CLASS(ClsOHEMLayer);
  REGISTER_LAYER_CLASS(ClsOHEM);

}  // namespace caffe
